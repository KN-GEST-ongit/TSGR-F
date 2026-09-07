"""Frozen-routing model builder and held-out ranking evaluator for TSGR-F 0.40.

The builder is TRAIN-only.  It never accepts a test-processing report or ground-truth
report.  The evaluator consumes only immutable per-fold artifacts produced by the
builder and never refits weights, feature subsets, or thresholds.
"""
from __future__ import annotations

import csv, hashlib, json, math, shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from tsgr.dataset.contract import resolve_dataset_root_from_plan
from tsgr.evaluation.classifier import TSGRFClassifier
from tsgr.evaluation.experiment_evaluator import _find_model_set, _load_run_features
from tsgr.evaluation.image_feature_support import build_image_feature_matrix, image_features
from tsgr.evaluation.io_utils import read_csv_rows, write_workbook
from tsgr.evaluation.ranking_support import class_fused_score, global_inverse_within_weights, loo_scores, normalize_positive_weights
from tsgr.evaluation.routing_support import (
    DEFAULT_IY_FEATURE_IDS,
    apply_c_route,
    apply_fixed_iy_gate,
    apply_os_threshold,
    binary_conformity_predictions,
    c_one_vs_rest_top_indices,
    feature_indices,
    merge_best_threshold_plateaus,
    os_threshold_intervals,
    select_threshold_plateau,
)
from tsgr.evaluation.training_support import LandmarkArchiveCache, fold_run_map, load_fold_training_samples
from tsgr.utils.serialization import write_json
from tsgr.visualization.landmark_overlay import draw_landmarks_overlay

SCHEMA = "tsgrf_frozen_routing_model_v40"
DEFAULT_ALPHA = 0.65
DEFAULT_C_TOP_N = 5


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str,Any]], fields: Sequence[str] | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fields),extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def _safe_scale(values: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    center=np.median(values,axis=0)
    mad=np.median(np.abs(values-center[None,:]),axis=0)*1.4826
    pos=mad[mad>0]
    floor=max(float(np.median(pos))*1e-3 if pos.size else 1e-6,1e-9)
    return center,np.maximum(mad,floor)


def _pair_inverse_within_weights(values: np.ndarray, labels: np.ndarray, a: int, b: int) -> np.ndarray:
    vs=[]
    for c in (a,b):
        x=values[labels==c]
        vs.append(np.var(x,axis=0,ddof=1))
    within=np.mean(np.stack(vs),axis=0)
    pos=within[within>0]; floor=max(float(np.median(pos))*1e-3 if pos.size else 1e-6,1e-9)
    return normalize_positive_weights(1.0/np.maximum(within,floor))


def _query_class_scores(q: np.ndarray, ref: np.ndarray, labels: np.ndarray, weights: np.ndarray, class_count:int) -> np.ndarray:
    q=np.asarray(q,float); ref=np.asarray(ref,float); w=np.asarray(weights,float)
    out=np.full((len(q),class_count),np.inf,float)
    for c in range(class_count):
        rr=ref[labels==c]
        d=q[:,None,:]-rr[None,:,:]
        out[:,c]=np.sqrt(np.min(np.sum(d*d*w[None,None,:],axis=2),axis=1))
    return out


def _query_fused_score(q3,q2,r3,r2,labels,cls,w3,w2,alpha):
    idx=np.flatnonzero(labels==cls); a=float(alpha)
    ww3=np.asarray(w3,float)/float(np.sum(w3)); ww2=np.asarray(w2,float)/float(np.sum(w2))
    d3=q3[:,None,:]-r3[idx][None,:,:]; d2=q2[:,None,:]-r2[idx][None,:,:]
    s3=np.sum(d3*d3*ww3[None,None,:],axis=2); s2=np.sum(d2*d2*ww2[None,None,:],axis=2)
    return np.sqrt(np.maximum(np.min((1-a)*s3+a*s2,axis=1),0.0))


def _query_pair_scores(q,ref,labels,a,b,weights):
    scores=np.empty((len(q),2),float)
    for col,c in enumerate((a,b)):
        rr=ref[labels==c]; d=q[:,None,:]-rr[None,:,:]
        scores[:,col]=np.sqrt(np.min(np.sum(d*d*weights[None,None,:],axis=2),axis=1))
    return scores


def _top2_gate(scores: np.ndarray,a:int,b:int)->np.ndarray:
    top=np.sort(np.argsort(scores,axis=1)[:,:2],axis=1); lo,hi=sorted((a,b))
    return (top[:,0]==lo)&(top[:,1]==hi)


def build_frozen_routing_models(plan_dir: str|Path, *, output_dir: str|Path, branch_name:str,
                                feature_set:str='compact', orientation_mode:str='camera_aware',
                                scenarios:set[str]|None=None, folds:set[str]|None=None,
                                os_alpha:float=DEFAULT_ALPHA,
                                iy_feature_ids:Sequence[str]=DEFAULT_IY_FEATURE_IDS,
                                c_top_n:int=DEFAULT_C_TOP_N, model_variant:str|None=None,
                                dataset_root_override:str|Path|None=None, progress:bool=True) -> Path:
    plan=Path(plan_dir); dataset=resolve_dataset_root_from_plan(plan, override=dataset_root_override); out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    summary=[]; guard=[]
    for fold_json in sorted(plan.glob('*/*/fold.json')):
        meta=json.loads(fold_json.read_text(encoding='utf-8')); scenario=str(meta['scenario']); fold_id=str(meta['fold_id'])
        if scenarios and scenario not in scenarios: continue
        if folds and fold_id not in folds: continue
        if str(meta.get('fold_status','active'))!='active': continue
        fd=fold_json.parent; model_set=_find_model_set(fd,model_variant)
        clf=TSGRFClassifier(model_set,branch_name=branch_name,feature_set=feature_set,distance_metric='weighted_euclidean',orientation_mode=orientation_mode)
        samples=load_fold_training_samples(fd,branch_name=branch_name,dataset_root=dataset,gesture_ids=clf.gesture_ids)
        labels=np.asarray(samples.gesture_index,np.int64)
        x3=np.asarray(samples.values[:,clf.feature_mask],float)
        ids3=tuple(clf.space.feature_ids[int(i)] for i in clf.feature_indices.tolist())
        w3=normalize_positive_weights(np.asarray(clf.space.inverse_within_weights,float)[clf.feature_mask])
        cache=LandmarkArchiveCache(); img=build_image_feature_matrix(samples,run_map=fold_run_map(fd,dataset),cache=cache)
        x2=np.asarray(img.values,float); w2=global_inverse_within_weights(x2,labels,len(clf.gesture_ids))
        combined=np.column_stack((x3,x2)); combined_ids=tuple(ids3)+tuple(img.feature_ids)
        scores,_=loo_scores(x3,labels,len(clf.gesture_ids),w3); base=np.argmin(scores,axis=1)
        gi={g:i for i,g in enumerate(clf.gesture_ids)}
        for g in ('A','C','I','Y','O','S'):
            if g not in gi: raise RuntimeError(f'Missing required gesture {g} in {scenario}/{fold_id}')
        so=class_fused_score(x3,x2,labels,gi['O'],w3,w2,float(os_alpha)); ss=class_fused_score(x3,x2,labels,gi['S'],w3,w2,float(os_alpha))
        intervals=os_threshold_intervals(scores,base,labels,so,ss,class_o=gi['O'],class_s=gi['S'])
        plateau=select_threshold_plateau(merge_best_threshold_plateaus(intervals))
        if plateau is None: raise RuntimeError('Unable to learn O/S threshold plateau from TRAIN.')
        tau=float(plateau['midpoint'])
        pred_os,_,_,_,_=apply_os_threshold(scores,base,labels,so,ss,class_o=gi['O'],class_s=gi['S'],threshold=tau)
        iy_idx=feature_indices(combined_ids,iy_feature_ids); iy_w=_pair_inverse_within_weights(combined[:,iy_idx],labels,gi['I'],gi['Y'])
        # Training LOO I/Y diagnostic uses existing routine; inference uses stored pair-local weights.
        pred_iy,iy_metrics,_=apply_fixed_iy_gate(scores,pred_os,base,combined,labels,class_i=gi['I'],class_y=gi['Y'],selected_features=iy_idx)
        c_idx=c_one_vs_rest_top_indices(combined,labels,gi['C'],int(c_top_n)); c_ids=tuple(combined_ids[int(i)] for i in c_idx)
        pair_pred,_,c_bin_err,c_bin_acc=binary_conformity_predictions(combined,labels,gi['A'],gi['C'],c_idx,method='robust_z_rms')
        pred_final,c_metrics,_=apply_c_route(pred_iy,base,scores,pair_pred,labels,class_a=gi['A'],class_c=gi['C'],policy='top1_a_rescue_c_only')
        ca_center,ca_scale=_safe_scale(combined[labels==gi['A']][:,c_idx]); cc_center,cc_scale=_safe_scale(combined[labels==gi['C']][:,c_idx])
        fold_out=out/scenario/fold_id; fold_out.mkdir(parents=True,exist_ok=True)
        arrays=fold_out/'model_arrays.npz'
        np.savez_compressed(arrays,train3=x3,train2=x2,labels=labels,w3=w3,w2=w2,
                            iy_indices=iy_idx,iy_weights=iy_w,c_indices=c_idx,
                            c_a_center=ca_center,c_a_scale=ca_scale,c_c_center=cc_center,c_c_scale=cc_scale,
                            feature_indices=np.asarray(clf.feature_indices,np.int64),
                            feature_ids3=np.asarray(ids3),image_feature_ids=np.asarray(img.feature_ids),combined_feature_ids=np.asarray(combined_ids))
        model={
            'schema_version':SCHEMA,'created_at_utc':datetime.now(timezone.utc).isoformat(),'scenario':scenario,'fold_id':fold_id,
            'branch_name':branch_name,'feature_set':feature_set,'orientation_mode':orientation_mode,'gesture_ids':list(clf.gesture_ids),'reference_model_variant':model_variant or 'reference',
            'training_sample_count':int(len(labels)),'arrays_file':'model_arrays.npz','arrays_sha256':_sha256(arrays),
            'global':{'method':'nearest_exemplar','metric':'inverse_within_weighted_euclidean'},
            'os_specialist':{'gate':'baseline_top2_exact_OS','alpha':float(os_alpha),'threshold':tau,'plateau':plateau,'threshold_source':'TRAIN_LOO'},
            'iy_specialist':{'gate':'baseline_top2_exact_IY','feature_ids':list(iy_feature_ids),'weights_source':'TRAIN_pair_inverse_within'},
            'c_rescue':{'gate':'baseline_top1_A_rescue_C_only','method':'robust_z_rms','feature_rule':f'C_vs_rest_Fisher_top{int(c_top_n)}','feature_ids':list(c_ids)},
            'leakage_contract':{'test_processing_read_during_build':False,'test_ground_truth_read_during_build':False,'all_fitted_parameters_source':'fold_TRAIN_only'}
        }
        write_json(fold_out/'model.json',model)
        err0=int(np.sum(base!=labels)); err1=int(np.sum(pred_os!=labels)); err2=int(np.sum(pred_iy!=labels)); err3=int(np.sum(pred_final!=labels))
        summary.append({'scenario':scenario,'fold_id':fold_id,'training_samples':len(labels),'baseline_errors':err0,'after_os_errors':err1,'after_iy_errors':err2,'final_train_loo_errors':err3,'final_train_loo_accuracy':float(np.mean(pred_final==labels)),'os_threshold':tau,'os_plateau_width':plateau['width'],'iy_binary_errors':iy_metrics.pair_errors,'c_binary_errors':c_bin_err,'c_binary_accuracy':c_bin_acc,'c_feature_ids':';'.join(c_ids)})
        guard.append({'scenario':scenario,'fold_id':fold_id,'phase':'BUILD_TRAIN_ONLY','test_processing_report_read':0,'ground_truth_report_read':0,'fit_on_test':0,'model_json':str((fold_out/'model.json').relative_to(out)).replace('\\','/')})
        if progress: print(f'[{scenario}/{fold_id}] TRAIN LOO final errors={err3} | tau_OS={tau:.6g}')
    _write_csv(out/'frozen_training_loo_summary.csv',summary); _write_csv(out/'leakage_guard.csv',guard)
    write_json(out/'frozen_model_build_report.json',{'schema_version':'tsgrf_frozen_routing_build_v40','created_at_utc':datetime.now(timezone.utc).isoformat(),'plan':str(plan),'dataset_root':str(dataset),'fold_count':len(summary),'reference_model_variant':model_variant or 'reference','notes':['TRAIN-only build. No test-processing report or ground truth is accepted by this API.','S1-developed architecture is frozen; fold-local weights, O/S threshold, and C Top-5 are learned only from TRAIN.']})
    return out/'frozen_model_build_report.json'


def _load_model(root:Path,scenario:str,fold_id:str)->tuple[dict[str,Any],dict[str,np.ndarray]]:
    d=root/scenario/fold_id; m=json.loads((d/'model.json').read_text(encoding='utf-8')); a=d/m['arrays_file']
    if _sha256(a)!=m.get('arrays_sha256'): raise ValueError(f'Frozen model checksum mismatch: {a}')
    with np.load(a,allow_pickle=False) as z: arr={k:np.asarray(z[k]) for k in z.files}
    return m,arr


def _run_image_features(run:Path, frame_indices:np.ndarray)->tuple[np.ndarray,np.ndarray,tuple[str,...]]:
    cache=LandmarkArchiveCache(); rows=[]; ok=[]; ids=None
    for fi in frame_indices.tolist():
        vals,fids=image_features(cache.get(run,int(fi),'raw_image_landmarks'))
        if vals:
            if ids is None: ids=tuple(fids)
            rows.append(vals); ok.append(True)
        else: rows.append(None); ok.append(False)
    if ids is None: return np.full((len(frame_indices),0),np.nan),np.zeros(len(frame_indices),bool),tuple()
    mat=np.full((len(rows),len(ids)),np.nan,float)
    for i,r in enumerate(rows):
        if r is not None: mat[i]=r
    return mat,np.asarray(ok,bool),ids


def _predict_frozen(m:dict[str,Any],a:dict[str,np.ndarray],qfull:np.ndarray,q2:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    gestures=tuple(m['gesture_ids']); gi={g:i for i,g in enumerate(gestures)}; fidx=a['feature_indices'].astype(int)
    q3=np.asarray(qfull[:,fidx],float); labels=a['labels'].astype(int); train3=a['train3']; train2=a['train2']; w3=a['w3']; w2=a['w2']
    scores=_query_class_scores(q3,train3,labels,w3,len(gestures)); base=np.argmin(scores,axis=1); pred_os=base.copy()
    so=_query_fused_score(q3,q2,train3,train2,labels,gi['O'],w3,w2,float(m['os_specialist']['alpha']))
    ss=_query_fused_score(q3,q2,train3,train2,labels,gi['S'],w3,w2,float(m['os_specialist']['alpha']))
    gate=_top2_gate(scores,gi['O'],gi['S']); pred_os[gate]=np.where((so-ss)[gate]<=float(m['os_specialist']['threshold']),gi['O'],gi['S'])
    pred_iy=pred_os.copy(); comb=np.column_stack((q3,q2)); train_comb=np.column_stack((train3,train2)); iy_idx=a['iy_indices'].astype(int); iyw=a['iy_weights']
    iy_scores=_query_pair_scores(comb[:,iy_idx],train_comb[:,iy_idx],labels,gi['I'],gi['Y'],iyw); gateiy=_top2_gate(scores,gi['I'],gi['Y']); pred_iy[gateiy]=np.where(iy_scores[gateiy,0]<=iy_scores[gateiy,1],gi['I'],gi['Y'])
    final=pred_iy.copy(); cidx=a['c_indices'].astype(int); x=comb[:,cidx]
    za=np.sqrt(np.mean(((x-a['c_a_center'])/a['c_a_scale'])**2,axis=1)); zc=np.sqrt(np.mean(((x-a['c_c_center'])/a['c_c_scale'])**2,axis=1))
    rescue=(base==gi['A'])&(zc<za); final[rescue]=gi['C']
    return base,pred_os,pred_iy,final,scores


def evaluate_frozen_routing(plan_dir:str|Path, *, frozen_model_dir:str|Path, test_processing_report:str|Path,
                            ground_truth_report:str|Path, output_dir:str|Path, branch_name:str,
                            scenarios:set[str]|None=None, folds:set[str]|None=None,
                            dataset_root_override:str|Path|None=None,
                            copy_problem_cases:bool=True, progress:bool=True)->Path:
    plan=Path(plan_dir); dataset=resolve_dataset_root_from_plan(plan, override=dataset_root_override); models=Path(frozen_model_dir); proc=Path(test_processing_report); gtroot=Path(ground_truth_report); out=Path(output_dir)
    status_csv=proc/'test_take_mediapipe_status.csv'; gt_csv=gtroot/'frame_ground_truth.csv'
    if not status_csv.is_file():
        raise ValueError(
            f"Invalid test-processing report: missing {status_csv}. "
            "Pass the concrete processing report directory, e.g. "
            "tsgr_dataset/reports/test_processing/image/high_recall/ignore_handedness/processing_<TAG> "
            "or tsgr_dataset/reports/test_processing/video/balanced/recovery_on/ignore_handedness/processing_<TAG>."
        )
    if not gt_csv.is_file():
        raise ValueError(
            f"Invalid ground-truth report: missing {gt_csv}. "
            "The canonical directory is tsgr_dataset/reports/test_ground_truth/ground_truth_<TAG>."
        )
    manifest_rows=read_csv_rows(status_csv)
    if not manifest_rows:
        raise ValueError(f"Empty test-processing manifest: {status_csv}")
    manifest={str(r['take_id']):r for r in manifest_rows}
    gt_rows=read_csv_rows(gt_csv)
    if not gt_rows:
        raise ValueError(f"Empty frame ground truth: {gt_csv}")
    gt_by={}
    for r in gt_rows: gt_by.setdefault(str(r['take_id']),{})[int(r['frame_index'])]=r
    selected_folds=[]; required_takes=set()
    for fold_json in sorted(plan.glob('*/*/fold.json')):
        fm=json.loads(fold_json.read_text(encoding='utf-8')); scenario=str(fm['scenario']); fold_id=str(fm['fold_id'])
        if scenarios and scenario not in scenarios: continue
        if folds and fold_id not in folds: continue
        if str(fm.get('fold_status','active'))!='active': continue
        try:
            _load_model(models,scenario,fold_id)
        except FileNotFoundError:
            continue
        selected_folds.append((fold_json,scenario,fold_id))
        required_takes.update(str(r['take_id']) for r in read_csv_rows(fold_json.parent/'test_takes.csv'))
    if not selected_folds:
        raise ValueError('No active folds with frozen models were selected for held-out evaluation.')
    missing_processing=sorted(required_takes-set(manifest))
    if missing_processing:
        sample=', '.join(missing_processing[:10])
        raise ValueError(
            f"Processing report {proc} does not cover the selected held-out folds: "
            f"missing {len(missing_processing)} of {len(required_takes)} required takes; first: {sample}. "
            "Check that the report is the full all-data IMAGE/VIDEO cache for the same dataset/tag."
        )
    missing_gt=sorted(required_takes-set(gt_by))
    if missing_gt:
        sample=', '.join(missing_gt[:10])
        raise ValueError(
            f"Ground-truth report {gtroot} does not cover the selected held-out folds: "
            f"missing {len(missing_gt)} of {len(required_takes)} required takes; first: {sample}."
        )
    out.mkdir(parents=True,exist_ok=False)
    frame_rows=[]; fold_rows=[]; gesture_rows=[]; guards=[]
    for fold_json,scenario,fold_id in selected_folds:
        fm=json.loads(fold_json.read_text(encoding='utf-8'))
        m,a=_load_model(models,scenario,fold_id)
        if str(m['branch_name'])!=branch_name: raise ValueError(f'Branch mismatch for {scenario}/{fold_id}')
        gestures=tuple(m['gesture_ids']); gi={g:i for i,g in enumerate(gestures)}; test_takes={str(r['take_id']) for r in read_csv_rows(fold_json.parent/'test_takes.csv')}
        total=covered=correct_base=correct_final=0; conf={}; per={g:[0,0,0] for g in gestures}
        for take in sorted(test_takes):
            mr=manifest.get(take)
            if mr is None: raise ValueError(f'Missing take {take} in processing report.')
            run=Path(str(mr.get('run_path',''))); run=run if run.is_absolute() else dataset/run
            fi,status,val=_load_run_features(run,branch_name); pos={int(v):i for i,v in enumerate(fi.tolist())}; imgmat,imgok,imgids=_run_image_features(run,fi)
            expected=tuple(str(x) for x in a['image_feature_ids'].tolist())
            if imgids and tuple(imgids)!=expected: raise ValueError(f'IMAGE feature schema mismatch for {take}')
            rows=[r for k,r in sorted(gt_by.get(take,{}).items()) if str(r.get('evaluation_state','')).startswith('GESTURE_')]
            for r in rows:
                total+=1; idx=pos.get(int(r['frame_index'])); gtg=str(r.get('gesture_id') or str(r.get('evaluation_state','')).removeprefix('GESTURE_')).upper(); per.setdefault(gtg,[0,0,0])[0]+=1
                baseg=finalg=''; input_ok=False
                if idx is not None and np.isfinite(val[idx]).all() and idx<len(imgok) and bool(imgok[idx]) and np.isfinite(imgmat[idx]).all():
                    input_ok=True; covered+=1; b,o,i,f,s=_predict_frozen(m,a,val[idx:idx+1],imgmat[idx:idx+1]); baseg=gestures[int(b[0])]; osg=gestures[int(o[0])]; iyg=gestures[int(i[0])]; finalg=gestures[int(f[0])]
                    correct_base+=int(baseg==gtg); correct_final+=int(finalg==gtg); per[gtg][1]+=int(baseg==gtg); per[gtg][2]+=int(finalg==gtg); conf[(gtg,finalg)]=conf.get((gtg,finalg),0)+1
                if not input_ok: osg=iyg=''
                frame_rows.append({'scenario':scenario,'fold_id':fold_id,'take_id':take,'public_subject_id':r.get('public_subject_id',''),'background':r.get('background',''),'gesture_id':gtg,'frame_index':r['frame_index'],'input_available':int(input_ok),'baseline_prediction':baseg,'after_os_prediction':osg,'after_iy_prediction':iyg,'final_prediction':finalg,'baseline_correct':int(input_ok and baseg==gtg),'after_os_correct':int(input_ok and osg==gtg),'after_iy_correct':int(input_ok and iyg==gtg),'final_correct':int(input_ok and finalg==gtg),'os_changed':int(input_ok and osg!=baseg),'iy_changed':int(input_ok and iyg!=osg),'c_rescue_changed':int(input_ok and finalg!=iyg)})
                if copy_problem_cases and input_ok and finalg!=gtg:
                    case=out/'problem_cases'/scenario/fold_id/f'{take}__frame_{int(r["frame_index"]):06d}__GT_{gtg}__PRED_{finalg}'; case.mkdir(parents=True,exist_ok=True)
                    rel=str(r.get('frame_relative_path','')); src=dataset/rel if rel else None
                    if src is not None and src.is_file():
                        shutil.copy2(src,case/f'query_original{src.suffix.lower() or ".jpg"}'); im=cv2.imread(str(src),cv2.IMREAD_COLOR); lm=LandmarkArchiveCache().get(run,int(r['frame_index']),'raw_image_landmarks')
                        if im is not None: cv2.imwrite(str(case/'query_image_landmarks.jpg'),draw_landmarks_overlay(im,lm,title=f'GT {gtg} | final {finalg}'))
                    write_json(case/'summary.json',frame_rows[-1])
        fold_rows.append({'scenario':scenario,'fold_id':fold_id,'gt_gesture_frames':total,'complete_input_frames':covered,'feature_coverage':covered/total if total else float('nan'),'baseline_conditional_accuracy':correct_base/covered if covered else float('nan'),'final_conditional_accuracy':correct_final/covered if covered else float('nan'),'baseline_end_to_end_accuracy':correct_base/total if total else float('nan'),'final_end_to_end_accuracy':correct_final/total if total else float('nan'),'final_errors_given_input':covered-correct_final})
        for g,(n,bc,fc) in per.items(): gesture_rows.append({'scenario':scenario,'fold_id':fold_id,'gesture_id':g,'gt_frames':n,'baseline_correct':bc,'final_correct':fc,'baseline_accuracy_given_input':bc/n if n else float('nan'),'final_accuracy_given_input':fc/n if n else float('nan')})
        guards.append({'scenario':scenario,'fold_id':fold_id,'phase':'HELD_OUT_EVALUATION','model_refit_during_evaluation':0,'threshold_refit_on_test':0,'feature_selection_on_test':0,'ground_truth_used_for_scoring_only':1})
        if progress: print(f'[{scenario}/{fold_id}] coverage={covered}/{total} | final={correct_final}/{covered} ({(correct_final/covered if covered else 0):.4f})')
    _write_csv(out/'frame_predictions.csv',frame_rows); _write_csv(out/'fold_summary.csv',fold_rows); _write_csv(out/'gesture_summary.csv',gesture_rows); _write_csv(out/'leakage_guard.csv',guards)
    # Scenario macro summaries.
    scen=[]
    for s in sorted({r['scenario'] for r in fold_rows}):
        rr=[r for r in fold_rows if r['scenario']==s]; tot=sum(r['gt_gesture_frames'] for r in rr); cov=sum(r['complete_input_frames'] for r in rr); err=sum(r['final_errors_given_input'] for r in rr)
        scen.append({'scenario':s,'fold_count':len(rr),'gt_gesture_frames':tot,'complete_input_frames':cov,'feature_coverage':cov/tot if tot else float('nan'),'final_conditional_accuracy':1-err/cov if cov else float('nan'),'final_errors_given_input':err})
    _write_csv(out/'scenario_summary.csv',scen)
    write_workbook(out/'frozen_routing_generalization.xlsx',{'folds':(list(fold_rows[0]) if fold_rows else [],fold_rows),'scenarios':(list(scen[0]) if scen else [],scen),'gestures':(list(gesture_rows[0]) if gesture_rows else [],gesture_rows),'errors':(list(frame_rows[0]) if frame_rows else [],[r for r in frame_rows if r.get('input_available') and not r.get('final_correct')])})
    report={'schema_version':'tsgrf_frozen_routing_generalization_v40','created_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_model_dir':str(models),'test_processing_report':str(proc),'ground_truth_report':str(gtroot),'fold_count':len(fold_rows),'notes':['Held-out evaluator does not fit or tune model parameters.','Metrics here validate gesture-class ranking on GT gesture frames and do not replace acceptance/no-hand evaluation.','Feature coverage is reported separately; missing input counts as an end-to-end gesture-frame error.']}
    write_json(out/'frozen_routing_generalization_report.json',report); return out/'frozen_routing_generalization_report.json'
