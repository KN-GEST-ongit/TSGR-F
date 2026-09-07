from pathlib import Path
from tsgr.utils.run_paths import latest_run_directory,resolve_results_path

def test_latest_run_is_resolved(tmp_path:Path):
    for name in ('run_001','run_002'):
        p=tmp_path/'runs'/name; p.mkdir(parents=True); (p/'frame_results.jsonl').write_text('{}\n')
    assert latest_run_directory(tmp_path).name=='run_002'; assert resolve_results_path(tmp_path).name=='frame_results.jsonl'
