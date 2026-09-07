# TSGR-F

**Trustworthy Static Gesture Recognition Framework** is a research framework for interpretable recognition and evaluation of 17 static hand gestures. The pipeline uses MediaPipe hand landmarks, explicit geometric features, a nearest-exemplar ranking model, selective pair-specific routing, and a separate acceptance/rejection stage.

The public repository contains the final research pipeline used for the reported experiments. Development-only model-selection audits, author-side dataset import tools, manuscript-generation utilities, private acquisition metadata, and local machine paths are intentionally excluded.

## Repository

Project repository: <https://github.com/KN-GEST-ongit/TSGR-F>

The code package is published as `tsgr-framework` and imported as:

```python
import tsgr
```

## Reference environment

The reported experiments were developed and validated with **CPython 3.12.10**. The package declares Python `>=3.12,<3.13`; other interpreter versions are not claimed as validated.

The external SGRF comparison toolbox uses a separate **CPython 3.11** environment. Do not install its TensorFlow-based dependency stack into the main TSGR-F environment.

## Installation

From a local clone:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

From GitHub:

```bash
pip install git+https://github.com/KN-GEST-ongit/TSGR-F.git
```

After the first PyPI release:

```bash
pip install tsgr-framework
```

For an environment matching the published experiments as closely as possible:

```powershell
python -m pip install -r .\requirements-lock.txt
python -m pip install -e . --no-deps
```

Download the MediaPipe Hand Landmarker model:

```powershell
tsgr-download-model
```

## Single-image inference

After the MediaPipe model, fixed routing models, and matching TRAIN-only acceptance calibration are available, the final IMAGE pipeline can be called through the high-level API:

```python
from tsgr.inference import TSGRFImageRecognizer

with TSGRFImageRecognizer(
    mediapipe_model="models/hand_landmarker.task",
    routing_model_root="results/fixed_routing_models",
    acceptance_root="results/acceptance",
    scenario="S1_ALL_IN_DOMAIN",
    fold_id="all",
) as recognizer:
    result = recognizer.predict("example.jpg")
    print(result.label)
```

`result.label` is the accepted gesture identifier, `NO_GESTURE`, or `MISSING_HAND`. The result object also exposes the routed candidate, route-specific score and threshold, detector status, and global Top-3 ranking. See `examples/single_image_prediction.py`.

## Dataset

The public dataset is distributed as assets of a GitHub Release in the same repository. It is intentionally not stored in Git history or installed with the Python package.

The release is divided by public subject identifier:

```text
tsgr_dataset_v1.0__P01.zip
tsgr_dataset_v1.0__P02.zip
tsgr_dataset_v1.0__P03.zip
tsgr_dataset_v1.0__P04.zip
tsgr_dataset_v1.0__P05.zip
tsgr_dataset_v1.0__manifest.zip
```

The five subject archives contain media. The manifest archive contains `annotations.csv`, `annotations.json`, SHA-256 checksums, and release metadata.

Download and assemble the complete dataset:

```powershell
tsgr-download-dataset --output-dir .\data\tsgr_dataset
```

Install already downloaded release ZIP files without network access:

```powershell
tsgr-download-dataset `
  --archive-dir .\dataset_release `
  --output-dir .\data\tsgr_dataset
```

Install selected subject packages only:

```powershell
tsgr-download-dataset `
  --subject P01 `
  --subject P03 `
  --output-dir .\data\tsgr_dataset
```

The complete reconstructed dataset has the public structure:

```text
tsgr_dataset/
├── training/
│   ├── P01/
│   ├── P02/
│   ├── P03/
│   ├── P04/
│   └── P05/
├── testing/
│   ├── P01/
│   ├── P02/
│   ├── P03/
│   ├── P04/
│   └── P05/
├── annotations.csv
└── annotations.json
```

Validate the assembled dataset before processing:

```powershell
tsgr-audit-dataset .\data\tsgr_dataset --require-videos --strict
```

See [docs/DATASET.md](docs/DATASET.md) for the public contract.

## Reviewer one-script reproduction

The canonical reviewer workflow is maintained as a single resumable PowerShell script. Place the six dataset release ZIP files in one directory and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\reproduce_reference.ps1 `
  -DatasetPackageDir ".\dataset_release" `
  -WorkspaceRoot ".\reproduction_workspace"
```

The script creates or reuses a dedicated Python 3.12.10 virtual environment, installs the public repository, assembles and audits the released dataset, downloads the MediaPipe model, and reproduces the core TRAIN/IMAGE/VIDEO, ground-truth, S1-S4, fold-model, fixed-routing, acceptance, and end-to-end stages. Completed checkpoints are reused automatically, so an interrupted reproduction can be resumed by running the same command again. Use `-Force` only when the complete workspace should be rebuilt from scratch.

For staged diagnostics, `-ThroughStage` accepts `dataset`, `processing`, `ground_truth`, `folds`, `models`, `routing`, `acceptance`, or `evaluation`. The no-argument terminal stage is `all`.

The script is the canonical executable reproduction recipe and will be extended together with the public release as additional publication checks are finalized. The individual commands below remain documented for inspection and debugging.

## Reproducing the TSGR-F pipeline

The commands below use only relative paths. The exact output directory names may be changed without altering the method.

Set convenient local variables:

```powershell
$DATASET = ".\data\tsgr_dataset"
$RESULTS = ".\results"
$MODEL = ".\models\hand_landmarker.task"
$BRANCH = "raw.wrist_middle_mcp"
```

> **Reference-policy note:** the published IMAGE and VIDEO reference results use `--handedness-policy ignore_handedness`. Raw and effective MediaPipe handedness diagnostics are still recorded; handedness is simply removed from the classification acceptance gate. `reject_left` remains available as a separate system-policy ablation and must not be substituted when reproducing the published reference results.

### 1. Process training photographs

```powershell
tsgr-process-training $DATASET `
  --model $MODEL `
  --all-data `
  --detection-profile high_recall `
  --report-name reference_training `
  --workers 16 `
  --progress
```

### 2. Process test data in IMAGE mode

```powershell
tsgr-process-testing $DATASET `
  --model $MODEL `
  --test-mode image `
  --handedness-policy ignore_handedness `
  --all-data `
  --minimum-success-rate 0.95 `
  --detection-profile high_recall `
  --no-video-recovery `
  --report-name reference_image `
  --workers 16 `
  --progress `
  --no-fail-fast
```

### 3. Process test data in VIDEO mode

```powershell
tsgr-process-testing $DATASET `
  --model $MODEL `
  --test-mode video `
  --handedness-policy ignore_handedness `
  --all-data `
  --fallback-fps 30 `
  --minimum-success-rate 0.95 `
  --detection-profile balanced `
  --video-recovery `
  --video-recovery-after 1 `
  --report-name reference_video `
  --workers 16 `
  --progress `
  --no-fail-fast
```

### 4. Build the detector-derived hand-presence reference

```powershell
tsgr-build-hand-presence-reference $DATASET `
  --model $MODEL `
  --output-dir "$RESULTS\ground_truth\hand_presence" `
  --workers 16 `
  --progress
```

### 5. Build the frame-level evaluation ground truth

```powershell
tsgr-analyze-test-ground-truth $DATASET `
  --hand-presence-reference "$RESULTS\ground_truth\hand_presence" `
  --output-dir "$RESULTS\ground_truth\final"
```

### 6. Generate S1-S4 folds

```powershell
tsgr-generate-folds $DATASET `
  --output-dir "$RESULTS\experiment_plan" `
  --workers 8 `
  --progress
```

The four evaluation settings are:

- `S1_ALL_IN_DOMAIN` — all subjects and backgrounds are represented in training;
- `S2_LOBO` — leave one background out;
- `S3_LOSO` — leave one subject out;
- `S4_LOSO_BACKGROUND` — leave one subject and one background out simultaneously.

### 7. Build fold-local reference models from TRAIN

The experiment plan deliberately does not retain an author-specific absolute dataset path. Fold manifests contain the processed TRAIN runs needed for model aggregation.

```powershell
tsgr-build-fold-models "$RESULTS\experiment_plan" `
  --all-scenarios `
  --all-folds `
  --branch $BRANCH `
  --compact-correlation-threshold 0.995 `
  --workers 4 `
  --progress
```

### 8. Build the fixed routing models

```powershell
tsgr-build-fixed-routing "$RESULTS\experiment_plan" `
  --dataset-root-override $DATASET `
  --output-dir "$RESULTS\fixed_routing_models" `
  --all-scenarios `
  --branch $BRANCH `
  --feature-set compact `
  --orientation-mode camera_aware `
  --os-alpha 0.65 `
  --c-top-n 5 `
  --progress
```

`--dataset-root-override` is the intended portability mechanism for a sanitized experiment plan whose stored `dataset_root` is `.`. It does not modify the plan and is required whenever the plan is outside the dataset directory tree.

### 9. Build training-only acceptance thresholds

```powershell
tsgr-build-fixed-acceptance "$RESULTS\experiment_plan" `
  --fixed-model-dir "$RESULTS\fixed_routing_models" `
  --output-dir "$RESULTS\acceptance" `
  --all-scenarios `
  --objective mcc `
  --min-positive-recall 0.99 `
  --progress
```

### 10. Evaluate ranking/routing and end-to-end behavior

Use the IMAGE or VIDEO processing report created in steps 2-3. With the explicit report names above, the directories are:

```powershell
$TEST_IMAGE = "$DATASET\reports\test_processing\image\high_recall\ignore_handedness\reference_image"
$TEST_VIDEO = "$DATASET\reports\test_processing\video\balanced\recovery_on\ignore_handedness\reference_video"
```

```powershell
tsgr-evaluate-fixed-routing "$RESULTS\experiment_plan" `
  --dataset-root-override $DATASET `
  --fixed-model-dir "$RESULTS\fixed_routing_models" `
  --test-processing-report "<PROCESSING_REPORT>" `
  --ground-truth-report "$RESULTS\ground_truth\final" `
  --output-dir "<ROUTING_RESULTS>" `
  --all-scenarios `
  --branch $BRANCH `
  --progress
```

```powershell
tsgr-evaluate-fixed-end-to-end "$RESULTS\experiment_plan" `
  --dataset-root-override $DATASET `
  --fixed-model-dir "$RESULTS\fixed_routing_models" `
  --acceptance-dir "$RESULTS\acceptance" `
  --test-processing-report "<PROCESSING_REPORT>" `
  --ground-truth-report "$RESULTS\ground_truth\final" `
  --output-dir "<E2E_RESULTS>" `
  --all-scenarios `
  --branch $BRANCH `
  --prediction-batch-size 64 `
  --progress
```

`<PROCESSING_REPORT>`, `<ROUTING_RESULTS>`, and `<E2E_RESULTS>` are placeholders, not machine-specific paths. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a full clean-room workflow.

## PLCC sensitivity

The public package includes the post-hoc PLCC sensitivity analysis used to test whether the selected compact-feature correlation threshold materially changes held-out performance. It does not change the fixed reference configuration after held-out results are observed.

```powershell
tsgr-run-plcc-sensitivity "$RESULTS\experiment_plan" `
  --dataset-root-override $DATASET `
  --test-processing-report "<IMAGE_PROCESSING_REPORT>" `
  --ground-truth-report "$RESULTS\ground_truth\final" `
  --output-dir "$RESULTS\plcc_sensitivity" `
  --branch $BRANCH `
  --scenario S1_ALL_IN_DOMAIN `
  --scenario S2_LOBO `
  --scenario S3_LOSO `
  --scenario S4_LOSO_BACKGROUND `
  --standard-grid `
  --model-workers 8 `
  --evaluation-workers 8 `
  --resume `
  --progress
```

## External SGRF baselines

The thirteen external comparison methods use `sgrf==3.2.0` in a separate CPython 3.11 environment. See [external_requirements/README.md](external_requirements/README.md) and [docs/COMMANDS.md](docs/COMMANDS.md).

## Reference results

Machine-readable CSV and JSON reference outputs are stored under `reference_results/`. Author-side plotting and manuscript-generation utilities are intentionally not part of this repository.

After reproducing a reference run, compare the generated outputs with the published result tree:

```powershell
tsgr-verify-reference-results `
  .\reference_results `
  .\results\reference_run `
  --output-dir .\results\verification `
  --require-match
```

A strict run reports `Reference-result verification: MATCH` only when all published CSV/JSON files match semantically.

## Reproducibility policy

Model selection, routing structure, specialist feature definitions, and held-out evaluation settings are fixed before interpreting S2-S4 results. Fold-local quantities are learned from the training portion of each fold only. Public scripts use relative paths and the released dataset contract.

For deterministic algorithmic stages, the clean-room release test compares outputs exactly where possible. Raw-media preprocessing also depends on MediaPipe and the native numerical stack, so the published reference environment is retained for the strictest reproduction target.

## Authors

- [Dawid Kalandyk](https://github.com/DawidKalandyk) — [ORCID 0000-0002-7317-5499](https://orcid.org/0000-0002-7317-5499)
- [Zuzanna Makowiecka](https://github.com/MZuza) — [ORCID 0009-0004-1333-6627](https://orcid.org/0009-0004-1333-6627)
- [Igor Stępień](https://github.com/Igorles) — [ORCID 0000-0001-6614-1218](https://orcid.org/0000-0001-6614-1218)

Repository maintained under the [KN GEST GitHub organization](https://github.com/KN-GEST-ongit).

## Citation

If you use TSGR-F, the accompanying dataset, or the reported reference results in scientific or academic work, **please cite the associated peer-reviewed publication**. The article is currently in preparation; its DOI and final BibTeX entry will be added after publication.

GitHub can also read [`CITATION.cff`](CITATION.cff) to expose the repository citation metadata.

## License

Code and the released TSGR-F dataset are distributed under the [MIT License](LICENSE).

> Reproduction note: `tsgr-audit-dataset --release-layout --strict` is for the immutable dataset immediately after extraction. The maintained `scripts/reproduce_reference.ps1` automatically switches to `--no-release-layout --strict` after derived `runs/`/`reports/` exist, so interrupted runs can be resumed safely. It also verifies the installed package/runtime version against the public `pyproject.toml` and recreates only the reviewer `.venv` if stale package code is detected; dataset and completed research checkpoints are preserved.
