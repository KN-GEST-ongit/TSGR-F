# Reproducibility guide

The public release is designed for a clean-room workflow in which a reviewer starts from only:

1. the public Git repository or code ZIP;
2. the public dataset release ZIP files;
3. the documented Python environments.

No author-side dataset import tree is required.

## Strict reference environment

Main framework:

- CPython 3.12.10
- NumPy 2.5.1
- opencv-contrib-python 4.13.0.92
- MediaPipe 0.10.35
- PyYAML 6.0.3
- Matplotlib 3.11.1
- openpyxl 3.1.5

External SGRF toolbox:

- CPython 3.11
- versions listed in `external_requirements/sgrf_3_2_0_requirements.txt`

## Clean-room directory layout

```text
review/
├── TSGR-F/
├── dataset_release/
├── data/
│   └── tsgr_dataset/
└── results/
```

All examples use paths relative to the repository. Absolute author-machine paths are neither required nor embedded in the public scripts.

## Reference handedness policy

The published IMAGE and VIDEO reference runs use `ignore_handedness`. MediaPipe handedness labels and scores remain in the diagnostics, but they are not used as a rejection gate for the reference classification run. `reject_left` is supported only as a separate policy ablation and will not reproduce the published reference results.

## Verification stages

A strict reproduction should verify, in order:

1. dataset package SHA-256 checksums;
2. public annotation equivalence between CSV and JSON;
3. dataset file counts and canonical paths;
4. feature schema and processed-run completeness;
5. generated S1-S4 fold membership;
6. fixed-model fingerprints and fold-local parameters;
7. ranking predictions;
8. acceptance decisions;
9. scenario, fold, gesture, confusion-matrix, temporal, and bootstrap summaries;
10. PLCC sensitivity results;
11. external SGRF baseline summaries when the separate environment is used.

Exact file equality is required for deterministic machine-readable outputs whenever timestamps or runtime telemetry are not part of the file. For files containing timestamps or machine-specific performance measurements, semantic comparison must ignore those fields.

## No held-out retuning

S2-S4 results are evaluation outputs, not inputs to model selection. The routing architecture and its development-selected constants remain fixed. Fold-local weights, thresholds, feature masks, and calibration parameters are computed from the training portion of each fold only.

## Verify reproduced outputs

After a clean-room run, compare its machine-readable results with the published reference tree:

```powershell
tsgr-verify-reference-results `
  .\reference_results `
  .\results\reference_run `
  --output-dir .\results\verification `
  --require-match
```

The verifier parses CSV and JSON content so differences in JSON formatting or CSV line endings do not produce false mismatches.

## Portable plan continuation

After generating folds into a results directory outside `tsgr_dataset`, continue with:

```powershell
tsgr-build-fold-models .\results\experiment_plan --all-scenarios --all-folds --branch raw.wrist_middle_mcp --compact-correlation-threshold 0.995 --workers 4 --progress
tsgr-build-fixed-routing .\results\experiment_plan --dataset-root-override .\data\tsgr_dataset --output-dir .\results\fixed_routing_models --all-scenarios --branch raw.wrist_middle_mcp --feature-set compact --orientation-mode camera_aware --os-alpha 0.65 --c-top-n 5 --progress
```

The explicit dataset-root override is runtime-only and keeps the experiment plan free of machine-specific absolute paths.

> Reproduction note: `tsgr-audit-dataset --release-layout --strict` is for the immutable dataset immediately after extraction. The maintained `scripts/reproduce_reference.ps1` automatically switches to `--no-release-layout --strict` after derived `runs/`/`reports/` exist, so interrupted runs can be resumed safely. It also verifies the installed package/runtime version against the public `pyproject.toml` and recreates only the reviewer `.venv` if stale package code is detected; dataset and completed research checkpoints are preserved.
