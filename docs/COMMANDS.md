# Command reference

The top-level README contains the standard reproduction sequence. This document lists the public command groups.

## Dataset and preprocessing

- `tsgr-download-model`
- `tsgr-download-dataset`
- `tsgr-audit-dataset`
- `tsgr-process-training`
- `tsgr-process-testing`
- `tsgr-build-hand-presence-reference`
- `tsgr-analyze-test-ground-truth`
- `tsgr-generate-folds`
- `tsgr-build-fold-models`

## TSGR-F model and evaluation

- `tsgr-build-fixed-routing`
- `tsgr-evaluate-fixed-routing`
- `tsgr-build-fixed-acceptance`
- `tsgr-evaluate-fixed-end-to-end`
- `tsgr-bootstrap-fixed-end-to-end`
- `tsgr-benchmark-fixed-routing`
- `tsgr-run-plcc-sensitivity`

Use `--help` on every command for the complete option list.

## External SGRF comparison

- `tsgr-build-sgrf-baselines`
- `tsgr-evaluate-sgrf-baselines`
- `tsgr-report-sgrf-evaluation-progress`
- `tsgr-export-sgrf-evaluation-shard`
- `tsgr-import-sgrf-evaluation-shard`
- `tsgr-audit-sgrf-baselines`
- `tsgr-audit-sgrf-evaluation`
- `tsgr-export-sgrf-results`
- `tsgr-merge-method-results`

## Reproduction verification

- `tsgr-verify-reference-results`

Use this command to compare clean-room CSV/JSON outputs with the published `reference_results/` tree.

The external toolbox runs in a separate CPython 3.11 environment. The main TSGR-F package remains in CPython 3.12.

## Portable experiment plans

Commands that need to resolve media or processed TRAIN runs from a sanitized experiment plan accept `--dataset-root-override <PATH>`. Use it for fixed-routing build/evaluation, end-to-end evaluation, runtime benchmarking, and PLCC sensitivity whenever the plan is stored outside the dataset tree.
