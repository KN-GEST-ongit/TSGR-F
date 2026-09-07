# Public dataset contract

TSGR-F operates on the released dataset only. Author-side acquisition folders, participant-identifying source names, source paths, and dataset-import utilities are not part of the public repository.

## Top-level structure

```text
tsgr_dataset/
├── training/
├── testing/
├── annotations.csv
└── annotations.json
```

`annotations.csv` and `annotations.json` contain the same records and only these fields:

1. `public_subject_id`
2. `background`
3. `gesture`
4. `take_id`
5. `frame_count`
6. `gesture_start_frame`
7. `gesture_end_frame`

The gesture interval is inclusive.

## Training images

Canonical path:

```text
training/P01/BLACK/A/images/image_000001.jpg
```

## Testing takes

Canonical path:

```text
testing/P01/BLACK/A/take_0001/
├── take_0001.avi
└── frames/
    ├── frame_000000.jpg
    └── ...
```

## Release packaging

Media are distributed as five subject packages (`P01`-`P05`) plus one manifest package. The manifest provides annotation files, release metadata, and SHA-256 hashes for every subject archive.

The `tsgr-download-dataset` command can install the complete release or selected subject packages. A complete install is strictly validated against the public contract.

> Reproduction note: `tsgr-audit-dataset --release-layout --strict` is for the immutable dataset immediately after extraction. The maintained `scripts/reproduce_reference.ps1` automatically switches to `--no-release-layout --strict` after derived `runs/`/`reports/` exist, so interrupted runs can be resumed safely. It also verifies the installed package/runtime version against the public `pyproject.toml` and recreates only the reviewer `.venv` if stale package code is detected; dataset and completed research checkpoints are preserved.
