# External SGRF baseline environment

The thirteen comparison methods run in an isolated **CPython 3.11** environment. Do not install the SGRF dependency stack into the main TSGR-F CPython 3.12.10 environment.

Reference setup:

```powershell
py -3.11 -m venv .venv_sgrf
.\.venv_sgrf\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\external_requirements\sgrf_3_2_0_requirements.txt
```

Then return to the main TSGR-F environment and point the adapter at the external interpreter:

```powershell
deactivate
.\.venv\Scripts\Activate.ps1
$SGRF_PYTHON = (Resolve-Path ".\.venv_sgrf\Scripts\python.exe").Path

tsgr-audit-sgrf-baselines `
  --sgrf-python $SGRF_PYTHON `
  --output-dir ".\tsgr_dataset\reports\external_baselines\environment" `
  --no-allow-nonreference-python `
  --no-allow-nonreference-sgrf
```

The primary paper comparison uses the upstream closed-set top-1 output (`--rejection-policy closed_set`). The adapter stores both the raw upstream certainty and a normalized [0,1] value when a method exposes certainty. Upstream methods do not use one common certainty convention: Zhuang-Yang is unit-scaled, most methods are percentage-scaled, and Maung returns no certainty. Therefore `certainty_reject` is only a sensitivity-analysis option and its threshold is always specified with `--certainty-threshold-normalized`.

`--workers 1` is the reference starting point for both training and inference because several methods use TensorFlow/Keras or multithreaded numerical libraries internally. Increase external-process concurrency only after measuring RAM/VRAM/CPU behavior on the target machine.
