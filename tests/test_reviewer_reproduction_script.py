from __future__ import annotations

import re
from pathlib import Path


def _script_text() -> str:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "reproduce_reference.ps1"
    assert script.is_file()
    return script.read_text(encoding="utf-8")


def test_reviewer_script_contains_canonical_core_stages() -> None:
    text = _script_text()
    required = (
        "tsgr-download-dataset.exe",
        "tsgr-audit-dataset.exe",
        "tsgr-process-training.exe",
        "tsgr-process-testing.exe",
        "tsgr-build-hand-presence-reference.exe",
        "tsgr-analyze-test-ground-truth.exe",
        "tsgr-generate-folds.exe",
        "tsgr-build-fold-models.exe",
        "tsgr-build-fixed-routing.exe",
        "tsgr-build-fixed-acceptance.exe",
        "tsgr-evaluate-fixed-routing.exe",
        "tsgr-evaluate-fixed-end-to-end.exe",
    )
    for token in required:
        assert token in text


def test_reviewer_script_uses_reference_parameters_and_portable_plan_override() -> None:
    text = _script_text()
    for token in (
        "ignore_handedness",
        "--fallback-fps 30",
        "--compact-correlation-threshold 0.995",
        "--os-alpha 0.65",
        "--c-top-n 5",
        "--objective mcc",
        "--min-positive-recall 0.99",
        "--dataset-root-override $Dataset",
        '"raw.wrist_middle_mcp"',
    ):
        assert token in text


def test_reviewer_script_has_no_machine_specific_absolute_paths() -> None:
    text = _script_text()
    assert not re.search(r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]", text)


def test_reviewer_script_distinguishes_pristine_release_from_working_dataset() -> None:
    text = _script_text()
    assert "Test-DatasetRuntimeArtifacts" in text
    assert "--release-layout --strict" in text
    assert "--no-release-layout --strict" in text
    assert "derived runs/reports are allowed" in text


def test_dataset_audit_cli_exposes_release_layout_switch() -> None:
    root = Path(__file__).resolve().parents[1]
    cli = root / "src" / "tsgr" / "cli" / "audit_public_dataset.py"
    text = cli.read_text(encoding="utf-8")
    assert '"--release-layout"' in text
    assert "BooleanOptionalAction" in text


def test_reviewer_script_rebuilds_stale_virtual_environment_and_checks_versions() -> None:
    text = _script_text()
    assert "Get-ProjectVersion" in text
    assert "Get-InstalledFrameworkProbe" in text
    assert "DistributionVersion" in text
    assert "RuntimeVersion" in text
    assert "Recreating only .venv; data/results are preserved" in text
    assert "TSGR-F resolved outside the reviewer virtual environment" in text


def test_reviewer_script_checks_installed_audit_cli_capability_before_dataset_audit() -> None:
    text = _script_text()
    assert "$auditHelp" in text
    assert "--no-release-layout" in text
    assert "reviewer environment is inconsistent with the public source" in text


def test_reviewer_probe_is_non_terminating_under_windows_powershell_stop_policy() -> None:
    text = _script_text()
    assert "System.Diagnostics.ProcessStartInfo" in text
    assert "RedirectStandardOutput = $true" in text
    assert "RedirectStandardError = $true" in text
    assert "& $PythonPath -c $probeCode" not in text
    assert "if ($process.ExitCode -ne 0)" in text


def test_reviewer_report_uses_validated_runtime_version() -> None:
    text = _script_text()
    assert "tsgr_version = $installedProbe.RuntimeVersion" in text
    assert "tsgr_version = $installedVersion" not in text


def test_reviewer_script_contains_publication_post_core_stages() -> None:
    text = _script_text()
    for token in (
        "tsgr-bootstrap-fixed-end-to-end.exe",
        "tsgr-benchmark-fixed-routing.exe",
        "tsgr-run-plcc-sensitivity.exe",
        '"bootstrap"',
        '"benchmark"',
        '"plcc"',
    ):
        assert token in text


def test_reviewer_script_uses_canonical_publication_post_core_parameters() -> None:
    text = _script_text()
    for token in (
        "[int]$BootstrapSamples = 5000",
        "[int]$BootstrapSeed = 2026",
        "[int]$BenchmarkMaxFramesPerFold = 1000",
        "[int]$BenchmarkBatchSize = 64",
        "[int]$BenchmarkRepeats = 7",
        "[int]$BenchmarkWarmup = 2",
        "--standard-grid",
        "--scenario S1_ALL_IN_DOMAIN",
        "--scenario S2_LOBO",
        "--scenario S3_LOSO",
        "--scenario S4_LOSO_BACKGROUND",
        "--resume",
    ):
        assert token in text


def test_reviewer_report_marks_plcc_as_post_hoc_only() -> None:
    text = _script_text()
    assert "plcc_post_hoc_exploratory = $true" in text
    assert "plcc_eligible_for_final_model_reselection = $false" in text
