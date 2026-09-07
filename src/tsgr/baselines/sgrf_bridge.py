"""Subprocess bridge between TSGR-F Python 3.12 and the upstream SGRF environment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsgr.utils.serialization import write_json


REFERENCE_SGRF_PYTHON_SERIES = "3.11"
REFERENCE_SGRF_VERSION = "3.2.0"


@dataclass(frozen=True, slots=True)
class SGRFEnvironmentAudit:
    python_executable: str
    python_version: str
    sgrf_version: str
    import_ok: bool
    payload: dict[str, Any]

    @property
    def reference_python_ok(self) -> bool:
        return self.python_version.startswith(f"{REFERENCE_SGRF_PYTHON_SERIES}.") or self.python_version == REFERENCE_SGRF_PYTHON_SERIES

    @property
    def reference_sgrf_ok(self) -> bool:
        return self.sgrf_version == REFERENCE_SGRF_VERSION


def worker_script_path() -> Path:
    return Path(__file__).with_name("sgrf_worker.py").resolve()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_worker(
    sgrf_python: str | Path,
    command: str,
    *,
    result_json: str | Path,
    job_json: str | Path | None = None,
    log_path: str | Path | None = None,
    timeout_s: float | None = None,
    extra_env: dict[str, str] | None = None,
    stream_output: bool = False,
) -> dict[str, Any]:
    """Run one standalone worker command and return its JSON result.

    stdout/stderr are captured to make every external baseline job reproducible.
    The worker writes a result JSON even for most Python-level failures; a hard
    interpreter crash is represented by the subprocess return code and log.
    """
    python_path = str(Path(sgrf_python))
    result_path = Path(result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = [python_path, str(worker_script_path()), command, "--result-json", str(result_path)]
    if job_json is not None:
        command_line.extend(["--job-json", str(Path(job_json))])
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    started = time.perf_counter()
    if stream_output and timeout_s is None:
        process = subprocess.Popen(
            command_line,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        captured_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            captured_lines.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
        stdout_text = "".join(captured_lines)
        stderr_text = ""
    else:
        completed = subprocess.run(
            command_line,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env=env,
            check=False,
        )
        returncode = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
    elapsed = time.perf_counter() - started
    log_text = (
        f"command={command_line!r}\n"
        f"returncode={returncode}\n"
        f"elapsed_s={elapsed:.9f}\n"
        "--- stdout ---\n"
        f"{stdout_text}\n"
        "--- stderr ---\n"
        f"{stderr_text}\n"
    )
    if log_path is not None:
        log_target = Path(log_path)
        log_target.parent.mkdir(parents=True, exist_ok=True)
        log_target.write_text(log_text, encoding="utf-8")
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "worker_status": "error",
            "error_type": "ExternalWorkerFailure",
            "error": "Worker did not produce a result JSON.",
        }
    payload["bridge_returncode"] = returncode
    payload["bridge_elapsed_s"] = elapsed
    payload["bridge_command"] = command_line
    if returncode != 0:
        error = payload.get("error") or stderr_text.strip() or stdout_text.strip()
        raise RuntimeError(
            f"SGRF worker command {command!r} failed with exit code {returncode}: {error}. "
            f"See {Path(log_path).resolve() if log_path else 'captured subprocess output'}."
        )
    if payload.get("worker_status") == "error":
        raise RuntimeError(f"SGRF worker reported an error: {payload.get('error')}")
    return payload


def audit_sgrf_environment(
    sgrf_python: str | Path,
    *,
    output_dir: str | Path,
    allow_nonreference_python: bool = False,
    allow_nonreference_sgrf: bool = False,
) -> SGRFEnvironmentAudit:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "sgrf_environment.json"
    payload = run_worker(
        sgrf_python,
        "audit",
        result_json=result_path,
        log_path=output / "sgrf_environment.log",
    )
    audit = SGRFEnvironmentAudit(
        python_executable=str(payload.get("python_executable", sgrf_python)),
        python_version=str(payload.get("python_version", "")),
        sgrf_version=str(payload.get("sgrf_version", "")),
        import_ok=bool(payload.get("sgrf_import_ok", False)),
        payload=payload,
    )
    if not audit.import_ok:
        raise RuntimeError(f"The selected external interpreter cannot import SGRF: {payload.get('sgrf_import_error', '')}")
    if not audit.reference_python_ok and not allow_nonreference_python:
        raise RuntimeError(
            f"Reference SGRF runs require Python {REFERENCE_SGRF_PYTHON_SERIES}.x; selected interpreter reports "
            f"{audit.python_version!r}. Use the explicit non-reference override only for diagnostics."
        )
    if not audit.reference_sgrf_ok and not allow_nonreference_sgrf:
        raise RuntimeError(
            f"Reference SGRF runs require sgrf=={REFERENCE_SGRF_VERSION}; selected environment reports "
            f"{audit.sgrf_version!r}. Use the explicit non-reference override only for diagnostics."
        )
    return audit


def write_job(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path)
    write_json(target, payload)
    return payload_sha256(payload)
