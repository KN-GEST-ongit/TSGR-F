"""Runtime contract for the released TSGR-F dataset.

This module only reads and validates the public dataset layout. It intentionally
contains no import, migration, source-path reconstruction, or release-export logic.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from tsgr.utils.natural_sort import natural_key
from tsgr.utils.serialization import sha256_file

PUBLIC_CONTRACT_SCHEMA = "tsgr_public_dataset_v1"
PUBLIC_ANNOTATION_FIELDS: tuple[str, ...] = (
    "public_subject_id",
    "background",
    "gesture",
    "take_id",
    "frame_count",
    "gesture_start_frame",
    "gesture_end_frame",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PUBLIC_SUBJECT_RE = re.compile(r"^P\d{2,}$", re.IGNORECASE)
PUBLIC_TAKE_RE = re.compile(r"^take_\d{4,}$", re.IGNORECASE)
PUBLIC_IMAGE_RE = re.compile(r"^image_\d{6}\.(?:jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
PUBLIC_FRAME_RE = re.compile(r"^frame_\d{6}\.(?:jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
NONPORTABLE_TEXT_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"(?:^|[\\/])Users(?:[\\/]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])home(?:[\\/]|$)", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class PublicDatasetAudit:
    dataset_root: Path
    training_images: int
    test_takes: int
    test_frames: int
    test_videos: int
    annotations_csv: int
    annotations_json: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PUBLIC_CONTRACT_SCHEMA,
            "dataset_root": ".",
            "training_images": self.training_images,
            "test_takes": self.test_takes,
            "test_frames": self.test_frames,
            "test_videos": self.test_videos,
            "annotations_csv": self.annotations_csv,
            "annotations_json": self.annotations_json,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))



def _safe_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not boolean.")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer; got {value!r}.") from error
    return number


def public_take_key(public_subject_id: str, background: str, gesture: str, take_id: str) -> str:
    """Return the globally unique internal/public-safe key used in experiment artifacts."""
    return f"{public_subject_id}_{background}_{gesture}_{take_id}"


def _normalize_annotation(row: dict[str, Any], *, row_label: str = "annotation") -> dict[str, Any]:
    keys = set(row)
    missing = set(PUBLIC_ANNOTATION_FIELDS) - keys
    if missing:
        raise ValueError(f"{row_label} is missing public fields: {sorted(missing)}")
    subject = str(row["public_subject_id"]).strip().upper()
    background = str(row["background"]).strip().upper()
    gesture = str(row["gesture"]).strip().upper()
    take_id = str(row["take_id"]).strip().lower()
    if not PUBLIC_SUBJECT_RE.fullmatch(subject):
        raise ValueError(f"{row_label}: invalid public_subject_id={subject!r}.")
    if not background or any(char in background for char in "/\\"):
        raise ValueError(f"{row_label}: invalid background={background!r}.")
    if not gesture or any(char in gesture for char in "/\\"):
        raise ValueError(f"{row_label}: invalid gesture={gesture!r}.")
    if not PUBLIC_TAKE_RE.fullmatch(take_id):
        raise ValueError(f"{row_label}: invalid take_id={take_id!r}.")
    frame_count = _safe_int(row["frame_count"], "frame_count")
    start = _safe_int(row["gesture_start_frame"], "gesture_start_frame")
    end = _safe_int(row["gesture_end_frame"], "gesture_end_frame")
    if frame_count <= 0:
        raise ValueError(f"{row_label}: frame_count must be positive.")
    if start < 0 or end < 0 or start > end or end >= frame_count:
        raise ValueError(
            f"{row_label}: invalid inclusive interval [{start}, {end}] for frame_count={frame_count}."
        )
    return {
        "public_subject_id": subject,
        "background": background,
        "gesture": gesture,
        "take_id": take_id,
        "frame_count": frame_count,
        "gesture_start_frame": start,
        "gesture_end_frame": end,
    }


def _annotation_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        natural_key(str(row["public_subject_id"])),
        natural_key(str(row["background"])),
        natural_key(str(row["gesture"])),
        natural_key(str(row["take_id"])),
    )


def _validate_annotation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows, start=1):
        item = _normalize_annotation(dict(row), row_label=f"annotation row {index}")
        key = (item["public_subject_id"], item["background"], item["gesture"], item["take_id"])
        if key in seen:
            raise ValueError(f"Duplicate public annotation identity: {'/'.join(key)}")
        seen.add(key)
        normalized.append(item)
    normalized.sort(key=_annotation_sort_key)
    return normalized


def load_public_annotations(dataset_root: str | Path, *, require_both: bool = True) -> list[dict[str, Any]]:
    """Load and cross-check the public CSV/JSON annotation contract."""
    root = Path(dataset_root)
    csv_path = root / "annotations.csv"
    json_path = root / "annotations.json"
    if not csv_path.is_file() and not json_path.is_file():
        raise FileNotFoundError(
            f"Public annotation contract is missing. Expected {csv_path.name} and {json_path.name} in {root}."
        )
    if require_both and (not csv_path.is_file() or not json_path.is_file()):
        raise FileNotFoundError("Both annotations.csv and annotations.json are required by the public contract.")
    csv_rows = _validate_annotation_rows(_read_csv(csv_path)) if csv_path.is_file() else []
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("annotations.json must contain a JSON array of annotation objects.")
        json_rows = _validate_annotation_rows(payload)
    else:
        json_rows = []
    if csv_rows and json_rows and csv_rows != json_rows:
        raise ValueError("annotations.csv and annotations.json are not semantically identical.")
    return csv_rows or json_rows



def _image_info(path: Path) -> tuple[int | str, int | str]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return "", ""
    height, width = image.shape[:2]
    return int(width), int(height)


def discover_public_training_images(
    dataset_root: str | Path,
    *,
    compute_sha256: bool = False,
    read_dimensions: bool = False,
) -> list[dict[str, Any]]:
    """Reconstruct portable training metadata from public folders only."""
    root = Path(dataset_root)
    training_root = root / "training"
    rows: list[dict[str, Any]] = []
    for subject_dir in sorted((p for p in training_root.iterdir() if p.is_dir()), key=natural_key) if training_root.is_dir() else []:
        for background_dir in sorted((p for p in subject_dir.iterdir() if p.is_dir()), key=natural_key):
            for gesture_dir in sorted((p for p in background_dir.iterdir() if p.is_dir()), key=natural_key):
                images_dir = gesture_dir / "images" if (gesture_dir / "images").is_dir() else gesture_dir
                images = sorted(
                    (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=natural_key,
                ) if images_dir.is_dir() else []
                for ordinal, path in enumerate(images, start=1):
                    match = re.search(r"(\d+)$", path.stem)
                    number = int(match.group(1)) if match else ordinal
                    width, height = _image_info(path) if read_dimensions else ("", "")
                    rows.append(
                        {
                            "image_id": f"{subject_dir.name}_{background_dir.name.upper()}_{gesture_dir.name.upper()}_{number:06d}",
                            "public_subject_id": subject_dir.name.upper(),
                            "background": background_dir.name.upper(),
                            "gesture_id": gesture_dir.name.upper(),
                            "filename": path.name,
                            "relative_path": path.relative_to(root).as_posix(),
                            "sha256": sha256_file(path) if compute_sha256 else "",
                            "size_bytes": path.stat().st_size,
                            "width": width,
                            "height": height,
                            "input_is_mirrored": "true",
                        }
                    )
    return rows


def _video_for_take(take_dir: Path, local_take_id: str) -> Path | None:
    candidates = [take_dir / f"{local_take_id}.avi", take_dir / "source_video.avi"]
    candidates.extend(sorted(take_dir.glob("*.avi"), key=natural_key))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _video_fps(path: Path | None) -> float | str:
    if path is None:
        return ""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return ""
        value = float(capture.get(cv2.CAP_PROP_FPS))
        return value if value > 0 else ""
    finally:
        capture.release()


def discover_public_test_takes(
    dataset_root: str | Path,
    *,
    read_video_fps: bool = False,
) -> list[dict[str, Any]]:
    """Reconstruct test-take rows from public annotations and hierarchy."""
    root = Path(dataset_root)
    annotations = load_public_annotations(root, require_both=True)
    output: list[dict[str, Any]] = []
    for ann in annotations:
        subject = str(ann["public_subject_id"])
        background = str(ann["background"])
        gesture = str(ann["gesture"])
        local_take = str(ann["take_id"])
        take_dir = root / "testing" / subject / background / gesture / local_take
        video = _video_for_take(take_dir, local_take)
        output.append(
            {
                "take_id": public_take_key(subject, background, gesture, local_take),
                "public_take_id": local_take,
                "public_subject_id": subject,
                "background": background,
                "gesture_id": gesture,
                "relative_path": take_dir.relative_to(root).as_posix(),
                "canonical_frame_count": int(ann["frame_count"]),
                "source_video_present": "true" if video is not None else "false",
                "source_video_fps": _video_fps(video) if read_video_fps else "",
                "annotation_present": "true",
                "gesture_start_frame": int(ann["gesture_start_frame"]),
                "gesture_end_frame": int(ann["gesture_end_frame"]),
                "annotation_complete": "true",
                "input_is_mirrored": "true",
            }
        )
    return output


def discover_public_test_frames(
    dataset_root: str | Path,
    *,
    take_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build frame rows from annotations and public frame filenames."""
    root = Path(dataset_root)
    takes = discover_public_test_takes(root, read_video_fps=False)
    output: list[dict[str, Any]] = []
    for take in takes:
        global_take = str(take["take_id"])
        if take_ids and global_take not in take_ids:
            continue
        take_dir = root / str(take["relative_path"])
        frames_dir = take_dir / "frames" if (take_dir / "frames").is_dir() else take_dir
        frame_paths = sorted(
            (p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=natural_key,
        ) if frames_dir.is_dir() else []
        expected = int(take["canonical_frame_count"])
        if len(frame_paths) != expected:
            raise ValueError(f"Frame count mismatch for {global_take}: expected {expected}, found {len(frame_paths)}.")
        start = int(take["gesture_start_frame"])
        end = int(take["gesture_end_frame"])
        for frame_index, path in enumerate(frame_paths):
            is_gesture = start <= frame_index <= end
            output.append(
                {
                    "take_id": global_take,
                    "public_take_id": take["public_take_id"],
                    "frame_index": frame_index,
                    "true_label": take["gesture_id"] if is_gesture else "NO_GESTURE",
                    "is_gesture_frame": "1" if is_gesture else "0",
                    "filename": path.name,
                    "relative_path": path.relative_to(root).as_posix(),
                    "public_subject_id": take["public_subject_id"],
                    "background": take["background"],
                    "gesture_id": take["gesture_id"],
                }
            )
    return output


def _public_contract_payload(root: Path) -> dict[str, Any]:
    rows = load_public_annotations(root, require_both=True)
    return {
        "schema_version": PUBLIC_CONTRACT_SCHEMA,
        "dataset_root": ".",
        "annotations_csv": "annotations.csv",
        "annotations_json": "annotations.json",
        "annotation_record_count": len(rows),
        "annotations_csv_sha256": sha256_file(root / "annotations.csv"),
        "annotations_json_sha256": sha256_file(root / "annotations.json"),
    }


def resolve_dataset_root_from_plan(
    experiment_plan_dir: str | Path,
    *,
    override: str | Path | None = None,
) -> Path:
    """Resolve a dataset root without relying on a creator-specific absolute path."""
    if override is not None:
        root = Path(override).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root override does not exist: {root}")
        return root
    plan = Path(experiment_plan_dir).resolve()
    meta_path = plan / "experiment_plan.json"
    if meta_path.is_file():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = str(payload.get("dataset_root", "")).strip()
        if raw and raw not in {".", "__RUNTIME__"}:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (plan / candidate).resolve()
            if candidate.is_dir():
                return candidate
    for candidate in [plan, *plan.parents]:
        if (candidate / "training").is_dir() and (candidate / "testing").is_dir() and (candidate / "annotations.csv").is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to resolve public dataset root from experiment plan. Pass --dataset-root-override."
    )


def _text_has_nonportable_token(text: str) -> str | None:
    for pattern in NONPORTABLE_TEXT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def validate_public_dataset(
    dataset_root: str | Path,
    *,
    require_videos: bool = True,
    require_annotations: bool = True,
    release_layout: bool = False,
) -> PublicDatasetAudit:
    """Strictly validate the release-safe folder contract."""
    root = Path(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return PublicDatasetAudit(root, 0, 0, 0, 0, 0, 0, (f"Dataset root does not exist: {root}",), ())
    if not (root / "training").is_dir():
        errors.append("Missing training/ directory.")
    if not (root / "testing").is_dir():
        errors.append("Missing testing/ directory.")
    try:
        annotations = load_public_annotations(root, require_both=require_annotations)
    except Exception as error:
        annotations = []
        errors.append(str(error))
    training_rows = discover_public_training_images(root, compute_sha256=False, read_dimensions=False)
    test_takes = 0
    test_frames = 0
    test_videos = 0
    if annotations:
        for ann in annotations:
            take_dir = root / "testing" / ann["public_subject_id"] / ann["background"] / ann["gesture"] / ann["take_id"]
            if not take_dir.is_dir():
                errors.append(f"Missing public take directory: {take_dir.relative_to(root).as_posix()}")
                continue
            test_takes += 1
            frames_dir = take_dir / "frames" if (take_dir / "frames").is_dir() else take_dir
            frames = sorted(
                (p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                key=natural_key,
            ) if frames_dir.is_dir() else []
            test_frames += len(frames)
            if len(frames) != int(ann["frame_count"]):
                errors.append(
                    f"Frame-count mismatch for {ann['public_subject_id']}/{ann['background']}/{ann['gesture']}/{ann['take_id']}: "
                    f"annotation={ann['frame_count']} filesystem={len(frames)}"
                )
            video = _video_for_take(take_dir, str(ann["take_id"]))
            if video is not None:
                test_videos += 1
            elif require_videos:
                errors.append(f"Missing AVI for {take_dir.relative_to(root).as_posix()}")
    csv_count = len(_read_csv(root / "annotations.csv")) if (root / "annotations.csv").is_file() else 0
    json_count = 0
    if (root / "annotations.json").is_file():
        try:
            payload = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
            json_count = len(payload) if isinstance(payload, list) else 0
        except json.JSONDecodeError:
            pass
    forbidden_names = {"metadata", "reports", "derived", "runs", "cache", "__pycache__"}
    for forbidden in sorted(forbidden_names):
        if (root / forbidden).exists():
            message = f"Internal directory present: {forbidden}/"
            if release_layout:
                errors.append(message)
            else:
                warnings.append(message)
    if release_layout:
        allowed_top = {"training", "testing", "annotations.csv", "annotations.json"}
        unexpected_top = sorted(path.name for path in root.iterdir() if path.name not in allowed_top)
        if unexpected_top:
            errors.append(f"Unexpected top-level release entries: {unexpected_top}")
        for path in root.rglob("*"):
            if path == root:
                continue
            if path.is_dir() and path.name.lower() in forbidden_names:
                errors.append(f"Forbidden release directory: {path.relative_to(root).as_posix()}")
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in {"annotations.csv", "annotations.json"}:
                continue
            parts = Path(rel).parts
            if rel.startswith("training/") and path.suffix.lower() in IMAGE_EXTENSIONS:
                # Accepted public training layouts:
                # training/P01/BLACK/A/images/image_000001.jpg
                # training/P01/BLACK/A/image_000001.jpg
                valid_layout = (
                    (len(parts) == 6 and parts[4].lower() == "images")
                    or len(parts) == 5
                )
                subject = parts[1] if len(parts) > 1 else ""
                filename = parts[-1] if parts else ""
                if not valid_layout or not PUBLIC_SUBJECT_RE.fullmatch(subject) or not PUBLIC_IMAGE_RE.fullmatch(filename):
                    errors.append(f"Noncanonical public training path: {rel}")
                continue
            if rel.startswith("testing/") and path.suffix.lower() in IMAGE_EXTENSIONS:
                # Canonical release frames are always nested under frames/.
                valid_layout = len(parts) == 7 and parts[5].lower() == "frames"
                subject = parts[1] if len(parts) > 1 else ""
                take_id = parts[4] if len(parts) > 4 else ""
                filename = parts[-1] if parts else ""
                if (
                    not valid_layout
                    or not PUBLIC_SUBJECT_RE.fullmatch(subject)
                    or not PUBLIC_TAKE_RE.fullmatch(take_id)
                    or not PUBLIC_FRAME_RE.fullmatch(filename)
                ):
                    errors.append(f"Noncanonical public test-frame path: {rel}")
                continue
            if rel.startswith("testing/") and path.suffix.lower() == ".avi":
                valid_layout = len(parts) == 6
                subject = parts[1] if len(parts) > 1 else ""
                take_id = parts[4] if len(parts) > 4 else ""
                filename = parts[-1] if parts else ""
                if (
                    not valid_layout
                    or not PUBLIC_SUBJECT_RE.fullmatch(subject)
                    or not PUBLIC_TAKE_RE.fullmatch(take_id)
                    or filename.lower() != f"{take_id.lower()}.avi"
                ):
                    errors.append(f"Noncanonical public AVI path: {rel}")
                continue
            errors.append(f"Forbidden release file: {rel}")
    for text_path in (root / "annotations.csv", root / "annotations.json"):
        if text_path.is_file():
            token = _text_has_nonportable_token(text_path.read_text(encoding="utf-8", errors="ignore"))
            if token:
                errors.append(f"Nonportable path token {token!r} found in {text_path.name}.")
    return PublicDatasetAudit(
        root,
        len(training_rows),
        test_takes,
        test_frames,
        test_videos,
        csv_count,
        json_count,
        tuple(errors),
        tuple(warnings),
    )
