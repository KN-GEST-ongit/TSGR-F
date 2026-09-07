"""Registry of SGRF comparison algorithms used by the TSGR-F benchmark.

The earlier SGRF validation published fifteen methods, but Murthy-Jadon and
Islam-Hossain-Andersson were excluded from the validation because they require a
separate background image.  The TSGR-F comparison therefore targets the same
remaining thirteen methods.

No TSGR-F landmarks, hand bounding boxes or MediaPipe outputs are supplied to
these methods.  A method receives the canonical image and performs its own
preprocessing.  Three upstream implementations require a coordinate payload
unconditionally; for those adapters only the full-frame rectangle is supplied
as an API compatibility value, never a detected hand ROI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SGRFMethodSpec:
    method_id: str
    display_name: str
    sort_order: int
    payload_kind: str
    learning_data_kind: str
    coordinate_policy: str
    certainty_scale: str
    included: bool = True
    exclusion_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Keep the upstream method order after removing the two unsupported background-dependent methods.
COMPARABLE_SGRF_METHODS: tuple[SGRFMethodSpec, ...] = (
    SGRFMethodSpec("MAUNG", "Maung", 10, "coords", "coords", "full_frame", "none"),
    SGRFMethodSpec("ADITHYA_RAJESH", "Adithya-Rajesh", 20, "coords", "coords", "none", "percent"),
    SGRFMethodSpec("EID_SCHWENKER", "Eid-Schwenker", 30, "image", "image", "none", "percent"),
    SGRFMethodSpec("PINTO_BORGES", "Pinto-Borges", 40, "coords", "coords", "full_frame", "percent"),
    SGRFMethodSpec("MOHMMAD_DADI", "Mohmmad-Dadi", 50, "image", "image", "none", "percent"),
    SGRFMethodSpec("GUPTA_JAAFAR", "Gupta-Jaafar", 60, "coords", "coords", "full_frame", "percent"),
    SGRFMethodSpec("MOHANTY_RAMBHATLA", "Mohanty-Rambhatla", 70, "coords", "coords", "none", "percent"),
    SGRFMethodSpec("ZHUANG_YANG", "Zhuang-Yang", 80, "coords", "coords", "none", "unit"),
    SGRFMethodSpec("CHANG_CHEN", "Chang-Chen", 90, "coords", "coords", "none", "percent"),
    SGRFMethodSpec("NAIDOO_OMLIN", "Naidoo-Omlin", 100, "image", "image", "none", "percent"),
    SGRFMethodSpec("JOSHI_KUMAR", "Joshi-Kumar", 110, "coords", "coords", "none", "percent"),
    SGRFMethodSpec("NGUYEN_HUYNH", "Nguyen-Huynh", 120, "coords", "coords", "none", "percent"),
    SGRFMethodSpec("OYEDOTUN_KHASHMAN", "Oyedotun-Khashman", 130, "coords", "coords", "none", "percent"),
)

EXCLUDED_SGRF_METHODS: tuple[SGRFMethodSpec, ...] = (
    SGRFMethodSpec(
        "MURTHY_JADON",
        "Murthy-Jadon",
        9010,
        "background",
        "background",
        "not_applicable",
        "not_applicable",
        included=False,
        exclusion_reason="Excluded from the published SGRF validation; requires a separate background image.",
    ),
    SGRFMethodSpec(
        "ISLAM_HOSSAIN_ANDERSSON",
        "Islam-Hossain-Andersson",
        9020,
        "background_coords",
        "background_coords",
        "not_applicable",
        "not_applicable",
        included=False,
        exclusion_reason="Excluded from the published SGRF validation; requires a separate background image.",
    ),
)

_METHOD_BY_ID = {item.method_id: item for item in (*COMPARABLE_SGRF_METHODS, *EXCLUDED_SGRF_METHODS)}


def comparable_method_ids() -> tuple[str, ...]:
    return tuple(item.method_id for item in COMPARABLE_SGRF_METHODS)


def method_spec(method_id: str) -> SGRFMethodSpec:
    key = str(method_id).strip().upper().replace("-", "_")
    try:
        return _METHOD_BY_ID[key]
    except KeyError as error:
        raise ValueError(f"Unknown SGRF method {method_id!r}.") from error


def normalize_method_selection(methods: Iterable[str] | None, *, all_methods: bool) -> tuple[SGRFMethodSpec, ...]:
    if all_methods and methods:
        raise ValueError("Choose either all comparable SGRF methods or an explicit method list, not both.")
    if all_methods:
        return COMPARABLE_SGRF_METHODS
    selected = list(methods or [])
    if not selected:
        raise ValueError("An explicit SGRF method selection is required unless all_methods=True.")
    resolved: list[SGRFMethodSpec] = []
    seen: set[str] = set()
    for method in selected:
        spec = method_spec(method)
        if not spec.included:
            raise ValueError(f"{spec.method_id} is intentionally excluded: {spec.exclusion_reason}")
        if spec.method_id not in seen:
            resolved.append(spec)
            seen.add(spec.method_id)
    return tuple(sorted(resolved, key=lambda item: item.sort_order))


def registry_rows() -> list[dict[str, object]]:
    rows = [item.to_dict() for item in (*COMPARABLE_SGRF_METHODS, *EXCLUDED_SGRF_METHODS)]
    rows.append(
        {
            "method_id": "TSGRF",
            "display_name": "TSGRF",
            "sort_order": 9999,
            "payload_kind": "native_tsgrf",
            "learning_data_kind": "native_tsgrf",
            "coordinate_policy": "native_tsgrf",
            "certainty_scale": "native_tsgrf",
            "included": True,
            "exclusion_reason": "",
        }
    )
    return rows
