"""
patient_data.py: per-patient baseline brace parameters.

Reads "Patient specific data.csv" from the same folder as the Streamlit app.
The UI force sliders use a force-share convention:
    main share = 0.50 and counter split = 0.50
which displays as main:proximal:distal = 2:1:1.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_DATA_CSV = Path(__file__).parent / "Patient specific data.csv"

_DEFAULT_FORCE_RATIO = 0.50
_DEFAULT_COUNTER_SPLIT = 0.50
_DEFAULT_INTRUSION_RATIO = 1.33
_DEFAULT_SPAN = 2


def _f(value: Any) -> float | None:
	try:
		if value is None:
			return None
		s = str(value).strip()
		if not s or s.lower() in {"nan", "n/a", "na", "none", "-"}:
			return None
		return float(s)
	except (ValueError, TypeError):
		return None


def _get(row: dict[str, str], *names: str) -> str:
	norm = {k.strip().lower(): v for k, v in row.items()}
	for name in names:
		hit = norm.get(name.strip().lower())
		if hit is not None:
			return hit
	return ""


def load_patient_baselines() -> dict[str, dict]:
	out: dict[str, dict] = {}

	with open(_DATA_CSV, "r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		for raw in reader:
			row = {k.strip(): str(v).strip() for k, v in raw.items() if k is not None}
			pid = _get(row, "Patient ID", "Patient", "PID").strip().upper()
			if not pid:
				continue

			h_m = _f(_get(row, "Main pad height"))
			w_m = _f(_get(row, "Main pad width"))
			h_u = _f(_get(row, "Upper Counter pad height", "Upper counter pad height"))
			w_u = _f(_get(row, "Upper Counter pad width", "Upper counter pad width"))
			h_l = _f(_get(row, "Lower Counter pad height", "Lower counter pad height"))
			w_l = _f(_get(row, "Lower counter pad width", "lower counter pad width"))
			d_m = _f(_get(row, "Main pad distance"))
			d_u = _f(_get(row, "Upper counter pad distance", "Upper Counter pad distance"))
			d_l = _f(_get(row, "Lower Counterpad distance", "Lower Counter pad distance", "Lower counter pad distance"))

			main_area = h_m * w_m if h_m is not None and w_m is not None else None
			up_area = h_u * w_u if h_u is not None and w_u is not None else None
			lo_area = h_l * w_l if h_l is not None and w_l is not None else None

			if up_area is not None and lo_area is not None:
				counter_area = up_area + lo_area
			elif up_area is not None:
				counter_area = up_area * 2.0
			elif lo_area is not None:
				counter_area = lo_area * 2.0
			else:
				counter_area = None

			if d_m is not None and d_u is not None and d_l is not None:
				d_ctr = (d_u + d_l) / 2.0
				intrusion_ratio = d_m / d_ctr if d_ctr > 0 else _DEFAULT_INTRUSION_RATIO
			elif d_m is not None and (d_u is not None or d_l is not None):
				d_ctr = d_u if d_u is not None else d_l
				intrusion_ratio = d_m / d_ctr if d_ctr and d_ctr > 0 else _DEFAULT_INTRUSION_RATIO
			else:
				intrusion_ratio = _DEFAULT_INTRUSION_RATIO

			out[pid] = {
				"main_area_mm2": main_area,
				"counter_area_mm2": counter_area,
				"baseline_ratio": round(intrusion_ratio, 3),
				"baseline_force_ratio": _DEFAULT_FORCE_RATIO,
				"baseline_counter_split": _DEFAULT_COUNTER_SPLIT,
				"baseline_span": _DEFAULT_SPAN,
			}

	return out


def list_patients() -> list[str]:
	return sorted(load_patient_baselines().keys())
