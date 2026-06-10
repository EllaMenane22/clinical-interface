"""
AIS Brace Correction Predictor - Clinical Decision Support
Patient-specific finite-element exploration for scoliosis brace design.
"""

import shutil as _shutil, pathlib as _pathlib
_cache = _pathlib.Path(__file__).parent / "__pycache__"
if _cache.exists():
    _shutil.rmtree(_cache, ignore_errors=True)

import math
import sys
import traceback
import json
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titleweight'] = 'bold'
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
import streamlit as st

from patient_data import load_patient_baselines

if "fe_runner" in sys.modules:
    del sys.modules["fe_runner"]
from fe_runner import run_real_model, get_last_run_info

BASE_DIR         = Path(r"C:\Users\<username>\OneDrive - Imperial College London\Year 4\FYP\Patient_1_final")
SENSITIVITY_DIR  = BASE_DIR / "Automated models" / "Sensitivities 2 multi-patient"
RUN_ROOT         = Path(r"C:\Temp\marc_ui_runs")
ACTIVE_MODEL_DIR = BASE_DIR / "Automated models" / "Updated automated"
UI_VERSION       = "v8_redesigned_compact_2026_06_01"

SPINE_LEVELS = [
    "L5","L4","L3","L2","L1",
    "T12","T11","T10","T9","T8","T7","T6",
    "T5","T4","T3","T2","T1",
    "C7","C6","C5","C4","C3","C2","C1",
]
_LIDX = {lv: i for i, lv in enumerate(SPINE_LEVELS)}

# Validated 2:1:1 baseline uses a SINGLE universal pad pressure of 7.5151 kPa for every
# region; the mid value is the baseline (pressure / p_mid = 1.0 at baseline). Low/high only
# bound the slider.
PRESSURE_RANGE = {
    "Thoracic":      (6.0, 7.5151, 9.5),
    "Thoracolumbar": (6.0, 7.5151, 9.5),
    "Lumbar":        (6.0, 7.5151, 9.5),
}
_K_RZ = 3.065e8

NAVY        = "#1F4E79"
STEEL       = "#4F81BD"
TEAL        = "#2A9D8F"
ORANGE      = "#E76F51"
DARK_TEXT   = "#212529"
MID_GREY    = "#6C757D"
BORDER      = "#E5E5E5"
BG          = "#F8F9FA"
CARD        = "#FFFFFF"
RED         = "#C0392B"
GREEN       = TEAL
AFTER       = "#9B59B6"   # post-brace Cobb lines / overlay (distinct from pad colours)

BASELINE_AREA = math.pi * (144.725 / 2.0) ** 2  # π × r² with diameter 144.725 mm ≈ 16,442 mm²

# Universal baseline pad forces (mN) — identical to the sweep generator
# (01_generate_all_patient_cases.py) and the Updated-automated baseline metadata.
# The force-ratio control holds TOTAL = main + 2·counter fixed and redistributes it.
BASELINE_MAIN_FORCE_MN          = 123_624.9084
BASELINE_COUNTER_FORCE_EACH_MN  = 61_812.4542
TOTAL_PAD_FORCE_MN              = BASELINE_MAIN_FORCE_MN + 2.0 * BASELINE_COUNTER_FORCE_EACH_MN

# Pad force-ratio options (main : combined counter). 50:50 reproduces the validated
# 2:1:1 baseline (main = 2·counter_each). main_fraction = main share of the total.
FORCE_RATIO_OPTIONS = {"40:60": 0.40, "50:50": 0.50, "60:40": 0.60}

# Anatomical anchor display labels (match sweep_counter_position / sweep_distal_counter_position)
_PROX_ANCHOR_LABEL = {"just_above_curve": "Just above", "mid": "Mid", "sub_axillary": "Sub-axillary"}
_DIST_ANCHOR_LABEL = {"just_below_curve": "Just below", "mid": "Mid", "sacral_anchor": "Sacral"}

# Orthotist-controlled parameters only. Values = cohort-median ABSOLUTE difference
# from baseline in predicted Cobb correction (tornado_orthotist_controlled_summary.csv,
# n = 13). Modelling-decision parameters (main span, force distribution, counter span)
# are fixed in this tool and deliberately excluded from the clinician-facing sensitivity.
_COHORT_DELTAS = [
    ("Proximal counter pad position", 4.87),
    ("Pad force ratio",               2.32),
    ("Distal counter pad position",   2.12),
    ("Pad pressure",                  2.02),
    ("Main pad position",             1.76),
]


def param_delta(sens: dict, name: str):
    deltas = sens.get("deltas") or {}
    if name in deltas and deltas[name] is not None:
        try:
            return abs(float(deltas[name])), "patient"
        except (TypeError, ValueError):
            pass
    cohort = dict(_COHORT_DELTAS)
    val = cohort.get(name)
    return (abs(val) if val is not None else None), "cohort"


def sens_ranking(sens: dict):
    ranked = []
    for name, _ in _COHORT_DELTAS:
        d, src = param_delta(sens, name)
        ranked.append((name, d if d is not None else 0.0, src))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def sens_caption(sens: dict, name: str) -> str:
    d, src = param_delta(sens, name)
    if d is None:
        return ""
    tag = "this patient" if src == "patient" else "cohort median"
    return f"({d:.1f}° max Δ from baseline)"

@st.cache_data
def load_clinical() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "clinical_patient_data.csv", encoding="utf-8-sig")
    df = df.rename(columns={
        "Patient's Number (New System)": "num",
        "Lenke Curve Type (1,2,3,4,5,6)": "lenke",
        "Thoracic Cobb Angle (Before Brace)":      "cobb_t",
        "Lumbar Cobb Angle (Before Brace)":         "cobb_l",
        "Thoraco Lumbar Cobb Angle (Before Brace)": "cobb_tl",
        "Thoracic Cobb Angle (In-Brace)":           "ib_t",
        "Lumbar Cobb Angle (In-Brace)":             "ib_l",
        "Thoraco Lumbar Cobb Angle (In-Brace)":     "ib_tl",
    })
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df = df.dropna(subset=["num"]).copy()
    df["num"] = df["num"].astype(int)
    df["pid"] = df["num"].apply(lambda n: f"P{n:02d}" if n < 100 else f"P{n}")
    return df

@st.cache_data
def load_convexity() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "Curve_convexity_directions.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Patient's Number (New System)": "num",
        "Thoracic Convexity":"cvx_t","Lumbar Convexity":"cvx_l",
        "Thoraco Lumbar Convexity":"cvx_tl",
        "Upper Cobb vertebrae":"upper_v","Lower Cobb vertebrae":"lower_v",
    })
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df = df.dropna(subset=["num"]).copy()
    df["num"] = df["num"].astype(int)
    df["pid"] = df["num"].apply(lambda n: f"P{n:02d}" if n < 100 else f"P{n}")
    for c in ("upper_v","lower_v","cvx_t","cvx_l","cvx_tl"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df

@st.cache_data
def load_patient_sens(pid: str) -> dict:
    path = SENSITIVITY_DIR / pid / "sensitivity_summary.csv"
    if not path.exists():
        return {"deltas": None, "cases": pd.DataFrame(), "source": "cohort"}
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    deltas, in_delta = {}, False
    for ln in lines:
        if "DELTA PER PARAMETER" in ln:
            in_delta = True; continue
        if in_delta and "PER-CASE RESULTS" in ln:
            in_delta = False; continue
        if in_delta and ln.strip() and not ln.startswith("Parameter"):
            parts = ln.strip().split(",")
            if len(parts) >= 2:
                try:
                    deltas[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass
    start = next((i+1 for i,l in enumerate(lines) if "PER-CASE RESULTS" in l), None)
    cases = pd.DataFrame()
    if start is not None:
        cases = pd.read_csv(StringIO("".join(lines[start:])))
        cases["Correction_deg"] = pd.to_numeric(cases["Correction_deg"], errors="coerce")
    return {"deltas": deltas or None, "cases": cases, "source": "patient"}

def curve_len(info) -> int:
    u,l = info.get("upper_v",""), info.get("lower_v","")
    if u in _LIDX and l in _LIDX:
        return abs(_LIDX[u]-_LIDX[l])+1
    return 6

def get_patient_info(pid, clin, conv, baselines) -> dict:
    cr = clin[clin["pid"]==pid]; vr = conv[conv["pid"]==pid]
    if cr.empty: return {}
    cr = cr.iloc[0]; vr = vr.iloc[0] if not vr.empty else None

    upper_v = str(vr["upper_v"]).strip() if vr is not None else ""
    lower_v = str(vr["lower_v"]).strip() if vr is not None else ""
    region = ("Thoracolumbar" if upper_v.startswith("T") and lower_v.startswith("L")
               else "Lumbar" if lower_v.startswith("L") else "Thoracic")

    def _f(c): return pd.to_numeric(cr.get(c), errors="coerce")
    def _pick(*cols):
        for c in cols:
            v = _f(c)
            if not pd.isna(v): return v
        return float("nan")
    def _cvx(*cols):
        if vr is None: return ""
        for c in cols:
            v = str(vr.get(c,"")).strip()
            if v and v.lower() != "nan": return v
        return ""

    if region == "Thoracic":
        cobb = _pick("cobb_t", "cobb_tl", "cobb_l")
        ib   = _pick("ib_t",   "ib_tl",   "ib_l")
        cvx  = _cvx("cvx_t", "cvx_tl", "cvx_l")
    else:
        cobb = _pick("cobb_tl", "cobb_l", "cobb_t")
        ib   = _pick("ib_tl",   "ib_l",   "ib_t")
        cvx  = _cvx("cvx_tl", "cvx_l", "cvx_t")

    apex = ""
    if upper_v in _LIDX and lower_v in _LIDX:
        ui,li = _LIDX[upper_v],_LIDX[lower_v]
        if ui<li: ui,li=li,ui
        apex = SPINE_LEVELS[round((ui+li)/2)]

    bl = baselines.get(pid, {})
    def _bl(*keys, default=None):
        for k in keys:
            if k in bl:
                v = bl.get(k)
                if v is None: continue
                try:
                    if pd.isna(v): continue
                except Exception: pass
                if isinstance(v, str) and not v.strip(): continue
                return v
        return default

    return {
        "pid":pid, "region":region,
        "cobb": None if pd.isna(cobb) else float(cobb),
        "inbrace": None if pd.isna(ib) else float(ib),
        "convexity": cvx.capitalize() if cvx else "Unknown",
        "upper_v": upper_v, "lower_v": lower_v, "apex": apex,
        "main_area_mm2": _bl("main_area_mm2", "pad_area_mm2"),
        "baseline_span": int(_bl("baseline_span", "counter_span", default=2) or 2),
        "baseline_main_span": int(_bl("baseline_main_span", "main_span", default=curve_len({"upper_v": upper_v, "lower_v": lower_v})) or curve_len({"upper_v": upper_v, "lower_v": lower_v})),
        "baseline_main_pos_offset": int(_bl("baseline_main_pos_offset", "main_pos_offset", default=0) or 0),
        "baseline_force_ratio": float(_bl("baseline_force_ratio", "force_ratio", default=0.50) or 0.50),
        "baseline_counter_split": float(_bl("baseline_counter_split", "counter_split", default=0.50) or 0.50),
        "baseline_pressure": _bl("baseline_pressure", "pad_pressure", "pressure_kpa"),
        "baseline_force_distribution": str(_bl("baseline_force_distribution", "force_distribution", default="Uniform") or "Uniform"),
        "baseline_prox_level": _bl("baseline_prox_level", "prox_level", "prox_pos"),
        "baseline_dist_level": _bl("baseline_dist_level", "dist_level", "dist_pos"),
        "baseline_prox_offset": int(_bl("baseline_prox_offset", default=1) or 1),
        "baseline_dist_offset": int(_bl("baseline_dist_offset", default=1) or 1),
    }

def _safe_level(idx: int) -> str:
    return SPINE_LEVELS[max(0, min(idx, len(SPINE_LEVELS) - 1))]

_N_SPINE = len(SPINE_LEVELS)

def _curve_ends(info: dict) -> tuple:
    """Return (upper_i, lower_i) curve-end indices with upper_i (cranial) >= lower_i."""
    upper_i = _LIDX.get(info.get("upper_v", "T9"), 8)
    lower_i = _LIDX.get(info.get("lower_v", "L3"), 2)
    if upper_i < lower_i:
        upper_i, lower_i = lower_i, upper_i
    return upper_i, lower_i

def proximal_anchors(info: dict) -> list:
    """Per-patient proximal counter-pad anchors, IDENTICAL to sweep_counter_position:
    just_above_curve (upper+1); mid (T6 if T6 sits above just_above, else the midpoint
    to T2); sub_axillary (T2). Only anatomically valid, strictly-ascending anchors that
    sit above the upper curve end. Returns [(tag, level), ...] (baseline first)."""
    upper_i, _ = _curve_ends(info)
    t6, t2 = _LIDX["T6"], _LIDX["T2"]
    ja = upper_i + 1
    mid = t6 if t6 > ja else (ja + t2) // 2
    hi = t2
    cand = []
    if 0 <= ja < _N_SPINE:
        cand.append(("just_above_curve", ja))
    if 0 <= mid < _N_SPINE and mid > ja:
        cand.append(("mid", mid))
    if 0 <= hi < _N_SPINE and hi > mid:
        cand.append(("sub_axillary", hi))
    return [(tag, SPINE_LEVELS[idx]) for tag, idx in cand if idx > upper_i]

def distal_anchors(info: dict) -> list:
    """Per-patient distal counter-pad anchors, IDENTICAL to sweep_distal_counter_position:
    just_below_curve (lower-1); mid (L4 if L4 sits below just_below, else midpoint to L5);
    sacral_anchor (L5). Only valid, strictly-descending anchors below the lower curve end.
    Returns [(tag, level), ...] (baseline first)."""
    _, lower_i = _curve_ends(info)
    l5, l4 = 0, _LIDX["L4"]
    jb = lower_i - 1
    mid = l4 if l4 < jb else (jb + l5) // 2
    sac = l5
    raw = [("just_below_curve", jb), ("mid", mid), ("sacral_anchor", sac)]
    out, prev, seen = [], None, set()
    for tag, idx in raw:
        if not (0 <= idx < _N_SPINE):
            continue
        if idx >= lower_i:
            continue
        if prev is not None and idx >= prev:
            continue
        lv = SPINE_LEVELS[idx]
        if lv in seen:
            continue
        out.append((tag, lv)); seen.add(lv); prev = idx
    return out

def baseline_anchor_levels(info: dict) -> tuple:
    """(proximal, distal) baseline counter positions = the just-above / just-below
    anchors (the first anchor each), matching the validated 2:1:1 baseline."""
    pa, da = proximal_anchors(info), distal_anchors(info)
    upper_i, lower_i = _curve_ends(info)
    prox = pa[0][1] if pa else _safe_level(upper_i + 1)
    dist = da[0][1] if da else _safe_level(lower_i - 1)
    return prox, dist

def _anchor_label(tag: str, level: str, kind: str) -> str:
    # Label by the LEVEL's anatomical meaning, not just the sweep tag: the middle anchor
    # is "Intermediate", but when it collapses onto the far anatomical bound (T2 proximal /
    # L5 distal) it is named for that bound — so "Mid (L5)" correctly reads "Sacral (L5)".
    if kind == "prox":
        if level == "T2":                    word = "Sub-axillary"
        elif tag == "just_above_curve":      word = "Just above"
        else:                                word = "Intermediate"
    else:
        if level == "L5":                    word = "Sacral"
        elif tag == "just_below_curve":      word = "Just below"
        else:                                word = "Intermediate"
    return f"{word} ({level})"

def baseline_counter_positions(info: dict, ctr_opts: dict) -> tuple:
    upper_v = info.get("upper_v", "T9")
    lower_v = info.get("lower_v", "L3")
    upper_i = _LIDX.get(upper_v, 8)
    lower_i = _LIDX.get(lower_v, 2)
    if upper_i < lower_i:
        upper_i, lower_i = lower_i, upper_i

    explicit_prox = info.get("baseline_prox_level")
    explicit_dist = info.get("baseline_dist_level")
    target_prox = explicit_prox or _safe_level(upper_i + int(info.get("baseline_prox_offset", 1) or 1))
    target_dist = explicit_dist or _safe_level(lower_i - int(info.get("baseline_dist_offset", 1) or 1))

    def closest(target: str, options: list) -> str:
        if not options: return target
        if target in options: return target
        ti = _LIDX.get(target, 999)
        return min(options, key=lambda v: abs(_LIDX.get(v, 999) - ti))

    prox = closest(target_prox, ctr_opts.get("proximal", []))
    dist = closest(target_dist, ctr_opts.get("distal", []))
    return prox, dist

def run_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)

def exact_baseline_fe_params(pid: str) -> dict:
    return dict(
        pid=pid, area_mult=1.0, ratio_pct=0.0, K_RZ=_K_RZ, pos_offset=0,
        span_extra=0, main_pos_offset=0, main_span=None, distal_span=None,
        distal_offset=0, cal_factor=1.0, counter_split=0.5,
    )

def is_at_ui_baseline(info: dict, main_fraction, main_span,
                      prox_pos, dist_pos, ctr_span, main_pos, pressure, force_dist) -> bool:
    """True only when every design control still holds its baseline value:
    50:50 force ratio, counter positions at the just-above / just-below anchors,
    main pad at apex over the full curve, neutral pressure, uniform force."""
    prox_base, dist_base = baseline_anchor_levels(info)
    region = info.get("region", "Thoracic")
    _, p_mid, _ = PRESSURE_RANGE.get(region, PRESSURE_RANGE["Thoracic"])

    return (
        abs(float(main_fraction) - 0.5) < 1e-6
        and int(main_span) == int(curve_len(info))
        and str(prox_pos) == str(prox_base)
        and str(dist_pos) == str(dist_base)
        and int(ctr_span) == int(info.get("baseline_span", 2) or 2)
        and int(main_pos) == 0
        and abs(float(pressure) - float(p_mid)) < 1e-4
        and str(force_dist) == "Uniform"
    )

def get_counter_options(sens: dict, info: dict) -> dict:
    cases = sens.get("cases", pd.DataFrame())
    upper_v = info.get("upper_v","T9")
    lower_v = info.get("lower_v","L3")
    upper_i = _LIDX.get(upper_v, 8)
    lower_i = _LIDX.get(lower_v, 2)
    if upper_i < lower_i: upper_i, lower_i = lower_i, upper_i

    def _options(param, reverse=False):
        if cases.empty: return []
        rows = cases[cases["Parameter"]==param]["TestedValue"].dropna().astype(str).str.strip()
        vals = [v for v in rows.unique() if v and v != "nan"]
        try:
            vals.sort(key=lambda v: _LIDX.get(v, 999), reverse=reverse)
        except Exception: pass
        return vals

    prox = _options("Proximal counter pad position", reverse=True)
    dist = _options("Distal counter pad position", reverse=False)
    span = sorted([int(float(v)) for v in _options("Counter pad span") if v.replace(".","").isdigit()])

    if not prox:
        prox = [SPINE_LEVELS[min(upper_i+1, len(SPINE_LEVELS)-1)],
                SPINE_LEVELS[min(upper_i+2, len(SPINE_LEVELS)-1)],
                SPINE_LEVELS[min(upper_i+3, len(SPINE_LEVELS)-1)]]
    if not dist:
        dist = [SPINE_LEVELS[max(lower_i-1, 0)],
                SPINE_LEVELS[max(lower_i-2, 0)],
                SPINE_LEVELS[max(lower_i-3, 0)]]
    if not span: span = [1, 2, 3]

    return {"proximal": prox, "distal": dist, "span": span}

def reset_sliders_to_baseline(pid: str, info: dict, ctr_opts: dict):
    region = info.get("region", "Thoracic")
    # Pad area source back to the standard baseline area (not patient-specific)
    st.session_state["area_src_w"] = f"Baseline Area ({BASELINE_AREA:.0f} mm²)"
    # Force ratio back to the 50:50 baseline (validated 2:1:1)
    st.session_state["force_ratio_w"] = "50:50"
    st.session_state["mainpos_w"] = "At apex"

    # Counter positions back to the just-above / just-below baseline anchors
    prox_default, dist_default = baseline_anchor_levels(info)
    st.session_state["prox_anchor_w"] = prox_default
    st.session_state["dist_anchor_w"] = dist_default

    p_lo, p_mid, p_hi = PRESSURE_RANGE.get(region, (6.0, 7.5, 10.0))
    st.session_state["pres_w"] = min(max(float(p_mid), p_lo), p_hi)
    st.session_state["_pres_pid"] = pid

    for k in ("fe_result", "fe_stats", "_pending", "_last_run_sig"):
        st.session_state.pop(k, None)

def map_fe_params(info, main_fraction, prox_pos, dist_pos,
                   ctr_span, main_pos, main_span, pressure, force_dist,
                   pad_area_mm2, use_patient_area=True) -> dict:
    region      = info.get("region","Thoracic")
    upper_i, lower_i = _curve_ends(info)

    # Use selected area source as baseline for area_mult calculation
    if use_patient_area:
        baseline_area = info.get("main_area_mm2") or BASELINE_AREA
    else:
        baseline_area = BASELINE_AREA  # Standard baseline for all patients

    area_mult  = max(0.5, min(2.0, pad_area_mm2 / baseline_area)) if baseline_area else 1.0
    if abs(area_mult - 1.0) < 1e-6:
        area_mult = 1.0

    curve_l = abs(upper_i - lower_i) + 1
    effective_main_span = None if (int(main_span) >= curve_l and main_pos == 0) else int(main_span)

    # ratio_pct flags an off-baseline force ratio for the baseline-detection
    # predicate (=0 at 50:50). counter_split stays 0.5 — the counter share is
    # always split EQUALLY between proximal and distal, as in sweep_force_ratio.
    main_fraction = float(main_fraction)
    ratio_pct     = round(200.0 * (main_fraction - 0.5), 1)
    counter_split = 0.5

    # Counter-pad offsets, matching the sweep's anchor placement exactly:
    # the proximal pad starts at upper+1+pos_offset, so pos_offset = anchor-(upper+1);
    # the distal pad top sits at lower-1-distal_offset, so distal_offset = (lower-1)-anchor.
    prox_i        = _LIDX.get(prox_pos, upper_i + 1)
    pos_offset    = max(0, prox_i - (upper_i + 1))
    dist_i        = _LIDX.get(dist_pos, lower_i - 1)
    distal_offset = max(0, (lower_i - 1) - dist_i)

    span_extra    = max(0, ctr_span - 2)
    distal_span_v = None if ctr_span == 2 else ctr_span

    _, p_mid, _ = PRESSURE_RANGE.get(region, PRESSURE_RANGE['Thoracic'])
    pressure_mult = pressure / p_mid
    effective_area_mult = area_mult * pressure_mult
    # Snap tiny float noise to exactly 1.0 so baseline mode stays reachable
    if abs(effective_area_mult - 1.0) < 1e-3:
        effective_area_mult = 1.0

    # Derive the three pad forces EXACTLY as sweep_force_ratio does: hold the total
    # 3-pad force fixed, give the main pad its fraction, split the remaining counter
    # share equally between the two counters. The pressure slider (and pad-area
    # source) scale the magnitude via `effective_area_mult`, so the two reconcile as
    #   main          = TOTAL · main_fraction       · effective_area_mult
    #   counter_each  = TOTAL · (1-main_fraction)/2 · effective_area_mult
    # At 50:50 + baseline pressure this is exactly main=123624.9, counter=61812.45.
    mag = effective_area_mult
    main_force_mn         = TOTAL_PAD_FORCE_MN * main_fraction * mag
    counter_force_each_mn = TOTAL_PAD_FORCE_MN * (1.0 - main_fraction) / 2.0 * mag

    return dict(
        pid=info["pid"],
        area_mult=round(effective_area_mult, 4),
        ratio_pct=ratio_pct,
        K_RZ=_K_RZ,
        pos_offset=pos_offset,
        span_extra=span_extra,
        main_pos_offset=int(main_pos),
        main_span=effective_main_span,
        distal_span=distal_span_v,
        distal_offset=distal_offset,
        cal_factor=1.0,
        counter_split=float(counter_split),
        main_force_mn=round(main_force_mn, 4),
        counter_force_each_mn=round(counter_force_each_mn, 4),
    )

def parse_marc_log(pid: str) -> dict:
    out = RUN_ROOT / pid / f"{pid}_ui.out"
    t16 = RUN_ROOT / pid / f"{pid}_ui.t16"
    if not out.exists(): return {}
    try:
        lines = out.read_text(encoding="latin-1", errors="ignore").splitlines()
    except Exception: return {}
    last_inc, run_date, exit_code = 0, "", ""
    for ln in lines:
        s = ln.strip()
        if "output for increment" in s and "lcase" in s:
            try: last_inc = int(s.split("increment")[1].split(".")[0].strip())
            except: pass
        if "date:" in s and "202" in s:
            run_date = s.split("date:")[-1].strip()
        if "Exit number" in s:
            exit_code = s.split()[-1]
    return {"inc":last_inc,"date":run_date,"exit":exit_code,"out":str(out),"t16":str(t16)}

def _spine_fig(info, main_pos, prox_pos, dist_pos, main_span, ctr_span,
               correction=None) -> plt.Figure:
    import matplotlib.transforms as mtransforms
    upper_v = info.get("upper_v") or "T9"
    lower_v = info.get("lower_v") or "L2"
    apex_v  = info.get("apex")    or "T11"
    cvx     = info.get("convexity", "Right")
    cobb    = float(info.get("cobb") or 30)
    if upper_v not in _LIDX or lower_v not in _LIDX:
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=10); ax.axis("off"); return fig

    ui, li = _LIDX[upper_v], _LIDX[lower_v]
    if ui < li: ui, li = li, ui
    api  = _LIDX.get(apex_v, round((ui + li) / 2))
    ctri = _LIDX.get(prox_pos, ui + 1)
    dctr = _LIDX.get(dist_pos, max(0, li - 1))
    cvs  = +1.0 if "Right" in cvx else -1.0

    PITCH = 0.92
    VW, VH = 0.44, 0.34
    span_lv = max(1, ui - li)
    amp = min(1.7, max(0.9, cobb / 30.0))

    def curve_x(lv_i, a):
        if li <= lv_i <= ui:
            t = (lv_i - li) / span_lv
            return cvs * a * math.sin(math.pi * t)
        return curve_x(li, a) if lv_i < li else curve_x(ui, a)

    def tilt(lv_i, a):
        if not (li <= lv_i <= ui): return 0.0
        half = (cobb / 2.0) * (a / max(amp, 1e-6))
        t = (lv_i - li) / span_lv
        # Negative so the upper and lower end-vertebra endplates tilt TOWARD each
        # other and their extension lines converge (true Cobb construction).
        return -half * (2.0 * t - 1.0)

    # Show the FULL spine (every vertebra C1..L5), matching the FE model, so the
    # schematic reads as a complete spine and fills the column vertically.
    disp_bot = 0
    disp_top = len(SPINE_LEVELS) - 1
    disp = SPINE_LEVELS[disp_bot:disp_top + 1]
    n = len(disp)
    def yco(i): return (i - disp_bot) * PITCH

    fig, ax = plt.subplots(figsize=(5.0, 7.5))
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.set_aspect("equal")

    def draw_spine(a, faint=False):
        for lv in disp:
            lv_i = _LIDX[lv]; x0 = curve_x(lv_i, a); y0 = yco(lv_i); ang = tilt(lv_i, a)
            if faint:
                fc, ec_c, lw, al = "#F2E9F7", AFTER, 0.8, 0.55
            else:
                region_fc = "#FCEFE0" if lv.startswith("L") else ("#E7F0FA" if lv.startswith("T") else "#F4F4F4")
                if lv_i in (ui, li): fc, ec_c, lw, al = NAVY, NAVY, 1.3, 1.0
                elif li < lv_i < ui: fc, ec_c, lw, al = region_fc, STEEL, 0.9, 1.0
                else: fc, ec_c, lw, al = region_fc, "#C9CDD2", 0.7, 1.0
            box = mpatches.FancyBboxPatch((x0 - VW/2, y0 - VH/2), VW, VH,
                boxstyle="round,pad=0.015,rounding_size=0.08",
                fc=fc, ec=ec_c, lw=lw, alpha=al, zorder=2 if faint else 3)
            box.set_transform(mtransforms.Affine2D().rotate_deg_around(x0, y0, ang) + ax.transData)
            ax.add_patch(box)
            if not faint:
                ax.text(x0 - VW/2 - 0.22, y0, lv, ha="right", va="center", fontsize=8.5,
                        color=(RED if lv_i == api else NAVY if lv_i in (ui, li) else "#6B7280"),
                        fontweight="bold" if lv_i in (ui, li, api) else "normal")

    corrected = correction is not None and abs(correction) > 0.05
    a_after = amp * max(0.05, 1.0 - min(0.9, correction / max(cobb, 1.0))) if corrected else amp

    # Spine: the PRE-BRACE curve is the solid/prominent spine; after a run the predicted
    # IN-BRACE (corrected/straightened) curve is overlaid as the faint purple ghost — same
    # purple as the in-brace Cobb lines — so the reduction reads at a glance. Before a run,
    # just the pre-brace spine.
    if corrected:
        draw_spine(a_after, faint=True)        # predicted in-brace ghost (purple)
    cl = list(range(disp_bot, disp_top + 1))
    ax.plot([curve_x(i, amp) for i in cl], [yco(i) for i in cl],
            color="#B6C2CE", lw=1.2, zorder=1, solid_capstyle="round")
    draw_spine(amp, faint=False)               # pre-brace spine (solid, prominent)

    # Cobb construction — BOTH the pre-brace (red) and corrected (purple) angles are
    # measured between the SAME two fixed end vertebrae: ui = upper_v, li = lower_v
    # (the levels also passed to _extract_cobb.py). Each angle's two endplate lines
    # start at these identical vertebrae; only the endplate TILT changes with the spine
    # geometry (pre-brace `amp` vs corrected `a_after`), so the before/after comparison
    # is valid. The end vertebrae sit on the central axis (curve_x = 0 at the ends), so
    # the upper/lower anchor points are the same in both overlays.
    _eU = (curve_x(ui, amp), yco(ui))     # upper end-vertebra anchor
    _eL = (curve_x(li, amp), yco(li))     # lower end-vertebra anchor (identical for both)
    _yU, _yL = _eU[1], _eL[1]
    _ymid = (_yU + _yL) / 2.0
    # Both angles share the PRE-BRACE vertex (where the upper & lower end-vertebra endplate
    # lines cross). The pre-brace (red) wedge opens to the full angle with its arms reaching
    # the end vertebrae. The corrected (purple) wedge is the HONEST reduced angle — measured
    # from the corrected endplate tilts at the SAME end vertebrae — drawn NESTED INSIDE the
    # pre-brace wedge so the reduction reads as a smaller angle within the larger one. (A
    # smaller angle's true vertex lies further out, so nesting it at the shared vertex is the
    # only way to keep it inside the initial angle.)
    _du = (math.cos(math.radians(tilt(ui, amp))), math.sin(math.radians(tilt(ui, amp))))
    _dl = (math.cos(math.radians(tilt(li, amp))), math.sin(math.radians(tilt(li, amp))))
    _den = _du[0]*_dl[1] - _du[1]*_dl[0]
    if abs(_den) > 1e-9:
        _s = ((_eL[0]-_eU[0])*_dl[1] - (_eL[1]-_eU[1])*_dl[0]) / _den
        _Vx, _Vy = _eU[0] + _s*_du[0], _eU[1] + _s*_du[1]
    else:
        _Vx, _Vy = _eU[0] - cvs*8.0, _ymid
    _ang_pre = abs(tilt(ui, amp) - tilt(li, amp))           # = cobb (tilt is in degrees)
    _bis = math.atan2(_ymid - _Vy, ((_eU[0]+_eL[0])/2.0) - _Vx)   # vertex -> spine
    _armL = math.hypot(_eU[0] - _Vx, _eU[1] - _Vy)
    _verts = [(_Vx, _Vy)]

    def _wedge(angle_deg, col, lw, r_arc, label_r):
        h = math.radians(angle_deg / 2.0)
        for sgn in (1, -1):
            aa = _bis + sgn * h
            ax.plot([_Vx, _Vx + (_armL + 0.55)*math.cos(aa)],
                    [_Vy, _Vy + (_armL + 0.55)*math.sin(aa)],
                    color=col, lw=lw, zorder=7, solid_capstyle="round")
        ax.add_patch(mpatches.Arc((_Vx, _Vy), 2*r_arc, 2*r_arc,
                     theta1=math.degrees(_bis - h), theta2=math.degrees(_bis + h),
                     color=col, lw=1.0, zorder=8))
        ax.text(_Vx + math.cos(_bis)*label_r, _Vy + math.sin(_bis)*label_r,
                f"{angle_deg:.1f}°", ha="center", va="center",
                fontsize=11, color=col, fontweight="600", zorder=9)

    if corrected:
        _ang_cor = abs(tilt(ui, a_after) - tilt(li, a_after))   # honest corrected Cobb angle
        _wedge(_ang_pre, RED,   1.7, 1.0, _armL*0.62)   # pre-brace (outer, arms to end vertebrae)
        _wedge(_ang_cor, AFTER, 1.7, 0.6, _armL*0.40)   # corrected — nested INSIDE the pre-brace
    else:
        _wedge(_ang_pre, RED, 1.8, 1.0, _armL*0.62)

    GAP, PAD_W = 0.10, 0.22
    def draw_pad(center_i, n_levels, on_convex, fc, ec):
        cx0 = curve_x(center_i, amp); cy0 = yco(center_i)
        # height scales clearly with the covered levels so span changes are obvious
        h = max(0.8, min(n_levels * 0.8, span_lv + 1.5)) * PITCH
        side = cvs if on_convex else -cvs
        x_left = cx0 + VW/2 + GAP if side > 0 else cx0 - VW/2 - GAP - PAD_W
        ax.add_patch(mpatches.FancyBboxPatch((x_left, cy0 - h/2), PAD_W, h,
            boxstyle="round,pad=0.005,rounding_size=0.1", fc=fc, ec=ec, lw=1.0, alpha=0.9, zorder=4))
        ax_tail = x_left + (PAD_W if side < 0 else 0)
        ax_head = cx0 + side * (VW/2 + 0.02)
        ax.annotate("", xy=(ax_head, cy0), xytext=(ax_tail, cy0),
                    arrowprops=dict(arrowstyle="-|>", color=ec, lw=1.1, mutation_scale=8), zorder=5)

    pci = max(li, min(ui, api + int(main_pos)))
    draw_pad(pci, main_span, True, GREEN, TEAL)
    for pad_i in (ctri, dctr):
        if disp_bot <= pad_i <= disp_top:
            draw_pad(pad_i, ctr_span, False, STEEL, NAVY)

    handles = [mpatches.Patch(fc=GREEN, ec=TEAL, label="Main pad"),
               mpatches.Patch(fc=STEEL, ec=NAVY, label="Counter pads")]
    if corrected:
        handles.append(mpatches.Patch(fc="#F2E9F7", ec=AFTER, label="Predicted in-brace"))
    ax.legend(handles=handles, loc="lower center", ncol=len(handles),
              fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    title = f"{info.get('pid','')}  ·  Initial Cobb Angle = {cobb:.1f}°"
    subtitle = "Coronal plane · pre-brace configuration"
    if corrected:
        subtitle = "Coronal plane · custom configuration"
    fig.suptitle(title, fontsize=14.5, fontweight="700", color=DARK_TEXT, y=0.992)
    fig.text(0.5, 0.967, subtitle, ha="center", va="top", fontsize=11, color=MID_GREY)

    # Frame to include the spine AND every Cobb vertex (either side), with a small pad
    _spine_x = amp + VW/2 + PAD_W + 1.2
    _vxs = [v[0] for v in _verts] or [0.0]
    ax.set_xlim(min(-_spine_x, min(_vxs) - 1.0), max(_spine_x, max(_vxs) + 1.0))
    ax.set_ylim(-0.6, (n-1)*PITCH + 0.6)
    # Reserve the top strip for the two-line title so the spine never collides with it
    ax.axis("off"); fig.tight_layout(rect=[0, 0, 1, 0.955])
    return fig

def _cobb_fig(pre_brace, fe_result, clinical_ib) -> plt.Figure:
    # Compare the CORRECTION (degrees removed), not the absolute in-brace angle.
    rows = []
    if clinical_ib is not None:
        rows.append(("Clinical", round(float(pre_brace) - float(clinical_ib), 1), STEEL))
    if fe_result is not None:
        rows.append(("Predicted", round(float(fe_result), 1), TEAL))

    if not rows:
        fig,ax = plt.subplots(figsize=(5.0, 2.0))
        ax.text(0.5,0.5,"Run FE to compute", ha="center",va="center",
                transform=ax.transAxes, fontsize=11,color=MID_GREY); ax.axis("off"); return fig

    labels=[r[0] for r in rows]; vals=[r[1] for r in rows]
    cols=[r[2] for r in rows]
    x_max = max(max(vals)*1.3, 10)

    fig,ax = plt.subplots(figsize=(5.0, 2.2))
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    bars = ax.barh(range(len(rows)), vals, color=cols, edgecolor="white", lw=1.0, height=0.62)
    for i,(bar,val) in enumerate(zip(bars,vals)):
        bw = bar.get_width(); inside = bw > x_max*0.30
        ax.text(bw-0.4 if inside else bw+0.3, i, f"{val:.1f}°", va="center",
                ha="right" if inside else "left", fontsize=13.5, fontweight="700",
                color="white" if inside else DARK_TEXT)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=12.5, color=DARK_TEXT, fontweight="600")
    ax.set_xlabel("Cobb-angle correction (°)", fontsize=11.5, color=MID_GREY)
    ax.set_title("Predicted vs clinical correction",
                 fontsize=13.5, fontweight="700", color=DARK_TEXT, pad=8)
    ax.set_xlim(0, x_max); ax.set_ylim(-0.5, len(rows)-0.5)
    ax.grid(axis="x", color=BORDER, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=MID_GREY, labelsize=10.5)
    fig.tight_layout(); return fig

_CSS = f"""<style>
* {{ box-sizing: border-box; }}
/* Fluid base type so the UI fills the viewport instead of looking tiny */
html {{ font-size: clamp(15px, 1.0vw + 9px, 20px) !important; }}
html, body, [class*="css"], [class*="st-"], body, div, span, p, label, button {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #1a1a1a !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 0.25rem 0 !important;
    line-height: 1.3 !important;
}}
h2 {{ font-size: 18px !important; }}
h3 {{
    font-size: 18.5px !important; font-weight: 600 !important;
    letter-spacing: 0.2px !important; color: #212529 !important;
    padding-bottom: 12px !important; border-bottom: 1px solid #E5E5E5 !important;
    margin: 0 0 30px !important;
}}
p, div, span, label {{
    font-size: 15px !important;
    color: #404040 !important;
    line-height: 1.45 !important;
}}

.block-container {{
    max-width: 100% !important; width: 100% !important; margin: 0 auto !important;
    padding: 0.8rem 1.6rem 0.4rem 1.6rem !important;
    background: #FFFFFF !important;
}}
/* Let matplotlib images fill their column */
[data-testid="stImage"], [data-testid="stImage"] img {{ max-width: 100% !important; }}

/* Cap the SCHEMATIC (middle column image only) to the viewport so the whole app
   fits on screen without vertical scrolling. The image keeps its aspect ratio
   (width auto) and stays centred; the value subtracts the banner + headings +
   footer + padding from the viewport height. Sidebar + comparison charts unaffected. */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stImage"] img {{
    max-height: calc(100vh - 300px) !important;
    width: auto !important;
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
}}

/* ---- Vertical dividers between the 3 main columns (prototype look) ---- */
/* Streamlit columns stretch to equal height, so a right border reads as a
   full-height divider between Design Parameters | Brace Configuration | Results.
   Zero the flex GAP so the border is not pushed to one side, then give every
   column the SAME symmetric padding -> the divider sits dead-centre with an
   identical 28px gutter on both sides, for all three columns. */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]) {{
    gap: 0 !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    padding-left: 28px !important;
    padding-right: 28px !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:not(:last-child) {{
    border-right: 1px solid #E5E5E5 !important;
}}
/* Clear Streamlit's top toolbar so the banner title is never clipped */
[data-testid="stAppViewContainer"] .main .block-container {{
    padding-top: 0.8rem !important;
}}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent !important; border-bottom: none !important; }}

[data-testid="stSidebar"] {{
    background: #F2F6FB !important;
    border-right: 1px solid #D6E0EC !important;
}}
[data-testid="stSidebar"] h2 {{
    font-size: 14px !important;
    color: #1a1a1a !important;
    margin: 0.5rem 0 0.4rem 0 !important;
    font-weight: 500 !important;
}}
.sb-title {{ font-size: 18px !important; font-weight: 700 !important; color: #1F4E79 !important; margin-bottom: 2px !important; }}
.sb-sub {{ font-size: 12.5px !important; color: #6C757D !important; margin-bottom: 10px !important; }}
/* Collapsed sidebar toggle -> one solid navy vertical tab (arrow on top, label below),
   matching the prototype. Flex column so the ::after label sits INSIDE the navy bar. */
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{
    background: #1F4E79 !important;
    border-radius: 0 7px 7px 0 !important;
    padding: 11px 7px 16px 7px !important;
    margin: 8px 0 0 0 !important;
    position: relative !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.25) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-start !important;
}}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {{
    background: transparent !important; border: none !important; box-shadow: none !important;
    padding: 0 !important; margin: 0 !important;
}}
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg {{
    color: #FFFFFF !important; fill: #FFFFFF !important; width: 1.4rem !important; height: 1.4rem !important;
}}
[data-testid="stSidebarCollapsedControl"]::after {{
    content: "PATIENT & SENSITIVITY";
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    white-space: nowrap;
    color: #FFFFFF; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    margin-top: 12px;
}}

.stButton > button {{
    background: #1F4E79 !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 9px 18px !important;
}}
.stButton > button:hover {{
    background: #163A5A !important;
}}
/* Force white text on ALL buttons (Run FE Model, Replay correction, Reset) */
.stButton button, .stButton button * {{ color: #FFFFFF !important; }}

/* Segmented control (Main Pad Position): keep all three options on ONE line.
   Streamlit 1.58 renders this as data-testid="stButtonGroup". */
/* ---- Clean navy segmented bar (Pad force ratio, Main pad position, counter
   positions). Selected = solid navy + white text; unselected = white + dark text;
   thin grey border; rounded ends; every label on ONE line, never truncated. ---- */
[data-testid="stButtonGroup"] {{
    display: flex !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    gap: 0 !important;
}}
[data-testid="stButtonGroup"] button {{
    flex: 1 1 auto !important;          /* size to content, share leftover width */
    min-width: 0 !important;
    height: auto !important;
    padding: 6px 8px !important;
    font-size: 11px !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;     /* single line */
    overflow: visible !important;
    text-overflow: clip !important;
    border: 1px solid #D8DEE6 !important;
    border-radius: 0 !important;
    background: #FFFFFF !important;
    color: #212529 !important;
    box-shadow: none !important;
}}
[data-testid="stButtonGroup"] button:not(:first-child) {{ border-left: none !important; }}
[data-testid="stButtonGroup"] button:first-child {{
    border-top-left-radius: 6px !important; border-bottom-left-radius: 6px !important;
}}
[data-testid="stButtonGroup"] button:last-child {{
    border-top-right-radius: 6px !important; border-bottom-right-radius: 6px !important;
}}
[data-testid="stButtonGroup"] button p,
[data-testid="stButtonGroup"] button span,
[data-testid="stButtonGroup"] button div {{
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    color: inherit !important;
}}
/* selected segment: solid navy, white text (covers aria + Streamlit's active kinds) */
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[data-selected="true"],
[data-testid="stButtonGroup"] button[kind*="ctive"] {{
    background: #1F4E79 !important;
    border-color: #1F4E79 !important;
    color: #FFFFFF !important;
}}
[data-testid="stButtonGroup"] button[aria-checked="true"] *,
[data-testid="stButtonGroup"] button[aria-pressed="true"] *,
[data-testid="stButtonGroup"] button[data-selected="true"] *,
[data-testid="stButtonGroup"] button[kind*="ctive"] * {{ color: #FFFFFF !important; }}

/* Compact base vertical rhythm; the columns then distribute their content to
   fill the full (stretched) column height via space-between below. */
[data-testid="stVerticalBlock"] {{ gap: 0.7rem !important; }}

/* Vertical alignment of the 3 columns. Every column's content block fills the full
   (equal) column height. Cols 1 (Design Parameters) and 3 (Results) DISTRIBUTE their
   content evenly top-to-bottom (space-between) so any spare height is shared as small
   even gaps between controls — NOT dumped into one big gap above the last button. The
   middle (schematic) column has no spacer: its figure is the height driver. */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{
    height: 100% !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {{
    justify-content: space-between !important;
}}
.stSlider {{ padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; margin-bottom: 0 !important; }}
.stSlider > div {{ padding-bottom: 0 !important; }}
.stRadio {{ margin-bottom: 0 !important; }}
[data-testid="stWidgetLabel"] {{ margin-top: 0.6rem !important; margin-bottom: 0.15rem !important; }}
[data-testid="stCaptionContainer"] {{ margin-top: 0 !important; margin-bottom: 0.05rem !important; }}
/* Control labels: uppercase, 11px, muted (mockup .sublabel) */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
    font-size: 13px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.6px !important;
    color: #6C757D !important;
}}

/* Debug Information — deliberately understated so it doesn't compete with results:
   borderless/light, small muted-grey uppercase summary, faint content. */
[data-testid="stExpander"] {{
    border: none !important;
    border-top: 1px solid #F1F3F5 !important;
    overflow: visible !important;
    margin: 0.8rem 0 0.2rem 0 !important;
    opacity: 0.9 !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > div > button {{
    font-size: 10.5px !important;
    font-weight: 500 !important;
    color: #AEB6BF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 8px 0 !important;
}}
/* Faint, small content text inside the expander */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
    font-size: 11px !important;
    color: #AEB6BF !important;
    line-height: 1.4 !important;
}}
/* Hide the broken Material-icon ligature text and draw a real chevron glyph. */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] .st-emotion-cache-0 svg {{ display: none !important; }}
[data-testid="stExpanderToggleIcon"],
[data-testid="stIconMaterial"],
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] {{
    font-size: 0 !important; width: 1em !important; height: 1em !important;
    position: relative !important; color: transparent !important;
}}
[data-testid="stExpanderToggleIcon"]::after,
[data-testid="stIconMaterial"]::after {{
    content: "\\25B8" !important; font-size: 12px !important; color: #AEB6BF !important;
    position: absolute !important; left: 0 !important; top: 50% !important;
    transform: translateY(-50%) !important;
}}
[data-testid="stExpander"] details[open] [data-testid="stExpanderToggleIcon"]::after,
[data-testid="stExpander"] details[open] [data-testid="stIconMaterial"]::after {{
    content: "\\25BE" !important;
}}

.stMetric {{
    background: #FFFFFF !important;
    border: none !important;
    border-top: 1px solid #E5E5E5 !important;
    border-radius: 0 !important;
    padding: 12px 0 !important;
    margin: 0 !important;
}}
.stMetric label {{
    font-size: 12.5px !important;
    color: #666 !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
}}

.label-small {{
    font-size: 12.5px !important;
    color: #666 !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
    margin: 0.5rem 0 0.2rem 0 !important;
}}

.divider {{
    border-top: 1px solid #E5E5E5 !important;
    margin: 1rem 0 !important;
}}

.result-row {{
    display: flex;
    gap: 40px;
    align-items: flex-start;
    padding: 20px 0;
    border-top: 1px solid #E5E5E5;
}}
.result-col {{
    flex: 1;
}}
.result-label {{
    font-size: 12.5px !important;
    color: #666 !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.result-value {{
    font-size: 33px !important;
    font-weight: 700 !important;
    color: #1F4E79 !important;
    line-height: 1.1 !important;
}}
.result-divider {{
    width: 1px;
    background: #E5E5E5;
    min-height: 80px;
}}

/* Predicted-correction box: dedicated classes (with !important) so the big teal
   number isn't shrunk by the global p/div/span font-size rule. */
.corr-label {{
    font-size: 12.5px !important; font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important; color: #6C757D !important;
}}
.corr-value {{
    font-size: 30px !important; font-weight: 700 !important; color: #2A9D8F !important;
    line-height: 1.15 !important;
}}
.corr-pct {{
    font-size: 15px !important; font-weight: 400 !important; color: #6C757D !important;
}}

/* ---- Banner text (class rules override the global div colour) ---- */
.app-banner-title {{
    font-size: 29px !important; font-weight: 700 !important; color: #FFFFFF !important;
    letter-spacing: -0.2px !important; line-height: 1.1 !important;
}}
.app-banner-sub {{
    font-size: 15.5px !important; color: #BFD2E4 !important; font-weight: 400 !important;
    margin: 6px 0 0 !important;
}}
.app-banner-inst {{
    font-size: 11.5px !important; color: #7FA6C9 !important; letter-spacing: 0.5px !important;
    margin: 7px 0 0 !important;
}}

/* ---- Mobile: stack the three columns vertically and shrink the banner ---- */
@media (max-width: 820px) {{
    [data-testid="stHorizontalBlock"] {{
        flex-direction: column !important;
        gap: 0.4rem !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="column"],
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        border-right: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}
    .block-container {{ padding: 0.4rem 0.6rem !important; }}
    .app-banner-title {{ font-size: 21px !important; }}
    .app-banner-sub   {{ font-size: 12px !important; }}
}}
@media (max-width: 480px) {{
    .app-banner-title {{ font-size: 18px !important; }}
}}
</style>"""

def main():
    st.set_page_config(
        page_title="Brace Design Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)

    clin_df   = load_clinical()
    conv_df   = load_convexity()
    baselines = load_patient_baselines()
    available = sorted(baselines.keys())

    if not available:
        st.error("No patients found in Patient specific data.csv"); return

    with st.sidebar:
        st.markdown(
            "<div class='sb-title'>Patient &amp; Sensitivity</div>"
            "<div class='sb-sub'>Select a patient and view parameter influence</div>",
            unsafe_allow_html=True)

        def _patient_label(p):
            pi = get_patient_info(p, clin_df, conv_df, baselines)
            r = pi.get("region", ""); c = pi.get("cobb")
            return f"{p} · {r} · {c:.0f}°" if c is not None else f"{p} · {r}"

        pid = st.selectbox("Select patient", available,
                           index=available.index("P14") if "P14" in available else 0,
                           key="pid_sel", format_func=_patient_label)

        info = get_patient_info(pid, clin_df, conv_df, baselines)
        sens = load_patient_sens(pid)
        ctr_opts = get_counter_options(sens, info)

        _required_keys = ("force_ratio_w", "mainpos_w",
                          "prox_anchor_w", "dist_anchor_w", "pres_w")
        if (st.session_state.get("_fe_pid") != pid
                or any(k not in st.session_state for k in _required_keys)):
            reset_sliders_to_baseline(pid, info, ctr_opts)
            st.session_state["_fe_pid"] = pid

        # Reset-to-baseline button (in col1) sets this flag; apply it BEFORE the widgets render
        if st.session_state.pop("_do_reset", False):
            reset_sliders_to_baseline(pid, info, ctr_opts)

        region = info.get("region","Thoracic")
        cobb   = info.get("cobb")
        ib     = info.get("inbrace")
        clinical_correction = round(cobb - ib, 1) if cobb is not None and ib is not None else None

        # ---- Patient data card ----
        _cobb_txt = f"{cobb:.1f}°" if cobb is not None else "N/A"
        _ib_txt   = f"{ib:.1f}° (Δ {clinical_correction:+.1f}°)" if ib is not None else "N/A"
        _rows = [
            ("Pre-brace", _cobb_txt),
            ("In-brace",  _ib_txt),
            ("Curve",     f"{info.get('upper_v','-')} → {info.get('lower_v','-')} ({curve_len(info)}v)"),
            ("Apex",      info.get('apex','-') or "-"),
            ("Pattern",   f"{info.get('convexity','-')}-convex"),
        ]
        _rows_html = "".join(
            "<div style='display:flex; justify-content:space-between; font-size:12.5px; margin:3px 0;'>"
            f"<span style='color:#6C757D;'>{k}</span>"
            f"<span style='color:#212529; font-weight:600;'>{v}</span></div>"
            for k, v in _rows
        )
        st.markdown(
            "<div style='background:#F7F9FB; border:1px solid #E5E5E5; border-left:3px solid #1F4E79; "
            "border-radius:4px; padding:13px 15px; margin:10px 0 16px;'>"
            "<div style='font-size:10.5px; font-weight:600; text-transform:uppercase; "
            "letter-spacing:0.5px; color:#6C757D; margin-bottom:6px;'>Patient Data</div>"
            + _rows_html + "</div>",
            unsafe_allow_html=True)

        # ---- Parameter sensitivity ----
        st.markdown("<div class='label-small' style='margin-bottom:0;'>Parameter Sensitivity</div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='font-size:11px; color:#6C757D; margin:0 0 6px;'>"
                    "Absolute difference from baseline in predicted Cobb correction (°), ranked</div>",
                    unsafe_allow_html=True)

        # Full labels, identical to the published tornado figure, for consistency.
        _SENS_SHORT = {
            "Proximal counter pad position": "Proximal counter pad position",
            "Pad force ratio": "Pad force ratio",
            "Pad pressure": "Pad pressure",
            "Main pad position": "Main pad position",
            "Distal counter pad position": "Distal counter pad position",
        }
        _ranked = sens_ranking(sens)[:7]
        names  = [_SENS_SHORT.get(nm, nm) for nm, _, _ in _ranked]
        deltas = [d for _, d, _ in _ranked]
        ypos   = list(range(len(names)))[::-1]   # biggest at top
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")
        ax.barh(ypos, deltas, color=STEEL, height=0.62)
        ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=10.5, color=DARK_TEXT)
        ax.set_xlim(0, (max(deltas)*1.25) if deltas else 10)
        for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#E5E5E5")
        ax.tick_params(colors="#666", labelsize=9.5, length=0)
        ax.grid(axis="x", color="#E5E5E5", lw=0.5, alpha=0.5); ax.set_axisbelow(True)
        for y, val in zip(ypos, deltas):
            ax.text(val + (max(deltas)*0.02 if deltas else 0.1), y, f"{val:.1f}°",
                    va="center", fontsize=9.5, color="#444")
        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown(
        """<div style='background:#003E74; color:#fff; text-align:center;
                       padding:10px 20px; margin:0 0 8px 0; border-radius:0;'>
            <div class='app-banner-title'>Scoliosis Brace Design Explorer</div>
            <div class='app-banner-sub'>Predicting in-brace spinal correction before the brace is built</div>
            <div class='app-banner-inst'>IMPERIAL COLLEGE LONDON · DEPARTMENT OF BIOENGINEERING</div>
        </div>""",
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(f"<h3 style='margin-top:0;'>Design Parameters</h3>", unsafe_allow_html=True)

        st.markdown(f"<div class='label-small'>Pad Area Source</div>", unsafe_allow_html=True)

        area_source = st.radio(
            "Area source",
            options=[f"Baseline Area ({BASELINE_AREA:.0f} mm²)", "Patient-Specific Area"],
            horizontal=True,
            key="area_src_w",
            label_visibility="collapsed"
        )

        baseline_area = info.get("main_area_mm2")
        if "Baseline" in area_source:
            pad_area = BASELINE_AREA
            area_note = "(standard baseline)"
        else:
            pad_area = float(baseline_area) if baseline_area else BASELINE_AREA
            area_note = f"(patient {pid})"

        st.markdown(f"<div class='label-small'>Selected Area</div>", unsafe_allow_html=True)
        st.caption(f"**{pad_area:,.0f} mm²** {area_note}")

        # Pad force ratio (main : combined counter). Single radio styled like the Pad
        # Area Source control. 50:50 = validated 2:1:1 baseline; the counter share is
        # always split equally between the proximal and distal pads (sweep_force_ratio).
        st.markdown(f"<div class='label-small' style='margin-top:1rem;'>Pad force ratio (main:counter)</div>", unsafe_allow_html=True)
        ratio_choice = st.segmented_control(
            "Pad force ratio",
            options=list(FORCE_RATIO_OPTIONS.keys()),
            key="force_ratio_w",
            label_visibility="collapsed",
            help="Main : combined counter force share. 50:50 reproduces the validated baseline.") or "50:50"
        main_fraction = FORCE_RATIO_OPTIONS[ratio_choice]

        p_lo, p_mid, p_hi = PRESSURE_RANGE[region]
        if "pres_w" not in st.session_state or st.session_state.get("_pres_pid") != pid:
            st.session_state["pres_w"] = float(p_mid)
            st.session_state["_pres_pid"] = pid
        st.markdown(f"<div class='label-small' style='margin-top:1rem;'>Pad Pressure ({region}, kPa)</div>", unsafe_allow_html=True)
        pressure = st.slider("Pad pressure", float(p_lo), float(p_hi), float(p_mid),
                             0.01, key="pres_w", label_visibility="collapsed")

        st.markdown(f"<div class='label-small' style='margin-top:1rem;'>Main Pad Position (relative to apex)</div>", unsafe_allow_html=True)
        main_pos_lbl = st.segmented_control(
            "Main pad position",
            ["Apex − 1", "At apex", "Apex + 1"],
            key="mainpos_w", label_visibility="collapsed") or "At apex"
        main_pos = {"Apex − 1":-1,"At apex":0,"Apex + 1":1}[main_pos_lbl]

        # Proximal counter pad position — anatomical anchors (sweep_counter_position),
        # only the valid anchors for this patient. Default = just-above-curve baseline.
        _prox_anchors = proximal_anchors(info)
        _prox_levels  = [lv for _, lv in _prox_anchors] or [baseline_anchor_levels(info)[0]]
        _prox_labels  = {lv: _anchor_label(tag, lv, "prox") for tag, lv in _prox_anchors}
        st.markdown(f"<div class='label-small' style='margin-top:1rem;'>Proximal counter pad position</div>", unsafe_allow_html=True)
        prox_pos = st.segmented_control(
            "Proximal counter pad position", _prox_levels,
            format_func=lambda lv: _prox_labels.get(lv, lv),
            key="prox_anchor_w", label_visibility="collapsed") or _prox_levels[0]

        # Distal counter pad position — anatomical anchors (sweep_distal_counter_position).
        _dist_anchors = distal_anchors(info)
        _dist_levels  = [lv for _, lv in _dist_anchors] or [baseline_anchor_levels(info)[1]]
        _dist_labels  = {lv: _anchor_label(tag, lv, "dist") for tag, lv in _dist_anchors}
        st.markdown(f"<div class='label-small' style='margin-top:1rem;'>Distal counter pad position</div>", unsafe_allow_html=True)
        dist_pos = st.segmented_control(
            "Distal counter pad position", _dist_levels,
            format_func=lambda lv: _dist_labels.get(lv, lv),
            key="dist_anchor_w", label_visibility="collapsed") or _dist_levels[0]

        # Modelling-decision parameters (main pad span, counter pad span, force
        # distribution) stay fixed at their validated baseline and hidden. The FE
        # backend still receives them so the solve and the schematic stay correct.
        main_span = curve_len(info)
        ctr_span = max(1, min(3, int(info.get("baseline_span", 2) or 2)))
        force_dist = "Uniform"

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Reset to baseline", key="reset_btn", use_container_width=True):
            st.session_state["_do_reset"] = True
            st.rerun()

    # ---- FE parameter mapping (computed before col2 so the schematic can show straightening) ----
    use_patient_area = "Patient-Specific" in area_source
    mapped_fe_p = map_fe_params(info, float(main_fraction), prox_pos, dist_pos,
                                int(ctr_span), int(main_pos), int(main_span),
                                float(pressure), str(force_dist), float(pad_area),
                                use_patient_area=use_patient_area)

    baseline_mode = (
        ("Baseline" in area_source)
        and is_at_ui_baseline(
            info, float(main_fraction), main_span, prox_pos, dist_pos,
            ctr_span, main_pos, pressure, force_dist
        )
    )
    fe_p = exact_baseline_fe_params(pid) if baseline_mode else mapped_fe_p
    current_sig = run_signature(fe_p)

    fe_result = st.session_state.get("fe_result", None)
    fe_stats  = st.session_state.get("fe_stats",  {})

    # A stored FE result is only valid for the controls it was RUN with. If any control
    # has changed since the last run (current_sig != the saved _last_run_sig), the result
    # is STALE: we must not show it against the new configuration. pred_valid gates the
    # whole prediction (purple overlay + predicted values + correction + comparison).
    _run_sig   = st.session_state.get("_last_run_sig")
    pred_stale = fe_result is not None and _run_sig is not None and _run_sig != current_sig
    pred_valid = fe_result is not None and not pred_stale

    # Comparison chart uses an illustrative estimate so both bars always show.
    if info.get("cobb") is not None and info.get("inbrace") is not None:
        _illustrative = max(0.0, float(info["cobb"]) - float(info["inbrace"]))
    else:
        _illustrative = None
    # Spine corrected (purple) overlay + angle: shown ONLY for a VALID (in-sync) FE run.
    _spine_correction = float(fe_result) if pred_valid else None

    with col2:
        st.markdown(f"<h3 style='margin-top:0;'>Brace Configuration</h3>", unsafe_allow_html=True)
        spine_ph = st.empty()
        fig_s = _spine_fig(info, int(main_pos), prox_pos, dist_pos, int(main_span), int(ctr_span),
                           correction=_spine_correction)
        spine_ph.pyplot(fig_s, use_container_width=True); plt.close(fig_s)
        if _spine_correction is not None and abs(_spine_correction) > 0.05:
            if st.button("Replay correction", key="animate_btn", use_container_width=True):
                import time
                _steps = 14
                for _k in range(_steps + 1):
                    _c = _spine_correction * _k / _steps
                    _f = _spine_fig(info, int(main_pos), prox_pos, dist_pos,
                                    int(main_span), int(ctr_span), correction=_c)
                    spine_ph.pyplot(_f, use_container_width=True); plt.close(_f)
                    time.sleep(0.07)

    with col3:
        st.markdown(f"<h3 style='margin-top:0;'>Results</h3>", unsafe_allow_html=True)

        _mode_txt = ("Baseline mode: using validated baseline model" if baseline_mode
                     else "Custom mode: FE model computed with current parameters")
        st.markdown(
            "<div style='background:#F2F6FB; border-left:3px solid #1F4E79; border-radius:4px; "
            "padding:7px 12px; margin:2px 0 12px; font-size:11.5px; color:#5A6B7B;'>"
            f"{_mode_txt}</div>", unsafe_allow_html=True)

        # Stale-prediction notice: controls changed since the displayed run.
        if pred_stale:
            st.markdown(
                "<div style='background:#FFF4E6; border-left:3px solid #E76F51; border-radius:4px; "
                "padding:7px 12px; margin:2px 0 12px; font-size:11.5px; color:#A1542F;'>"
                "Parameters changed — re-run FE Model to update the prediction</div>",
                unsafe_allow_html=True)

        # Predicted vs Clinical (only when the stored result matches the current controls)
        if pred_valid:
            # cobb - fe_result: when fe_result < 0 this is LARGER than pre-brace (curve worsened) -> show it
            pred_val = f"{round(float(cobb) - fe_result, 1) + 0.0:.1f}°"
            pred_col = NAVY if fe_result >= 0 else ORANGE
        else:
            pred_val = "-"
            pred_col = "#9EB3C2"

        if clinical_correction is not None and cobb is not None:
            clin_val = f"{round(float(cobb) - clinical_correction, 1) + 0.0:.1f}°"
            clin_col = NAVY
        else:
            clin_val = "-"
            clin_col = "#9EB3C2"

        st.markdown(
            f"""<div class='result-row'>
              <div class='result-col'>
                <div class='result-label'>Predicted in-brace</div>
                <div class='result-value'>{pred_val}</div>
              </div>
              <div class='result-divider'></div>
              <div class='result-col'>
                <div class='result-label'>Clinical (X-ray)</div>
                <div class='result-value'>{clin_val}</div>
              </div>
            </div>""",
            unsafe_allow_html=True)

        # Predicted correction box (positive magnitude + percent)
        if pred_valid and cobb:
            corr_mag = abs(float(fe_result))
            corr_pct = corr_mag / float(cobb) * 100.0 if cobb else 0.0
            _verb = "reduction" if fe_result >= 0 else "increase"
            st.markdown(
                "<div style='background:#F7F9FB; border-left:3px solid #2A9D8F; border-radius:4px; "
                "padding:10px 14px; margin:8px 0 6px;'>"
                "<div class='corr-label'>Predicted correction</div>"
                f"<div class='corr-value'>{corr_mag:.1f}° "
                f"<span class='corr-pct'>({corr_pct:.1f}% {_verb})</span></div>"
                "</div>",
                unsafe_allow_html=True)

        if pred_valid and clinical_correction is not None:
            gap = (fe_result - clinical_correction) + 0.0
            status = "Within ±5° threshold" if abs(gap) <= 5.0 else f"Gap: {gap:+.1f}°"
            st.caption(f"Model error: {gap:+.1f}° ({status})")

        if fe_stats and not baseline_mode and pred_valid:
            st.caption(
                f"Marc run: {fe_stats.get('date','N/A')} | "
                f"Increments: {fe_stats.get('inc','?')} | "
                f"Exit code: {fe_stats.get('exit','?')}")

        if st.button("Run FE Model", type="primary", use_container_width=True):
            with st.spinner(f"Running Marc for {pid}..."):
                try:
                    result = run_real_model(**fe_p)
                    st.session_state["fe_result"] = float(result)
                    st.session_state["fe_stats"]  = parse_marc_log(pid)
                    st.session_state["_last_run_sig"] = current_sig
                    st.rerun()
                except Exception as e:
                    import traceback as _tb
                    st.error(f"FE run failed: {e}")
                    with st.expander("Error details", expanded=False):
                        st.code(_tb.format_exc())

        if cobb is not None:
            _eff_pred = fe_result if pred_valid else _illustrative
            fig_c = _cobb_fig(float(cobb), _eff_pred,
                               float(info["inbrace"]) if info.get("inbrace") is not None else None)
            st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

        with st.expander("Debug Information", expanded=False):
            st.write(f"Baseline mode: {baseline_mode}")
            st.write(f"Reference model (source): Updated automated/{pid}")
            if baseline_mode:
                st.write("Run: reads validated baseline .t16 (no re-solve)")
            else:
                st.write(f"Run: baseline .dat + adjustments, solved in C:\\Temp\\marc_ui_runs/{pid}")

    st.markdown(
        "<p style='text-align:center; font-size:11px; color:#6C757D; margin-top:8px; margin-bottom:0;'>"
        "Proof of concept: predictions shown are illustrative and not for clinical use.</p>",
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
