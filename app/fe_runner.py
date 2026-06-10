"""
fe_runner.py: real Marc FE model evaluation behind the Streamlit UI.

Each time the "Run FE model" button is pressed this:
  1. Reads the patient's geometry parameters from the Updated automated metadata CSV
  2. Recomputes pad centroids and RBE3 connectivity if the position or span changed
  3. Copies the baseline .dat to a local temp folder (outside OneDrive)
  4. Replaces the pad forces (area x ratio), K_rz, centroid coords and RBE3 tables
  5. Runs the Marc solver via run_marc.bat
  6. Reads the .t16 with py_post and returns the Cobb-angle correction in degrees

All five UI parameters feed through into the FE model. There's no Mentat batch
step, the .dat is edited directly in Python.

Units in the .dat: mN, mm, rad
"""

from __future__ import annotations

import csv
import shutil
import math
import re
import subprocess
from pathlib import Path


# Configuration (paths and thread counts, edit these for a different machine)

MARC_BAT = Path(
    r"C:\Program Files\MSC.Software\Marc\2024.1.0\marc2024.1\tools\run_marc.bat"
)

BASELINE_ROOT = Path(
    r"C:\Users\<username>\OneDrive - Imperial College London\Year 4\FYP"
    r"\Patient_1_final\Automated models\Updated automated"
)

# Marc output folder – outside OneDrive to avoid sync overhead
RUN_ROOT = Path(r"C:\Temp\marc_ui_runs")

# Thread counts (set to logical-core count - 2)
NTS = 14
NTE = 14

# Full spine definition used by Updated automated models (24 levels, L5 → C1)
SPINE_LEVELS_BOTTOM_TO_TOP: list[str] = [
    "L5", "L4", "L3", "L2", "L1",
    "T12", "T11", "T10", "T9", "T8", "T7", "T6",
    "T5", "T4", "T3", "T2", "T1",
    "C7", "C6", "C5", "C4", "C3", "C2", "C1",
]
_N_LEVELS = len(SPINE_LEVELS_BOTTOM_TO_TOP)        # 24
_N_SPINE_NODES = 1 + _N_LEVELS + (_N_LEVELS - 1)  # 48  (1 base + 24 bone + 23 disc)
_LEVEL_TO_IDX = {lv: i for i, lv in enumerate(SPINE_LEVELS_BOTTOM_TO_TOP)}

# Pad centroid node IDs in the .dat (spine nodes 1-48, centroids 49-51)
_CENTROID_NODE = {1: 49, 2: 50, 3: 51}   # pad number → Marc node ID

# Module-level cache for last FE run info
_LAST_RUN_INFO: dict = {}


# Marc float helpers (Marc stores floats in a fixed-width text format)

def _mf(v: float) -> str:
    """
    Format a float in Marc .dat scientific notation, always 20 chars:
      positive:  ' 4.000000000000000+8'
      negative:  '-6.181245400000000+4'
    """
    if v == 0.0:
        return " 0.000000000000000+0"
    sign = " " if v >= 0 else "-"
    av = abs(v)
    exp = int(math.floor(math.log10(av)))
    m = av / 10.0 ** exp
    if m >= 10.0 - 1e-12:
        m /= 10.0; exp += 1
    elif m < 1.0 - 1e-12:
        m *= 10.0; exp -= 1
    es = "+" if exp >= 0 else "-"
    return f"{sign}{m:.15f}{es}{abs(exp)}"


def _parse_mf(s: str) -> float:
    """Parse Marc .dat notation '1.23456789+5' → 1.23456789e5."""
    s = s.strip()
    s = re.sub(r"([0-9])([+-])(\d+)$", r"\1e\2\3", s)
    return float(s)


# Spine geometry, rebuilt the same way as the coordinate generator in the FE pipeline

def _bone_angles(cobb_deg: float, convexity: str,
                 upper_cobb: str, lower_cobb: str) -> list[float]:
    upper_idx = _LEVEL_TO_IDX[upper_cobb]
    lower_idx = _LEVEL_TO_IDX[lower_cobb]
    if upper_idx < lower_idx:
        upper_idx, lower_idx = lower_idx, upper_idx
    sign = +1.0 if convexity == "Right" else -1.0
    half = cobb_deg / 2.0
    angles = [0.0] * _N_LEVELS
    for i in range(lower_idx, upper_idx + 1):
        t = (i - lower_idx) / max(1, upper_idx - lower_idx)
        angles[i] = sign * (half * (1.0 - t) + (-half) * t)
    return angles


def _gen_nodes(angles: list[float],
               bone_step: float, disc_step: float) -> list[tuple[int, float, float]]:
    nodes: list[tuple[int, float, float]] = []
    nid = 1
    x = 400.0
    y = 650.0
    nodes.append((nid, x, y))
    for i, deg in enumerate(angles):
        rad = math.radians(deg)
        x += math.tan(rad) * bone_step
        y -= bone_step
        nid += 1
        nodes.append((nid, x, y))
        if i < len(angles) - 1:
            disc_deg = 0.5 * (deg + angles[i + 1])
            x += math.tan(math.radians(disc_deg)) * disc_step
            y -= disc_step
            nid += 1
            nodes.append((nid, x, y))
    return nodes


def _pad_regions(upper_cobb: str, lower_cobb: str,
                 counter_span: int, counter_offset: int,
                 main_pos_offset: int = 0, main_span: int | None = None,
                 distal_span: int | None = None,
                 distal_offset: int | None = None) -> dict:
    """Return main/distal/proximal index lists with clamping.
    distal_span / distal_offset fall back to the proximal values when None.
    """
    upper_idx = _LEVEL_TO_IDX[upper_cobb]
    lower_idx = _LEVEL_TO_IDX[lower_cobb]
    if upper_idx < lower_idx:
        upper_idx, lower_idx = lower_idx, upper_idx

    if main_span is not None and main_span > 0:
        apex = round((upper_idx + lower_idx) / 2) + main_pos_offset
        apex = max(lower_idx, min(upper_idx, apex))
        half = (main_span - 1) // 2
        bot  = max(lower_idx, apex - half)
        top  = min(upper_idx, apex + (main_span - 1 - half))
        main_indices = list(range(bot, top + 1)) or [apex]
    else:
        main_indices = list(range(lower_idx, upper_idx + 1))

    # Proximal counter pad
    pspan = max(1, counter_span)
    d_prox = int(counter_offset)
    bb = upper_idx + 1
    pb = min((_N_LEVELS - 1) - (pspan - 1), max(bb + d_prox, bb))
    pt = min(_N_LEVELS - 1, pb + (pspan - 1))
    proximal = [i for i in range(pb, pt + 1) if upper_idx < i < _N_LEVELS]
    if not proximal and upper_idx + 1 < _N_LEVELS:
        proximal = [upper_idx + 1]

    # Distal counter pad – independent span/offset when provided
    dspan  = max(1, distal_span   if distal_span  is not None else counter_span)
    d_dist = int(distal_offset    if distal_offset is not None else counter_offset)
    bt = lower_idx - 1
    dt = max(dspan - 1, min(bt - d_dist, bt))
    dt = max(0, dt)
    db = max(0, dt - (dspan - 1))
    distal = [i for i in range(db, dt + 1) if 0 <= i < lower_idx]
    if not distal and lower_idx - 1 >= 0:
        distal = [lower_idx - 1]

    return {"main": main_indices, "distal": distal, "proximal": proximal}


def _vertebra_node_ids(level_index: int) -> tuple[int, int]:
    return 2 * level_index + 1, 2 * level_index + 2


def _nodes_for_indices(indices: list[int]) -> list[int]:
    ids: list[int] = []
    for i in indices:
        ids.extend(_vertebra_node_ids(i))
    return sorted(set(ids))


def _centroid_of(nodes_list: list[tuple[int, float, float]],
                 node_ids: list[int]) -> tuple[float, float]:
    coord = {nid: (x, y) for nid, x, y in nodes_list}
    xs = [coord[n][0] for n in node_ids]
    ys = [coord[n][1] for n in node_ids]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _compute_pads(nodes: list[tuple[int, float, float]],
                  convexity: str, upper_cobb: str, lower_cobb: str,
                  counter_span: int, counter_offset: int,
                  pad_x_offset: float,
                  main_pos_offset: int = 0,
                  main_span: int | None = None,
                  distal_span: int | None = None,
                  distal_offset: int | None = None) -> dict[int, dict]:
    """
    Returns {pad_num: {centroid_x, centroid_y, dep_nodes, sign_main, sign_ctr}}
    sign_main / sign_ctr are +1 or -1 applied to the force magnitudes.
    """
    regions = _pad_regions(upper_cobb, lower_cobb, counter_span, counter_offset,
                           main_pos_offset, main_span,
                           distal_span, distal_offset)
    cvx = +1.0 if convexity == "Right" else -1.0

    defs: dict[int, dict] = {}
    for pad_num, role, indices in [
        (1, "distal",   regions["distal"]),
        (2, "main",     regions["main"]),
        (3, "proximal", regions["proximal"]),
    ]:
        dep = _nodes_for_indices(indices)
        cx, cy = _centroid_of(nodes, dep)
        if role == "main":
            cx += cvx * pad_x_offset
            force_sign = -cvx     # opposes convex side
        else:
            cx -= cvx * pad_x_offset
            force_sign = +cvx     # supports from concave side
        defs[pad_num] = {
            "cx": cx, "cy": cy,
            "dep_nodes": dep,
            "force_sign": force_sign,
        }
    return defs


# Editing the .dat directly: pad forces, stiffness, centroid coords and RBE3 tables

def _rbe3_block(name: str, centroid_node: int, dep_nodes: list[int]) -> list[str]:
    """Build the lines for one RBE3 entry in the .dat."""
    lines = [
        f"         0         0         1         0         0         0         1{name}",
        "         1",
        f"          {centroid_node}",
        " 1.000000000000000+0        12         0",
    ]
    # 13 node IDs per line, continuation marker if more follow
    for i in range(0, len(dep_nodes), 13):
        chunk = dep_nodes[i:i + 13]
        row = "".join(f"{n:12d}" for n in chunk)
        if i + 13 < len(dep_nodes):
            row += "   c"
        lines.append(row)
    return lines


def _modify_dat(src: Path, dst: Path,
                main_force_mn: float,
                proximal_force_mn: float,
                distal_force_mn: float,
                krz: float,
                new_centroids: dict[int, tuple[float, float]] | None = None,
                new_dep_nodes: dict[int, list[int]] | None = None,
                force_signs: dict[int, float] | None = None) -> None:
    """
    Copy src .dat to dst with replacements:
      - pad forces (main=pad2, proximal=pad3, distal=pad1), signs preserved
      - K_rz bushing stiffness
      - (optional) pad centroid node coordinates
      - (optional) RBE3 dependent node lists
    """
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = list(lines)

    # ── 1. Forces ──────────────────────────────────────────────────────────────
    # The BC definition header looks like "   1   0   0   0   0   0pad1_force"
    # The loadcase activations are bare "pad1_force" lines – must NOT match those.
    # Requiring at least one char before "pad" skips the bare-name lines.
    pad_re = re.compile(r".+pad(\d+)_force\s*$")
    for i, line in enumerate(lines):
        m = pad_re.search(line)
        if m and i + 1 < len(lines):
            pad_num = int(m.group(1))
            # Determine sign: from force_signs dict or from baseline
            if force_signs and pad_num in force_signs:
                sign = force_signs[pad_num]
            else:
                tok = lines[i + 1].strip().split()[0] if lines[i + 1].strip() else ""
                sign = -1.0 if tok.startswith("-") else +1.0
            if pad_num == 2:
                mag = main_force_mn
            elif pad_num == 3:
                mag = proximal_force_mn
            else:               # pad_num == 1 (distal counter)
                mag = distal_force_mn
            val = sign * mag
            out[i + 1] = f"{_mf(val)} 0.000000000000000+0 0.000000000000000+0"

    # ── 2. K_rz ───────────────────────────────────────────────────────────────
    # Find the bushing stiffness line: KX KY KRZ on a single line after "geom_discs"
    in_geom_discs = False
    for i, line in enumerate(lines):
        if "geom_discs" in line:
            in_geom_discs = True
            continue
        if in_geom_discs:
            # Skip the intermediate line (starts with " -1 " or similar)
            if line.strip().startswith("-1") or line.strip().startswith("-1."):
                continue
            # The stiffness line has 3 Marc floats: KX KY KRZ
            tokens = line.split()
            if len(tokens) >= 3:
                try:
                    kx = _parse_mf(tokens[0])
                    ky = _parse_mf(tokens[1])
                    out[i] = f"{_mf(kx)}{_mf(ky)}{_mf(krz)}"
                    in_geom_discs = False
                    break
                except (ValueError, IndexError):
                    pass

    # ── 3. Centroid coordinates ────────────────────────────────────────────────
    if new_centroids:
        for pad_num, (cx, cy) in new_centroids.items():
            nid = _CENTROID_NODE[pad_num]
            target = f"{nid:10d}"
            new_line = f"{target}{_mf(cx)}{_mf(cy)}{_mf(0.0)}"
            for i, line in enumerate(lines):
                if len(line) >= 10 and line[:10] == target:
                    out[i] = new_line
                    break

    # ── 4. RBE3 connectivity ──────────────────────────────────────────────────
    if new_dep_nodes:
        # Find the rbe3 section (header line "rbe3")
        rbe3_start = next((i for i, l in enumerate(lines) if l.strip() == "rbe3"), None)
        if rbe3_start is not None:
            # Find end: first line after the RBE3 blocks that's a known section header
            rbe3_end = rbe3_start + 1
            while rbe3_end < len(lines):
                stripped = lines[rbe3_end].strip()
                if stripped in ("no print", "post") or stripped.startswith("no print"):
                    break
                rbe3_end += 1
            # Build replacement RBE3 section
            new_rbe3 = ["rbe3", ""]
            for pad_num in sorted(new_dep_nodes.keys()):
                new_rbe3.extend(
                    _rbe3_block(f"rbe3_{pad_num}", _CENTROID_NODE[pad_num],
                                new_dep_nodes[pad_num])
                )
            out[rbe3_start:rbe3_end] = new_rbe3

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out), encoding="utf-8")


# Finding the right patient folder and its files (copes with P08, P8 and 8 naming)

def _patient_id_variants(pid: str) -> list[str]:
    """
    Return likely folder/file naming variants for a patient ID.
    Examples:
      P08 -> P08, P8, 08, 8, Patient_8, Patient 8
      P195 -> P195, 195, Patient_195
    """
    pid = pid.upper().strip()
    variants = [pid]

    m = re.search(r"(\d+)", pid)
    if m:
        n = int(m.group(1))
        variants.extend([
            f"P{n:02d}",
            f"P{n}",
            f"{n:02d}",
            f"{n}",
            f"Patient_{n}",
            f"Patient {n}",
            f"patient_{n}",
            f"patient {n}",
        ])

    # preserve order, remove duplicates
    return list(dict.fromkeys(variants))


def _resolve_patient_dir(pid: str) -> Path:
    """
    Resolve the patient folder inside BASELINE_ROOT.

    This guarantees every patient is read from:
      BASELINE_ROOT / patient-specific folder

    It does not use the Streamlit temp run folder as the source of truth.
    """
    variants = _patient_id_variants(pid)

    # 1) Direct folder names
    for v in variants:
        p = BASELINE_ROOT / v
        if p.exists() and p.is_dir():
            return p

    # 2) Case-insensitive direct child match
    children = [p for p in BASELINE_ROOT.iterdir() if p.is_dir()]
    by_lower = {p.name.lower(): p for p in children}
    for v in variants:
        hit = by_lower.get(v.lower())
        if hit is not None:
            return hit

    # 3) Folder containing the patient's metadata or dat file
    wanted_names = set()
    for v in variants:
        wanted_names.add(f"{v}_model_metadata.csv".lower())
        wanted_names.add(f"{v}.dat".lower())
        wanted_names.add(f"{v}_spine.csv".lower())

    for f in BASELINE_ROOT.rglob("*"):
        if f.is_file() and f.name.lower() in wanted_names:
            return f.parent

    raise FileNotFoundError(
        f"Could not resolve patient folder for {pid} inside:\n{BASELINE_ROOT}\n\n"
        f"Tried variants: {variants}"
    )


def _find_patient_file(patient_dir: Path, pid: str, suffix: str,
                       exact_patterns: list[str] | None = None) -> Path:
    """
    Find a patient-specific source file from the validated baseline folder.

    suffix examples: '.dat', '_spine.csv', '_model_metadata.csv'
    """
    variants = _patient_id_variants(pid)

    patterns = exact_patterns or []
    for v in variants:
        if suffix.startswith("_"):
            patterns.append(f"{v}{suffix}")
        else:
            patterns.append(f"{v}{suffix}")

    # 1) Exact file in the resolved patient folder
    for pat in patterns:
        p = patient_dir / pat
        if p.exists():
            return p

    # 2) Exact file recursively in any procedure subfolder
    lower_patterns = {p.lower() for p in patterns}
    matches = [
        f for f in patient_dir.rglob("*")
        if f.is_file() and f.name.lower() in lower_patterns
    ]
    if matches:
        # Prefer files not from UI/sensitivity/temp folders
        def score(p: Path):
            s = str(p).lower()
            bad = int(("ui" in s) or ("sens" in s) or ("sensitivity" in s) or ("temp" in s))
            depth = len(p.relative_to(patient_dir).parts)
            return (bad, depth, str(p).lower())
        return sorted(matches, key=score)[0]

    # 3) Fallback by suffix if there is only one sensible file
    matches = [
        f for f in patient_dir.rglob(f"*{suffix}")
        if f.is_file()
    ]
    if matches:
        def score(p: Path):
            s = str(p).lower()
            stem = p.stem.lower()
            contains_pid = any(v.lower() in stem for v in variants)
            bad = int(("ui" in s) or ("sens" in s) or ("sensitivity" in s) or ("temp" in s))
            depth = len(p.relative_to(patient_dir).parts)
            return (bad, not contains_pid, depth, str(p).lower())
        return sorted(matches, key=score)[0]

    raise FileNotFoundError(
        f"Could not find {suffix} file for {pid} inside:\n{patient_dir}"
    )


# Reading the metadata and mapping each vertebra level to its node numbers

def _read_metadata(folder: Path, pid: str) -> dict:
    path = _find_patient_file(folder, pid, "_model_metadata.csv")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty metadata: {path}")
    row = {k.strip(): (v or "").strip() for k, v in rows[0].items()}
    return {
        "cobb_deg":       float(row["ClinicalPreBraceCobb_deg"]),
        "convexity":      row["Convexity"],
        "lower_end":      row["LowerModelCurveEnd"].upper(),
        "upper_end":      row["UpperModelCurveEnd"].upper(),
        "span":           int(row.get("CounterPadLevelsEachSide", 2)),
        "pad_x_offset":   float(row.get("PadXOffset_mm", 35.0)),
        "bone_step":      float(row.get("BoneVerticalStep_mm", 21.48)),
        "disc_step":      float(row.get("DiscVerticalStep_mm", 6.00)),
        "main_force":     float(row.get("MainPadForce_N", 123_624.9084)),
        "counter_force":  float(row.get("CounterForceEach_N",  61_812.4542)),
        "_metadata_path": str(path),
    }


def _build_vmap(spine_csv: Path) -> dict:
    with open(spine_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    vmap = {}
    for i, label in enumerate(SPINE_LEVELS_BOTTOM_TO_TOP):
        bot = 2 * i + 1
        top = 2 * i + 2
        if top <= n:
            vmap[label] = {"bot": bot, "top": top}
    return vmap


# Cobb measurement, run in a subprocess (reason explained just below)

# Marc ships Python 3.11; the Streamlit app runs Python 3.14.
# py_post is a native extension compiled for 3.11 and cannot be loaded in 3.14.
# Solution: run _extract_cobb.py in Marc's bundled Python 3.11 as a subprocess.

_MARC_PYTHON = Path(
    r"C:\Program Files\MSC.Software\Marc\2024.1.0\mentat2024.1\python\WIN8664\python.exe"
)
_HELPER = Path(__file__).parent / "_extract_cobb.py"


def _cobb_from_t16_subprocess(
    t16: Path, spine_csv: Path, upper_v: str, lower_v: str
) -> tuple[float, float]:
    """
    Call _extract_cobb.py under Marc's Python 3.11 to read the .t16 post file.
    Returns (initial_cobb_deg, final_cobb_deg).
    """
    if not _MARC_PYTHON.exists():
        raise FileNotFoundError(
            f"Marc Python 3.11 not found:\n{_MARC_PYTHON}\n"
            "Check the Marc installation path in fe_runner.py."
        )

    result = subprocess.run(
        [str(_MARC_PYTHON), str(_HELPER),
         str(t16), str(spine_csv), upper_v, lower_v],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Cobb extraction failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    parts = result.stdout.strip().split()
    if len(parts) != 2:
        raise RuntimeError(
            f"Unexpected output from _extract_cobb.py: {result.stdout!r}"
        )
    return float(parts[0]), float(parts[1])


# Spotting when a requested run is just the stored baseline, so it can be reused

def _is_exact_baseline_payload(
    area_mult: float,
    ratio_pct: float,
    pos_offset: int,
    span_extra: int,
    main_pos_offset: int,
    main_span: int | None,
    distal_span: int | None,
    distal_offset: int | None,
    cal_factor: float,
    counter_split: float,
) -> bool:
    """
    True when the Streamlit UI is asking for the untouched baseline model.

    Important:
    - The true baseline should NOT trigger centroid/RBE3 rebuilding.
    - The true baseline should NOT rewrite forces/stiffness unless necessary.
    - This is what lets the UI reproduce the baseline Excel results.
    """
    # Treat 0 and None as equivalent for the offset/span params: both mean
    # "no perturbation". app.exact_baseline_fe_params sends distal_offset=0,
    # so requiring `is None` here would wrongly force a re-solve at baseline.
    return (
        abs(float(area_mult) - 1.0) < 1e-9
        and abs(float(ratio_pct)) < 1e-9
        and int(pos_offset) == 0
        and int(span_extra) == 0
        and int(main_pos_offset) == 0
        and (main_span is None)
        and (distal_span is None)
        and (distal_offset is None or int(distal_offset) == 0)
        and abs(float(cal_factor) - 1.0) < 1e-9
        and abs(float(counter_split) - 0.5) < 1e-9
    )


def _find_existing_baseline_t16(patient_dir: Path, pid: str) -> Path | None:
    """
    Find the validated baseline .t16 for this patient.

    Searches the patient folder AND its subfolders (the automated baseline often
    writes into a procedure subfolder). Excludes UI temp copies (PXX_ui.t16),
    sensitivity copies, and anything under a temp run folder, so a previous failed
    UI run can never be mistaken for the validated baseline. Prefers files whose
    name contains the patient id.
    """
    pid_l = pid.lower()
    bad_tokens = ("_ui", "ui_", "sens", "sensitivity", "temp", "tmp", "marc_ui_runs")

    candidates = [p for p in patient_dir.rglob("*.t16") if p.is_file()]
    if not candidates:
        return None

    def is_temp(p: Path) -> bool:
        s = str(p).lower()
        stem = p.stem.lower()
        return any(tok in stem for tok in bad_tokens) or "marc_ui_runs" in s

    clean = [p for p in candidates if not is_temp(p)]
    pool = clean or candidates  # if everything looks temp, fall back to all

    # rank: pid-named first, then shallower path, then newest
    def score(p: Path):
        stem = p.stem.lower()
        has_pid = (pid_l in stem)
        try:
            depth = len(p.relative_to(patient_dir).parts)
        except Exception:
            depth = 99
        return (not has_pid, depth, -p.stat().st_mtime)

    return sorted(pool, key=score)[0]


def _copy_baseline_side_files(patient_dir: Path, run_dir: Path, pid: str, job_id: str) -> None:
    """
    Copy representative baseline side files to the temp run folder so the UI log
    panel has local paths. The actual baseline source remains patient_dir.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    for ext in (".dat", ".proc.dat", ".out", ".sts", ".log", ".t16"):
        try:
            if ext == ".t16":
                src = _find_existing_baseline_t16(patient_dir, pid)
            elif ext == ".dat":
                src = _find_patient_file(patient_dir, pid, ".dat")
            else:
                matches = [p for p in patient_dir.rglob(f"*{ext}") if p.is_file()]
                src = sorted(matches, key=lambda p: -p.stat().st_mtime)[0] if matches else None

            if src is not None:
                dst = run_dir / f"{job_id}{ext}"
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
        except Exception:
            pass


# Public API: the functions app.py imports and calls

def run_real_model(
    pid: str,
    area_mult: float,
    ratio_pct: float,
    K_RZ: float,
    pos_offset: int = 0,
    span_extra: int = 0,
    main_pos_offset: int = 0,
    main_span: int | None = None,
    distal_span: int | None = None,
    distal_offset: int | None = None,
    cal_factor: float = 1.0,
    counter_split: float = 0.5,
    main_force_mn: float | None = None,
    counter_force_each_mn: float | None = None,
) -> float:
    """
    Run the Marc FE model for patient `pid` with all five UI parameters applied.

    main_force_mn / counter_force_each_mn : optional explicit pad-force override.
    When both are given (force-ratio control), they replace the computed forces –
    the app has already derived all three pad forces (total held fixed, main gets
    its fraction, counter share split EQUALLY between proximal and distal).

    Parameters
    ----------
    pid         : patient ID, e.g. "P08"
    area_mult   : pad-area multiplier (0.5 – 1.5)
    ratio_pct   : main:counter ratio offset (–40 to +40 %)
    K_RZ        : disc lateral-bending stiffness (mN·mm/rad)
    pos_offset  : counter-pad position (0–3 levels outward from baseline)
    span_extra  : additional counter-pad levels beyond baseline (0–4)

    Returns
    -------
    Cobb-angle correction in degrees  (positive = brace reduces the curve).
    """
    global _LAST_RUN_INFO

    pid = pid.upper().strip()
    patient_dir = _resolve_patient_dir(pid)
    dat_src     = _find_patient_file(patient_dir, pid, ".dat")
    spine_csv   = _find_patient_file(patient_dir, pid, "_spine.csv")

    meta = _read_metadata(patient_dir, pid)

    # ── Exact baseline shortcut ────────────────────────────────────────────────
    # If the UI has not changed any design parameter, do NOT rebuild pad
    # geometry/RBE3 tables and do NOT rerun a modified copy unless no validated
    # baseline .t16 exists. This is the critical fix for matching the Excel
    # baseline results from "Updated automated".
    exact_baseline = _is_exact_baseline_payload(
        area_mult=area_mult,
        ratio_pct=ratio_pct,
        pos_offset=pos_offset,
        span_extra=span_extra,
        main_pos_offset=main_pos_offset,
        main_span=main_span,
        distal_span=distal_span,
        distal_offset=distal_offset,
        cal_factor=cal_factor,
        counter_split=counter_split,
    )

    run_dir = RUN_ROOT / pid
    job_id = f"{pid}_ui"

    if exact_baseline:
        baseline_t16 = _find_existing_baseline_t16(patient_dir, pid)
        if baseline_t16 is None:
            raise FileNotFoundError(
                f"Baseline .t16 not found for {pid} in:\n{patient_dir}\n\n"
                f"Expected a pre-computed baseline result (.t16) in the Updated automated folder.\n"
                f"Check that the baseline model has been run and the .t16 file exists."
            )

        _copy_baseline_side_files(patient_dir, run_dir, pid, job_id)
        initial, final = _cobb_from_t16_subprocess(
            baseline_t16, spine_csv, meta["upper_end"], meta["lower_end"]
        )
        _LAST_RUN_INFO = {
            "source": "baseline cache",
            "patient_dir": str(patient_dir),
            "t16_path": str(baseline_t16),
            "initial_cobb": round(initial, 3),
            "final_cobb": round(final, 3),
            "correction": round(initial - final, 2),
        }
        return round(initial - final, 2)

    # ── Compute new forces ─────────────────────────────────────────────────────
    baseline_main = meta["main_force"]  # Already in mN
    ratio_factor  = 1.0 + ratio_pct / 100.0   # ratio_pct = 200*(force_ratio - 0.5)
    new_main      = baseline_main * area_mult * ratio_factor * cal_factor
    # Total counter = main force; split between proximal (pad3) and distal (pad1)
    new_counter_total = new_main
    new_proximal  = new_counter_total * float(counter_split)
    new_distal    = new_counter_total * (1.0 - float(counter_split))

    # Explicit force override (force-ratio control). The app has already split the
    # fixed total into main + two EQUAL counters (incl. pad-area/pressure scaling),
    # so just apply them. This is the only way to vary the main:counter ratio,
    # since the derivation above ties the counter total to the main force.
    if main_force_mn is not None and counter_force_each_mn is not None:
        new_main     = float(main_force_mn)
        new_proximal = float(counter_force_each_mn)
        new_distal   = float(counter_force_each_mn)

    # ── Compute modified geometry if position or span changes ──────────────────
    new_centroids: dict | None = None
    new_dep_nodes: dict | None = None
    force_signs:   dict | None = None

    # Only rebuild pad geometry when something geometric ACTUALLY changes. distal_offset
    # is checked with "!= 0" (not "is not None") to match pos_offset: the app always sends
    # distal_offset=0 at the baseline distal position, and "0 is not None" would wrongly
    # force a rebuild on every force-ratio/pressure-only run — replacing the validated
    # baseline centroids/RBE3 with an approximate regeneration and corrupting the solve.
    need_rebuild = (pos_offset != 0 or span_extra != 0
                    or main_pos_offset != 0 or main_span is not None
                    or distal_span is not None
                    or (distal_offset is not None and int(distal_offset) != 0))

    if need_rebuild:
        angles = _bone_angles(
            meta["cobb_deg"], meta["convexity"],
            meta["upper_end"], meta["lower_end"],
        )
        nodes = _gen_nodes(angles, meta["bone_step"], meta["disc_step"])

        new_span   = meta["span"] + span_extra
        pads       = _compute_pads(
            nodes, meta["convexity"],
            meta["upper_end"], meta["lower_end"],
            new_span, pos_offset, meta["pad_x_offset"],
            main_pos_offset=main_pos_offset,
            main_span=main_span,
            distal_span=distal_span,
            distal_offset=distal_offset,
        )
        new_centroids = {k: (v["cx"], v["cy"]) for k, v in pads.items()}
        new_dep_nodes = {k: v["dep_nodes"] for k, v in pads.items()}
        # Derive force signs from geometry (same as force_signs_for_convexity)
        force_signs = {}
        for pad_num, info in pads.items():
            # info["force_sign"] is the multiplier: +1 or -1
            force_signs[pad_num] = info["force_sign"]

    # ── Write modified .dat to temp folder ─────────────────────────────────────
    run_dir  = RUN_ROOT / pid
    run_dir.mkdir(parents=True, exist_ok=True)
    job_id   = f"{pid}_ui"
    dat_dst  = run_dir / f"{job_id}.dat"
    t16_path = run_dir / f"{job_id}.t16"

    for ext in (".t16", ".out", ".sts", ".log"):
        old = run_dir / f"{job_id}{ext}"
        if old.exists():
            old.unlink()

    _modify_dat(
        dat_src, dat_dst,
        main_force_mn=new_main,
        proximal_force_mn=new_proximal,
        distal_force_mn=new_distal,
        krz=K_RZ,
        new_centroids=new_centroids,
        new_dep_nodes=new_dep_nodes,
        force_signs=force_signs,
    )

    # ── Run Marc ───────────────────────────────────────────────────────────────
    if not MARC_BAT.exists():
        raise FileNotFoundError(f"run_marc.bat not found:\n{MARC_BAT}")

    cmd = f'"{MARC_BAT}" -jid "{job_id}" -back yes -nts {NTS} -nte {NTE}'
    result = subprocess.run(cmd, shell=True, cwd=str(run_dir), capture_output=True, text=True)

    if not t16_path.exists():
        out_file = run_dir / f"{job_id}.out"
        hint = f"\nCheck Marc log: {out_file}" if out_file.exists() else ""
        raise RuntimeError(
            f"Marc did not produce {t16_path.name} – job may have failed.{hint}\n"
            f"stdout:\n{result.stdout[-2000:]}\n\nstderr:\n{result.stderr[-2000:]}"
        )

    # ── Measure Cobb correction (Marc Python 3.11 subprocess) ──────────────────
    initial, final = _cobb_from_t16_subprocess(
        t16_path, spine_csv, meta["upper_end"], meta["lower_end"]
    )
    correction = round(initial - final, 2)
    _LAST_RUN_INFO = {
        "source": "modified Marc run",
        "patient_dir": str(patient_dir),
        "metadata": meta.get("_metadata_path", ""),
        "spine_csv": str(spine_csv),
        "t16_path": str(t16_path),
        "initial_cobb": round(initial, 3),
        "final_cobb": round(final, 3),
        "correction": correction,
    }
    return correction   # positive = brace corrects the curve


def get_last_run_info() -> dict:
    """Return cached FE run metadata for the UI to display."""
    return _LAST_RUN_INFO
