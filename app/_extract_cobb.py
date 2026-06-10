"""
_extract_cobb.py: py_post Cobb-angle extraction helper.

Run with Marc's bundled Python 3.11, not the Streamlit Python, to avoid native
py_post DLL/Python-version conflicts.

Usage:
    python _extract_cobb.py <t16_path> <spine_csv_path> <upper_vert> <lower_vert>

Prints:
    initial_cobb_deg final_cobb_deg
"""

import csv
import math
import os
import sys

MARC_PYTHON_PATHS = [
    r"C:\Users\<username>\OneDrive - Imperial College London\Year 4\FYP\Patient_1_final\shlib\win64",
    r"C:\Program Files\MSC.Software\Marc\2024.1.0\mentat2024.1\shlib\win64",
    r"C:\Program Files\MSC.Software\Marc\2024.1.0\mentat2024.1\python\WIN8664",
    r"C:\Program Files\MSC.Software\Marc\2024.1.0\mentat2024.1\python\WIN8664\Lib",
]

SPINE_LEVELS_BOTTOM_TO_TOP = [
    "L5", "L4", "L3", "L2", "L1",
    "T12", "T11", "T10", "T9", "T8", "T7", "T6",
    "T5", "T4", "T3", "T2", "T1",
    "C7", "C6", "C5", "C4", "C3", "C2", "C1",
]


def _setup() -> None:
    for folder in MARC_PYTHON_PATHS:
        if os.path.exists(folder) and folder not in sys.path:
            sys.path.insert(0, folder)
    if hasattr(os, "add_dll_directory"):
        for folder in MARC_PYTHON_PATHS:
            if os.path.exists(folder):
                try:
                    os.add_dll_directory(folder)
                except OSError:
                    pass


def _build_vmap(spine_csv: str) -> dict:
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


def _measure(t16_path: str, vmap: dict, top_v: str, bot_v: str) -> tuple[float, float]:
    from py_post import post_open  # type: ignore

    top_v = top_v.upper().strip()
    bot_v = bot_v.upper().strip()
    missing = [v for v in (top_v, bot_v) if v not in vmap]
    if missing:
        raise RuntimeError(f"End vertebra(e) missing from spine map: {missing}. Available: {sorted(vmap)}")

    p = post_open(t16_path)
    ninc = p.increments()
    if ninc < 2:
        p.close()
        raise RuntimeError(f"Only {ninc} increment(s) – solve may have failed.")

    def perp(inc: int, label: str):
        info = vmap[label]
        p.moveto(inc)

        def pos(nid):
            idx = p.node_sequence(nid)
            nd = p.node(idx)
            try:
                dx, dy, _ = p.node_displacement(idx)
            except Exception:
                dx, dy = 0.0, 0.0
            return nd.x + dx, nd.y + dy

        b = pos(info["bot"])
        t = pos(info["top"])
        ax = t[0] - b[0], t[1] - b[1]
        return -ax[1], ax[0]

    def angle(u, v) -> float:
        nu = math.hypot(*u)
        nv = math.hypot(*v)
        if nu == 0 or nv == 0:
            return 0.0
        dot = abs(u[0] * v[0] + u[1] * v[1])
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (nu * nv)))))

    initial = angle(perp(0, top_v), perp(0, bot_v))
    final = angle(perp(ninc - 1, top_v), perp(ninc - 1, bot_v))
    p.close()
    return initial, final


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: _extract_cobb.py <t16> <spine_csv> <upper_vert> <lower_vert>")

    t16_path, spine_csv, upper_v, lower_v = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    _setup()

    try:
        vmap = _build_vmap(spine_csv)
        initial, final = _measure(t16_path, vmap, upper_v, lower_v)
        print(f"{initial} {final}")
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)
