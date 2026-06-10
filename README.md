# AIS brace correction predictor (clinical interface)

A Streamlit decision-support interface that sits on top of the 2D finite element
scoliosis pipeline. An orthotist picks a patient, adjusts brace parameters (pad
positions, spans, force balance and pressure) with sliders, and the tool runs
the underlying Marc FE model and reports the predicted in-brace Cobb angle
correction against the clinical result.

This is a research proof of concept and is **not** a validated medical device or
intended for clinical use.

## Important: the data here is synthetic

The CSV files in `data/` are **dummy data**, not real patient records. They show
the column format the app expects and contain three made-up patients. See
[DISCLAIMER.md](DISCLAIMER.md) for the full note, including the extra files the
app needs in order to actually run a prediction.

## Layout

```
app/    the application code (app.py, fe_runner.py, patient_data.py, _extract_cobb.py)
data/   synthetic example CSVs (no real patient data)
```

## What it needs to run

- Python with `streamlit`, `pandas`, `numpy` and `matplotlib` (see `requirements.txt`)
- A local MSC Marc Mentat 2024.1 installation (the solver and its `py_post` library)
- The per-patient FE model files produced by the main pipeline

The interface will still open without Marc and the model files, but it cannot
run a prediction; it can only display the patient list and the layout.

### About py_post

`py_post.pyd` is the compiled post-processing library that reads Marc's `.t16`
result files. It ships as part of the MSC Marc installation and is proprietary
MSC software, so it **cannot be redistributed and is not included in this
repository for licensing reasons**. It is, however, required to run the
interface (the app uses it to read the solved Cobb angle out of the `.t16`). To
run the tool you must have your own licensed Marc install and point the paths in
`fe_runner.py` and `_extract_cobb.py` at its `py_post` location.

## Files required but not included in this repository

Several things the app reads are either proprietary, machine-specific, or
contain real patient data, so they are not committed here. To run the tool
end to end you need to supply:

**From your Marc installation (proprietary, see "About py_post"):**
- `py_post.pyd` (Marc post-processing library)
- Marc's bundled Python 3.11 (`...\mentat2024.1\python\WIN8664\python.exe`)
- `run_marc.bat` (the Marc solver launcher, `...\marc2024.1\tools\run_marc.bat`)

**Per-patient FE model files**, one folder per patient at
`Automated models/Updated automated/{PID}/`, each containing:
- `{PID}.dat` (the baseline Marc input deck)
- `{PID}.t16` (the solved baseline result)
- `{PID}_model_metadata.csv` (curve region, convexity, end vertebrae)
- `{PID}_spine.csv` (node coordinates)

**Per-patient sensitivity summaries** (optional, enables the parameter-sensitivity
panel; without them the app falls back to a cohort view):
- `Automated models/Sensitivities 2 multi-patient/{PID}/sensitivity_summary.csv`

**Clinical data** (the real versions are not published; synthetic samples are in
`data/`):
- `clinical_patient_data.csv` and `Curve_convexity_directions.csv` (read from `BASE_DIR`)
- `Patient specific data.csv` (read from the same folder as `patient_data.py`)

A temporary run folder (`C:\Temp\marc_ui_runs\`) is created automatically at
runtime and does not need to be supplied.

## File paths

The scripts use absolute Windows paths from the machine they were built on,
shown with a `<username>` placeholder, for example in `app.py`:

```python
BASE_DIR = Path(r"C:\Users\<username>\OneDrive - Imperial College London\Year 4\FYP\Patient_1_final")
```

Before running, edit `BASE_DIR` and the related paths in `app.py`, `fe_runner.py`
and `_extract_cobb.py` to match your own setup and Marc install. The app reads
the clinical CSVs from `BASE_DIR`, while `Patient specific data.csv` is read from
the same folder as `patient_data.py`.

## Running

```
streamlit run app/app.py
```

## Related

The finite element pipeline that generates the model files this interface
consumes is in a separate repository: the scoliosis 2D FE pipeline.
