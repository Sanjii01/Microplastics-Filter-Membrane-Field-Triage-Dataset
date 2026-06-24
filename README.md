# Microplastics Filter Membrane Field Triage

Complete Eris-style challenge package for a scientific microscope-image triage task.

- `raw/generate_raw.py` creates deterministic filter membrane field images and `raw/data.csv`.
- `dataset_description_eris_upload.md` is the dataset description for Eris.
- `prepare.py` creates public/private scene-level splits and copies images to opaque public filenames.
- `problem.md` is the solver-facing challenge statement.
- `grade.py` validates submissions and computes multiple-choice accuracy.
- `rubrics.yaml` contains task-specific rubric criteria.
- `solution.ipynb` and `reference_solution.py` provide a lightweight numpy/PIL solvability baseline.

## Submission Mapping

Dataset upload:
- Title: `Microplastics Filter Membrane Field Triage Dataset`
- Description: paste `dataset_description_eris_upload.md`
- Data files: upload a zip containing top-level `data.csv`, `images/`, and `generate_raw.py`
- License: `CC0 1.0 Public Domain`

Challenge:
- Domain: `Computer Vision`
- Difficulty: `Medium`
- GPU: `A10G`
- Title: `Microplastics Filter Membrane Field Triage`
- Grade direction: `Maximize`
- Min score: `0`
- Max score: `1`
- Tags: `image`, `multimodal`, `feature-engineering`, `small-data`
- Problem description: paste `problem.md`
- Grading script: paste `grade.py`
- Prepare script: paste `prepare.py`
- Rubrics: use `rubrics.yaml`
- Reference solution: upload `solution.ipynb`

Reviewer-facing notes:
- This is microscope field-of-view triage, not ordinary image classification.
- Train/test splitting is scene-level to avoid repeated-image leakage.
- The revised 224x224 renderer uses larger, more separated cues so the task is visually solvable.
- The sample submission scores above zero and the lightweight reference baseline scores above random guessing.
- The private split includes harder visibility and OOD-style shifts.
- The raw upload includes `generate_raw.py`, making the synthetic source auditable.
