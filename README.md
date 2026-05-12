# CLEAR-HEAD — Analysis Code

Reproducibility package for:

> **LLM-generated lay summaries improve patient satisfaction but not objective
> comprehension of radiology reports: a randomized controlled trial**
> Le Guellec B, Bentegeac R, Tran VT, El Homsi M, Amouyel P, Hacein-Bey L, Pruvo JP,
> Kuchcinski G\*, Hamroun A\* 

---

## Repository structure

```
upload_code/
├── analysis.py          # Primary & secondary GEE analyses → clearhead_long.csv
├── figure2_primary.py   # Figure 2: primary outcome bar chart
├── figure3_likert.py    # Figure 3: secondary outcome diverging bar charts
├── figure4_forest.py    # Figure 4: subgroup forest plot
└── README.md            # This file
```

The scripts assume the following layout relative to this directory:

```
project_root/
├── data/                # Raw questionnaire CSVs (not distributed — see below)
│   ├── QST_0661_20260426.csv
│   ├── qst_066[2-8]_20260426.csv
│   └── donnees_demographique_20260426.csv
├── graphs/              # Created automatically on first run
└── scripts/             # ← this directory
    ├── analysis.py
    ├── generate_summaries.py
    ├── figure2_primary.py
    ├── figure3_likert.py
    └── figure4_forest.py
```

---

## Data availability

Raw participant data are held by the **ComPaRe** e-cohort (AP-HP, Paris, France)
and cannot be publicly shared under the ComPaRe data governance framework.
De-identified data are available upon reasonable request to the corresponding
author, subject to a data sharing agreement with ComPaRe.

The six study vignettes (original reports + LLM-generated summaries) are
provided in `selected_reports_with_summaries.xlsx`.

---

## Reproducing the analyses

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scipy statsmodels openpyxl
```

Python ≥ 3.10 is recommended. All analyses were run with Python 3.12,
statsmodels 0.14, matplotlib 3.9, and scipy 1.13.

### 2. Run the statistical analysis

```bash
cd scripts/
python analysis.py
```

This reads the raw questionnaire CSVs from `../data/`, reshapes them into
long format, fits all GEE models, and saves `clearhead_long.csv`.

### 3. Regenerate figures

```bash
python figure2_primary.py   # Figure 2
python figure3_likert.py    # Figure 3
python figure4_forest.py    # Figure 4 (uses a cache — see note below)
```

PNG (300 dpi) and PDF (vector) outputs are written to `../graphs/`.

> **Note on the forest-plot cache** — `figure4_forest.py` saves GEE results
> to `figure4_forest_cache.pkl` on first run (~10 min). Subsequent runs load
> the cache instantly. Delete the `.pkl` file to force recomputation.

---

## Statistical methods

All inferential tests use **Generalized Estimating Equations (GEE)** with an
exchangeable within-participant correlation structure to account for the
repeated-measures design (six reports per participant). Report identity is
included as a fixed covariate to control for report-level difficulty.

| Outcome type       | GEE family | Link   |
|--------------------|-----------|--------|
| Binary (correct / yes / no) | Binomial  | Logit  |
| Likert (1–5, continuous)    | Gaussian  | Identity |

Effect sizes are reported as **Odds Ratios (OR)** with 95% Wald confidence
intervals for binary outcomes.

Subgroup interaction p-values are derived from a Wald chi-squared test on the
arm × subgroup interaction terms.

---

## LLM prompt

The exact prompt used to generate patient summaries with Mistral Small 3.2 is
provided in `generate_summaries.py`.  Summaries were generated in a zero-shot
setting with `temperature = 0` for deterministic output.

---

## License

Code is released under the **MIT License** (see `LICENSE`).
The study vignettes are released under **CC BY 4.0**.
