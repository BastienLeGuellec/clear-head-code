"""
analysis.py — Statistical analysis for the CLEAR-HEAD trial
============================================================
Randomized vignette-based trial of LLM-generated lay summaries for brain
MRI reports in headache patients.

Primary outcome : correct classification of the report as containing
                  an explanation for the patient's symptoms (Q1_correct).
Secondary outcomes : satisfaction (Q2), willingness to consult (Q3),
                     subjective comprehension (Q4), anxiety (Q5),
                     and normal/abnormal classification (anomalie_correct).

Statistical method : Generalized Estimating Equations (GEE) with an
                     exchangeable within-participant correlation structure
                     to account for the repeated-measures design (6 reports
                     per participant). Binomial family for binary outcomes,
                     Gaussian family for Likert-scale outcomes.

Outputs : clearhead_long.csv   — reshaped long-format dataset used by all
                                  figure scripts.

Usage:
    # Run from the project root (one level above scripts/)
    python scripts/analysis.py

Dependencies: pandas, numpy, statsmodels, scipy
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

warnings.filterwarnings('ignore')

# ── 0. Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')

# ── 1. Load raw questionnaire data ───────────────────────────────────────────
dfs = []
for f in sorted(
    glob.glob(os.path.join(DATA_DIR, 'QST_0*.csv')) +
    glob.glob(os.path.join(DATA_DIR, 'qst_0*.csv'))
):
    dfs.append(pd.read_csv(f, encoding='latin-1'))

df_all = pd.concat(dfs, ignore_index=True)

# Keep only participants who provided informed consent
df_p = df_all[df_all['participation_etude'] == 'Oui'].copy()
print(f'Participants who consented: {df_p["id_unik"].nunique()}')

# ── 2. Arm & report-order mapping ────────────────────────────────────────────
# Each VQST code corresponds to one questionnaire version.
# Paired versions (e.g., VQST_1452 / VQST_1451) share the same report order;
# one received the original report only (control) and the other received the
# report with the LLM-generated lay summary appended (intervention).

arm_map = {
    'VQST_1452': 'control',       # QST_0661 — original report only
    'VQST_1451': 'intervention',  # QST_0662 — report + LLM summary
    'VQST_1454': 'control',       # QST_0663
    'VQST_1453': 'intervention',  # QST_0664
    'VQST_1456': 'control',       # QST_0665
    'VQST_1455': 'intervention',  # QST_0666
    'VQST_1458': 'control',       # QST_0667
    'VQST_1457': 'intervention',  # QST_0668
}

# Order in which the 6 reports (R1–R6) appeared for each questionnaire version.
# Report IDs are de-identified institutional accession numbers.
report_order = {
    'VQST_1452': ['87142013', '91182589', '88811727', '88096936', '87381313', '87236611'],
    'VQST_1451': ['87142013', '91182589', '88811727', '88096936', '87381313', '87236611'],
    'VQST_1454': ['88096936', '87236611', '91182589', '87381313', '87142013', '88811727'],
    'VQST_1453': ['88096936', '87236611', '91182589', '87381313', '87142013', '88811727'],
    'VQST_1456': ['88811727', '87142013', '87381313', '91182589', '87236611', '88096936'],
    'VQST_1455': ['88811727', '87142013', '87381313', '91182589', '87236611', '88096936'],
    'VQST_1458': ['87236611', '87381313', '88096936', '88811727', '91182589', '87142013'],
    'VQST_1457': ['87236611', '87381313', '88096936', '88811727', '91182589', '87142013'],
}

# Ground truth per report (established by consensus of four neuroradiologists):
#   normal = 1 → no clinically relevant finding
#   causal = 1 → finding explaining the patient's headache symptoms
#   normal = 0, causal = 0 → incidental finding unrelated to symptoms
gt = {
    '87142013': {'normal': 1, 'causal': 0},
    '91182589': {'normal': 0, 'causal': 1},
    '88811727': {'normal': 0, 'causal': 0},
    '88096936': {'normal': 0, 'causal': 1},
    '87381313': {'normal': 0, 'causal': 0},
    '87236611': {'normal': 1, 'causal': 0},
}

df_p['arm'] = df_p['IdVersion'].map(arm_map)

# ── 3. Reshape wide → long ───────────────────────────────────────────────────
# Each participant answered questions for 6 reports (R1–R6).
# We produce one row per participant-report pair.
rows = []
for _, row in df_p.iterrows():
    vqst = row['IdVersion']
    if vqst not in report_order:
        continue
    for pos, report_id in enumerate(report_order[vqst], start=1):
        r = f'R{pos}'
        rows.append({
            'id_unik':   row['id_unik'],
            'vqst':      vqst,
            'arm':       arm_map[vqst],
            'report_id': report_id,
            'position':  pos,
            # Q1: "Does this report contain a finding that explains your symptoms?"
            'Q1':        row.get(f'Q1_{r}'),
            # Q2: Overall satisfaction (Likert 1–5)
            'Q2':        row.get(f'Q2_{r}'),
            # Q3: Willingness to contact a professional (Yes/No)
            'Q3':        row.get(f'Q3_{r}'),
            # Q4: Subjective comprehension (Likert 1–5)
            'Q4':        row.get(f'Q4_{r}'),
            # Q5: Self-reported anxiety (Likert 1–5)
            'Q5':        row.get(f'Q5_{r}'),
            # Anomalie: "Does this report contain any abnormal finding?" (Yes/No)
            'anomalie':  row.get(f'{r}_anomalie'),
            'normal':    gt[report_id]['normal'],
            'causal':    gt[report_id]['causal'],
        })

df_long = pd.DataFrame(rows)
print(f'Long-format dataset: {df_long.shape[0]} observations, '
      f'{df_long["id_unik"].nunique()} participants')

# ── 4. Derive outcome variables ───────────────────────────────────────────────

# Primary outcome – Q1_correct:
#   Causal reports  → "Oui" (Yes) is the correct answer
#   Normal/Incidental reports → "Non" (No) is the correct answer
df_long['Q1_bin'] = (df_long['Q1'] == 'Oui').astype(float)
df_long['Q1_correct'] = np.where(
    df_long['causal'] == 1,
    (df_long['Q1'] == 'Oui').astype(float),
    (df_long['Q1'] == 'Non').astype(float)
)
df_long['Q1_correct'] = df_long['Q1_correct'].where(df_long['Q1'].notna())

# Normal/abnormal classification – anomalie_correct:
#   Normal reports           → "Non" is correct
#   Causal + Incidental reports → "Oui" is correct (both have an anomaly)
df_long['anomalie_bin'] = (df_long['anomalie'] == 'Oui').astype(float)
df_long['anomalie_correct'] = np.where(
    df_long['normal'] == 1,
    (df_long['anomalie'] == 'Non').astype(float),
    (df_long['anomalie'] == 'Oui').astype(float)
)
df_long['anomalie_correct'] = df_long['anomalie_correct'].where(
    df_long['anomalie'].notna()
)

# Composite objective comprehension score (mean of Q1_correct & anomalie_correct)
df_long['composite'] = df_long[['Q1_correct', 'anomalie_correct']].mean(axis=1)

# Likert-scale items: extract the digit from the response string
def parse_likert(s):
    """Return the first digit found in string s, or NaN."""
    if pd.isna(s):
        return np.nan
    for c in str(s):
        if c.isdigit():
            return int(c)
    return np.nan

for q in ['Q2', 'Q4', 'Q5']:
    df_long[f'{q}_num'] = df_long[q].apply(parse_likert)

df_long['Q3_bin'] = (df_long['Q3'] == 'Oui').astype(float)

# Arm as binary indicator (intervention = 1, control = 0)
df_long['arm_bin'] = (df_long['arm'] == 'intervention').astype(float)

# ── 5. Descriptive statistics ─────────────────────────────────────────────────
print('\n=== Descriptive statistics by arm ===')
for outcome in ['composite', 'Q1_correct', 'anomalie_correct',
                'Q2_num', 'Q3_bin', 'Q4_num', 'Q5_num']:
    grp = df_long.groupby('arm')[outcome].agg(['mean', 'std', 'count'])
    print(f'\n{outcome}:\n{grp}')

# ── 6. GEE models ─────────────────────────────────────────────────────────────
# All models use an exchangeable within-participant correlation structure to
# account for the repeated-measures design (6 reports per participant).
# Report identity (C(report_id)) is included as a fixed covariate to control
# for report-level difficulty.

def gee_summary(result, label):
    """Print a concise summary of a GEE result focusing on the arm effect."""
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(result.summary().tables[1])
    params   = result.params
    pvalues  = result.pvalues
    ci       = result.conf_int()
    arm_idx  = [i for i, n in enumerate(params.index) if 'arm_bin' in n]
    if arm_idx:
        i = arm_idx[0]
        print(f'\n  → arm_bin: coef={params.iloc[i]:.4f}, '
              f'95% CI [{ci.iloc[i,0]:.4f}, {ci.iloc[i,1]:.4f}], '
              f'p={pvalues.iloc[i]:.4f}')

# 6a. Pre-registered primary outcome
print('\n=== Primary outcome: Q1_correct (GEE, logistic) ===')
df_m = df_long.dropna(subset=['Q1_correct', 'arm_bin']).sort_values('id_unik').reset_index(drop=True)
print(f'Model dataset: {df_m.shape[0]} observations, '
      f'{df_m["id_unik"].nunique()} participants')
res_q1 = GEE.from_formula(
    'Q1_correct ~ arm_bin + C(report_id)',
    groups='id_unik', data=df_m,
    family=Binomial(), cov_struct=Exchangeable()
).fit()
gee_summary(res_q1, 'Q1_correct — causal classification (primary, pre-registered)')

# 6b. Secondary: normal/abnormal classification
print('\n=== Secondary: anomalie_correct (GEE, logistic) ===')
df_anom = df_long.dropna(subset=['anomalie_correct']).sort_values('id_unik').reset_index(drop=True)
res_anom = GEE.from_formula(
    'anomalie_correct ~ arm_bin + C(report_id)',
    groups='id_unik', data=df_anom,
    family=Binomial(), cov_struct=Exchangeable()
).fit()
gee_summary(res_anom, 'anomalie_correct — normal/abnormal classification')

# 6c. Secondary: composite objective comprehension
print('\n=== Secondary: composite (GEE, Gaussian) ===')
df_comp = df_long.dropna(subset=['composite']).sort_values('id_unik').reset_index(drop=True)
res_comp = GEE.from_formula(
    'composite ~ arm_bin + C(report_id)',
    groups='id_unik', data=df_comp,
    family=Gaussian(), cov_struct=Exchangeable()
).fit()
gee_summary(res_comp, 'composite — mean objective comprehension')

# 6d. Secondary patient-reported outcomes
outcomes_secondary = [
    ('Q2_num', Gaussian(), 'Satisfaction (Likert 1–5)'),
    ('Q3_bin', Binomial(), 'Willingness to consult a professional (binary)'),
    ('Q4_num', Gaussian(), 'Subjective comprehension (Likert 1–5)'),
    ('Q5_num', Gaussian(), 'Anxiety (Likert 1–5)'),
]

for outcome, family, label in outcomes_secondary:
    df_s = df_long.dropna(subset=[outcome]).sort_values('id_unik').reset_index(drop=True)
    res_s = GEE.from_formula(
        f'{outcome} ~ arm_bin + C(report_id)',
        groups='id_unik', data=df_s,
        family=family, cov_struct=Exchangeable()
    ).fit()
    gee_summary(res_s, label)

# ── 7. Save long-format dataset ───────────────────────────────────────────────
out_path = os.path.join(SCRIPT_DIR, 'clearhead_long.csv')
df_long.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
