# Refill Irregularity & Adherence Risk Detection

> Predicting which patient-drug pairs are likely to refill late, using CMS DE-SynPUF Prescription Drug Events.

**Pharmacy2U Hackathon** — Hybrid approach: XGBoost (binary risk score) + Survival Analysis (time-to-refill with censoring).

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Install dependencies

```bash
git clone https://github.com/nilavans/pharmacy2u-hackathon-refill.git
cd pharmacy2u-hackathon-refill
pip install -r requirements.txt
```

### 2. Data setup

Download the CMS DE-SynPUF Prescription Drug Events (Sample 1):

1. Visit the [CMS DE-SynPUF Download Page](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1)
2. Download **"DE1.0 Sample 1 2008-2010 Prescription Drug Events"** (ZIP)
3. Unzip and place the CSV in the `data/` folder:

```
data/prescription_drug_event.csv
```

> The file is ~700MB with 5,552,421 rows and 8 columns (2008–2010 Medicare Part D events).

### 3. Run

```bash
jupyter notebook refill_irregularity_v2.ipynb
```

Select **Cell → Run All**. Full execution takes ~5–10 minutes.

### 4. Outputs

The notebook produces inline:
- **8 plots** — PR curve, calibration, SHAP beeswarm, SHAP waterfall, 2 Kaplan-Meier, Cox forest plot, patient timelines
- **Risk score table** — ranked patient-drug pairs with Low / Medium / High tiers
- **Model comparison** — XGBoost PR-AUC, ROC-AUC, Cox C-index

---

## Approach

### Problem

Predict which patient-drug pairs are likely to refill late next time, and produce a usable risk score.

### Label Definition

Per the hackathon spec:

```
expected_runout = SRVC_DT + DAYS_SUPLY_NUM
gap = next_fill_date − expected_runout
late = 1  if  gap > 7 days  (grace window)
```

### Grouping Strategy

NDC-11 (`PROD_SRVC_ID`) produces 99.9% singleton groups in this synthetic data. After testing every prefix length (NDC-5 through NDC-11), **NDC-5** (labeler prefix) gives 247K patient-drug groups with 3+ fills — the only workable level.

| Prefix | Unique | Groups ≥ 3 fills | Groups ≥ 5 fills |
|--------|--------|-------------------|-------------------|
| NDC-11 | 503,683 | 7 | 0 |
| NDC-9 | 117,029 | 1,057 | 4 |
| NDC-7 | 14,761 | 65,314 | 8,174 |
| **NDC-5** | **3,741** | **247,167** | **77,445** |

> With real Pharmacy2U data using dm+d/BNF codes, this workaround is unnecessary.

### Features — 21 features across 7 categories

| Category | Features | Source |
|----------|----------|--------|
| Gap statistics | Rolling mean, std, CV, trend, max | `SRVC_DT`, `DAYS_SUPLY_NUM` |
| Stockpiling | Early refill flag, days banked, cumulative stockpile | `SRVC_DT`, `DAYS_SUPLY_NUM` |
| Cadence stability | Regularity score, historical late % | Derived |
| Cost patterns | Previous pay, cost, burden ratio, pay change | `PTNT_PAY_AMT`, `TOT_RX_CST_AMT` |
| Quantity patterns | Qty per day, qty changed flag | `QTY_DSPNSD_NUM` |
| Polypharmacy | Unique drugs, 90-day fill count | `PROD_SRVC_ID` |
| Context | Refills so far, tenure, days supply | All |

All features are computed at **t−1** (lagged) to prevent information leakage.

### Models

| Model | Target | Handles Censoring | Primary Metric |
|-------|--------|-------------------|----------------|
| **XGBoost** | Binary: late vs on-time | No (drops censored rows) | PR-AUC |
| **Cox PH** | Time-to-next-refill (days) | Yes (last fill per pair) | C-index |

### Validation

- **Temporal split** — Train (< Jul 2009) → Val (Jul 2009 – Jan 2010) → Test (≥ Jan 2010)
- **No random splits** — prevents temporal leakage
- **Censoring** — 247K last-fill intervals: dropped by XGBoost, used by the survival model

---

## Results

| Metric | Value | Baseline |
|--------|-------|----------|
| PR-AUC | **0.713** | 0.560 |
| ROC-AUC | **0.688** | 0.500 |
| Cox C-index | **0.633** | 0.500 |

### Key Finding: Synthetic Data Limitation

The CMS DE-SynPUF deliberately destroys co-variation between variables for privacy. Temporal autocorrelation is absent (lift = 1.00x — being late once does NOT predict future lateness). The pipeline is designed to capture temporal persistence signals that would exist in real dispensing data.

---

## Demo Outputs

### Risk Score Table
Top patient-drug pairs ranked by risk score with tiers (Low / Medium / High), historical late rate, and expected run-out date.

### Patient Timeline
Visual showing coverage bars (green), grace window (amber), late periods (red), and risk score trend over time.

### SHAP Analysis
Feature importance beeswarm + single-patient waterfall explaining individual risk scores.

---

## Repo Structure

```
pharmacy2u-hackathon-refill/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── refill_irregularity.ipynb          # Main analysis notebook
└── data/
    └── prescription_drug_event.csv    # CMS PDE data (not in repo — see setup)
```
---

## Tech Stack

Python · pandas · NumPy · XGBoost · scikit-learn · SHAP · lifelines · matplotlib
