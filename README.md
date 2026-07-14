# Olist E-Commerce ML — MSc Dissertation

> Delivery Logistics, Geographic Distance, and Customer Satisfaction in Brazilian E-Commerce: A Data-Driven Analysis of the Olist Marketplace.
>
> **MSc Business Analytics (BEMM466), University of Exeter · Supervisor: Dr Lina Zhang · 2025**

---

## 🎯 Problem

Does the **geographic distance between sellers and customers** — combined with regional infrastructure gaps in Brazil — systematically drive customer satisfaction on the Olist marketplace, and do seller-level differences moderate that effect?

## 📦 Dataset

- **Source:** Kaggle — Olist Brazilian E-Commerce Public Dataset (CC-BY licence)
- **Scope:** 96,353 *delivered* orders across all 27 Brazilian states
- **Window:** September 2016 – August 2018
- **Merged tables:** orders, order_items, customers, sellers, products, geolocation, reviews

## 🧪 Methodology (three-phase design)

| Phase | Question | Method |
|---|---|---|
| **RQ1** Geographic analysis | Do same-state vs cross-state deliveries differ in satisfaction? | Corridor-level distance & satisfaction stats |
| **RQ2** Satisfaction prediction | Can we predict review satisfaction from delivery + product features? | XGBoost (+ Logistic baseline), SMOTE on train only |
| **RQ3** Seller benchmarking | Do sellers cluster into performance archetypes? | KMeans segmentation + strategic profiling |

**Methodological contribution:** a strict **80/20 temporal split** (train ≤ cut-off, test after) so seller-level features are computed **only from training-period data** — eliminating the temporal leakage present in most published Olist ML studies.

**Class imbalance:** 78.9% of orders are satisfied (4–5 star) → SMOTE applied **only** to the training fold.

## 📈 Key Findings

- Cross-state deliveries carry materially lower satisfaction than same-state
- Delivery-time features (actual vs estimated) dominate feature importance
- Sellers cluster into distinct performance archetypes with actionable strategic profiles (see report §5.5)
- Leakage-resistant XGBoost gives an honest performance benchmark that generalises beyond the training window

*Full metrics, ROC curves and segment profiles are in the [dissertation PDF](docs/Final_Dissertation_Report_and_summary.pdf).*

## 🧰 Tech Stack

`Python 3.11` · `pandas` · `scikit-learn` · `xgboost` · `imbalanced-learn` (SMOTE) · `geopy` · `matplotlib` · `seaborn`

## 📁 Repo Structure

```text
.
├── notebooks/          # EDA, geographic analysis, modelling, segmentation
├── src/                # reusable feature-engineering + modelling utilities
├── data/               # (gitignored) place Kaggle Olist dump here
├── docs/               # dissertation report + figures
├── requirements.txt
└── README.md
```

## ▶️ Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Place Olist CSVs in ./data/raw/
jupyter lab notebooks/
```

## 📄 Report

Full dissertation → [`docs/Final_Dissertation_Report_and_summary.pdf`](docs/Final_Dissertation_Report_and_summary.pdf)

---

<sub>MIT-licensed · Author: [Parth Badiger](https://github.com/parthbadiger24-creator)</sub>
