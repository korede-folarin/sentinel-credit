# References

Publications and datasets that informed engineering decisions throughout Sentinel Credit.

---

## Regulatory Publications

| Publication | Engineering Decision Informed |
|---|---|
| FCA Credit Rating Agencies Multi-Firm Review (May 2026) | Closed-loop governance architecture |
| Belton et al., FCA AI Explainability Research Note (February 2025) | Adaptive explanation engine |
| FCA / ICO / Alan Turing Institute Synthetic Data Framework (June 2023) | Synthetic data validation framework covering fidelity, utility, and privacy |
| PRA SS1/23 (April 2023) | Governance layer and model risk management principles |
| FCA Consumer Duty (PS22/9, July 2023) | Consumer-facing explanation design and vulnerability considerations |
| FCA CONC 5.2A | Affordability assessment and creditworthiness feature design |
| FCA Mills Review (July 2026) | AI governance expectations in retail financial services |
| Woolard Review (February 2021) | BNPL consumer risk characteristics and vulnerability indicators |
| Federal Reserve SR 11-7 (2011) | Independent model review and effective challenge principles |

---

## Academic and Industry Research

| Publication | Engineering Decision Informed |
|---|---|
| Kurian & Allali (2024) | Population Stability Index (PSI) thresholds and drift monitoring |
| Demajo et al. (2020) | Credit scoring model benchmarking and explainability |
| Khalili et al. (2026) | BNPL thin-file customer characteristics |
| Kurshan et al. (2020), JPMorgan AI Research | Human-in-the-loop AI governance |
| Brahmandam (2025) | MLOps automation architecture |
| Kiran & Saravana Kumar (2024) | Synthetic data generation and validation methodology |
| Immaneni (2022) | Database-level feature engineering to reduce training-serving skew |

---

## Calibration Sources

The synthetic data generator was calibrated using publicly available UK demographic, labour market, financial, and BNPL statistics.

| Source | Calibration Use |
|---|---|
| Klarna Wikipink UK 2024 | Age distribution and published UK default rate |
| Klarna Q2 2025 Investor Report | Merchant risk multiplier calibration and delinquency ranges |
| Klarna UK Store Directory | Merchant category distribution |
| Office for National Statistics – Population Estimates (2023) | Gender distribution |
| Office for National Statistics – Labour Market Overview (2024) | Employment status distribution |
| Office for National Statistics – Household Income Inequality (FYE 2023) | Income band calibration |
| FCA Financial Lives Survey (2022) | Financial vulnerability indicators |
| PwC & TotallyMoney YouGov Survey (2022) | Thin-file prevalence estimates |
| NHS Digital Adult Psychiatric Morbidity Survey | Consumer vulnerability flag calibration |
| World Bank Alternative Data Publications (2025) | Alternative data feature selection |
| LendingTree BNPL Reports (2025) | BNPL late payment behaviour context |

---

## Citation Notes

These publications are referenced as research inputs that informed engineering decisions throughout Sentinel Credit.

Their inclusion should not be interpreted as endorsement, approval, certification, or affiliation with the project.

Detailed mappings between individual publications and implementation decisions are documented in the [Engineering Decision Log](engineering_decision_log.md).