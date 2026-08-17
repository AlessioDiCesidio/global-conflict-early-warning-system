# Global Conflict Index (GCI)

A reproducible statistical research project on organized violence, escalation, conflict termination and cross-border spillover.

## Research questions
1. How has organized violence changed since 1989?
2. What predicts escalation in the next 6 months?
3. What predicts conflict termination?
4. Can a transparent momentum indicator describe relative change without claiming to measure "victory"?
5. How does violence spill across borders?

## Main design
- Main unit: country-month (1989–2026)
- Secondary unit: conflict-year / conflict-episode
- Historical event data: UCDP GED v26.1 (1989–2025)
- Near-real-time extension: UCDP Candidate Events monthly releases (2026)
- Political covariates: V-Dem v16
- Optional socioeconomic covariates: World Bank WDI

## Models
- Descriptive time series and change-point analysis
- Logistic regression / fixed effects for escalation
- Random forest / gradient boosting as nonlinear benchmarks
- Cox proportional hazards for termination
- Spatial/network exposure models for spillover
- Sensitivity analysis for all index thresholds and weights

## Important rule
Do not call the momentum score a "who is winning" score. It is an analytical indicator of relative change in measurable conflict characteristics.

## Sources
UCDP: https://ucdp.uu.se/downloads/
UCDP API: https://ucdp.uu.se/apidocs/
V-Dem: https://www.v-dem.net/data/the-v-dem-dataset/

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_ucdp_sample.py
python src/build_panel.py
python src/model_escalation.py
```
For the full dataset, replace the sample download with paginated UCDP GED ingestion or the bulk CSV release.
