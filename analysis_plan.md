# Analysis plan — Global Conflict Index

## Phase 1 — Data foundation
1. UCDP GED 26.1: full 1989–2025 event data.
2. UCDP Candidate Events: monthly 2026 extension. Candidate data are provisional; freeze a release date for each analysis.
3. UCDP/PRIO Armed Conflict 26.1: 1946–2025 conflict-year panel.
4. UCDP Conflict Termination Dataset v4-2024 for duration/termination.
5. V-Dem v16 for political covariates.
6. World Bank WDI for GDP/population.

## Phase 2 — Data engineering
Create a complete country-month calendar. Merge UCDP event aggregates into it so months with zero events are explicit zeros. Preserve raw event data and create a data dictionary.

Core measures:
- event_count
- fatalities_total
- battle_related_fatalities
- civilian_fatalities
- active_conflict_count
- conflict_intensity_6m
- intensity_change_6m
- neighbour_intensity

## Phase 3 — Escalation definition
Primary outcome: material increase in violence over the next six months.

Primary threshold:
future 6-month mean fatalities >= current 6-month mean * 1.50

Robustness thresholds: 25%, 50%, 100%; also test count-based outcomes and percentile-based definitions.

Avoid leakage: every predictor must be observable at t; outcome only uses t+1...t+6.

## Phase 4 — Models
Baseline:
- persistence model
- logistic regression

Inference:
- country fixed effects
- year/month fixed effects where appropriate
- clustered standard errors

ML:
- random forest
- gradient boosting

Metrics:
- ROC-AUC
- PR-AUC
- Brier score
- calibration curve
- recall at top-k risk countries

Validation:
- temporal holdout for final forecasting
- group-aware CV for robustness

## Phase 5 — Termination
Use UCDP termination episodes.
- Kaplan–Meier
- Cox PH
- AFT robustness model

Unit: conflict episode.
Censoring and termination type must be explicitly coded.

## Phase 6 — Spillover
Build country adjacency matrix. Exposure at t = weighted average of neighbours' conflict intensity at t. Estimate whether neighbour exposure predicts onset/escalation, controlling for own lagged violence and country/year effects.

## Phase 7 — Momentum indicator
Do not claim to measure victory. Build a transparent indicator from measurable changes:
- territorial control, only where reliable data exist
- conflict event intensity
- fatalities
- strategic-location control
- operational persistence

Standardize components. Publish equal-weight and alternative-weight versions. Report sensitivity intervals.

## Phase 8 — Outputs
- global conflict dashboard
- top current conflict intensities
- 6-month escalation probabilities
- survival curves
- spillover network
- country profiles
- reproducible report
- TikTok-ready charts with a clear methodology note
