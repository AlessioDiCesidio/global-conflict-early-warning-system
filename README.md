# Global Conflict Early-Warning System

A statistical and machine-learning project for analysing global conflict dynamics and estimating the probability of major conflict escalation within the following six months.

The project combines historical conflict data, country-level characteristics, machine learning and geospatial analysis into an interactive early-warning dashboard.

## Research question

Can historical conflict dynamics and country-level characteristics provide useful early-warning signals for subsequent conflict escalation?

The system is designed as a risk-ranking and analytical decision-support tool rather than a deterministic forecasting system.

## Main design

- **Unit of analysis:** country-month
- **Historical period:** 1989–2025
- **Prediction horizon:** 6 months
- **Training period:** 1989–2019
- **Out-of-sample evaluation:** 2020–2025

For each country-month observation, the model estimates the probability that a major conflict escalation will occur during the following six months.

The chronological train-test split prevents future observations from entering the training data and provides a more realistic evaluation of prospective predictive performance.

## Data

The main historical conflict data come from the **UCDP Georeferenced Event Dataset (GED)**.

Additional country-level information is used to capture political, socioeconomic and demographic characteristics.

Main sources:

- UCDP Georeferenced Event Dataset
- V-Dem
- World Bank World Development Indicators

## Models

The main predictive model is a **Gradient Boosting Classifier**.

**Logistic Regression** is used as a benchmark to assess whether the nonlinear model provides additional predictive value.

The project also includes:

- descriptive conflict analysis
- temporal performance analysis
- probability calibration
- permutation feature importance
- ablation analysis
- threshold sensitivity analysis
- risk concentration analysis

## Model performance

The current out-of-sample evaluation produces:

| Metric | Result |
|---|---:|
| Gradient Boosting ROC-AUC | 0.882 |
| Gradient Boosting PR-AUC | 0.287 |
| Logistic Regression ROC-AUC | 0.844 |
| Brier Score | 0.022 |
| PR-AUC Lift | 8.73× |

The Gradient Boosting model therefore provides stronger discrimination than the Logistic Regression benchmark.

The PR-AUC is particularly relevant because major escalation events are relatively rare in the dataset.

## Early-warning framework

The model produces a continuous probability score rather than a binary prediction.

Two thresholds are used for operational analysis.

### Early-Warning Threshold: 0.20

The 0.20 threshold prioritises recall.

Country-month observations with an estimated escalation probability of at least 20% are flagged for additional analytical attention.

This setting accepts more false alarms in exchange for detecting a larger proportion of subsequent escalation events.

### Conservative Threshold: 0.50

The 0.50 threshold produces a more selective alert system.

It generates fewer alerts and places greater emphasis on precision, but also misses a larger share of observed escalation events.

The choice of threshold therefore depends on the relative cost of false alarms and missed escalations.

## Risk ranking

The model can also be used as a ranking system.

Instead of applying a fixed probability threshold, observations can be ranked according to predicted escalation risk and the highest-risk observations can be monitored first.

Current out-of-sample results:

| Highest-risk observations | Escalations captured |
|---|---:|
| Top 1% | 14.7% |
| Top 5% | 45.4% |
| Top 10% | 57.5% |
| Top 20% | 75.8% |
| Top 30% | 85.3% |

This is useful when analytical resources are limited and attention needs to be concentrated on a smaller number of observations.

The ranking should not be interpreted as a classification of countries as safe or unsafe.

## Model diagnostics

The project includes several diagnostic analyses.

### Probability calibration

Predicted probabilities are compared with observed escalation frequencies to assess whether the model's probabilities are reasonably calibrated.

### Temporal stability

ROC-AUC and PR-AUC are evaluated separately across the 2020–2025 out-of-sample period.

### Ablation analysis

Different model specifications are compared to assess whether socioeconomic and demographic variables provide additional predictive information beyond historical conflict characteristics.

### Feature importance

Permutation importance is used to identify the variables that contribute most to predictive performance.

These results describe predictive association and should not be interpreted as causal effects.

### Threshold sensitivity

Precision and recall are evaluated across different probability thresholds to show the trade-off between false alarms and missed escalation events.

## Dashboard

The project includes an interactive Streamlit dashboard with the following sections:

- **Overview** — global conflict indicators and historical trends
- **Risk Monitor** — country-level predicted risk and interactive maps
- **Model Performance** — out-of-sample model performance and calibration
- **Early Warning** — operational thresholds, precision-recall trade-offs and risk ranking
- **Diagnostics** — calibration, temporal stability, ablation and feature importance
- **Methodology** — modelling framework, validation strategy and limitations

## Project structure

    global-conflict-early-warning-system/
    │
    ├── app.py
    ├── README.md
    ├── requirements.txt
    ├── analysis_plan.md
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── notebooks/
    │   ├── 01_eda.py
    │   ├── 02_rankings.py
    │   ├── 03_dynamics.py
    │   ├── 04_escalation.py
    │   ├── 05_features.py
    │   ├── 06_model.py
    │   ├── 08_external_data.py
    │   ├── 09_merge_external.py
    │   ├── 10_external_features.py
    │   └── 11_external_model.py
    │
    ├── src/
    │   ├── build_panel.py
    │   ├── download_ucdp_sample.py
    │   └── model_escalation.py
    │
    └── outputs/

## Running the project

Create a virtual environment:

    python -m venv .venv

On Windows:

    .venv\Scripts\Activate.ps1

Install the required packages:

    pip install -r requirements.txt

Run the dashboard:

    streamlit run app.py

The analytical and modelling scripts are contained in `src/` and `notebooks/`.

## Limitations

The model is based on historical relationships and these relationships may change over time.

Other limitations include:

- conflict events are subject to measurement and reporting limitations;
- major geopolitical shocks may create conditions not represented in the training data;
- model probabilities depend on the available features;
- country-level aggregation can hide substantial within-country variation;
- prediction does not imply causation;
- the six-month prediction horizon limits interpretation beyond that period.

## Interpretation

This project should be understood as an **early-warning and decision-support system**, not as an autonomous geopolitical forecasting system.

A high predicted probability means that an observation has characteristics associated with a higher probability of subsequent escalation in the historical data. It does not mean that escalation will necessarily occur.

Similarly, a low predicted probability does not imply that a country is safe.

The main purpose of the system is therefore to help prioritise analytical attention.

## Data sources

**UCDP**

https://ucdp.uu.se/downloads/

**UCDP API**

https://ucdp.uu.se/apidocs/

**V-Dem**

https://www.v-dem.net/data/the-v-dem-dataset/

**World Bank World Development Indicators**

https://data.worldbank.org/

## Disclaimer

This project is intended for research, analytical and educational purposes.

The predictions are statistical estimates based on historical data and should not be interpreted as definitive assessments of future political or military events.
