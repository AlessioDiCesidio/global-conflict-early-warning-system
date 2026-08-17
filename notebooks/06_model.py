import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ---------------------------------------------------------
# LOAD FEATURES
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/country_month_features.csv"
)

df["month"] = pd.to_datetime(df["month"])

df = df.sort_values(
    ["month", "country"]
).reset_index(drop=True)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

features = [
    "fatalities_lag_1",
    "fatalities_lag_3",
    "fatalities_lag_6",
    "regime_lag_1",
    "regime_lag_3",
    "regime_lag_6",
    "fatalities_mean_3m",
    "fatalities_mean_6m",
    "fatalities_mean_12m",
    "fatalities_std_6m",
    "fatalities_std_12m",
    "regime_change_1m",
    "regime_change_3m",
    "months_active_12m"
]


# ---------------------------------------------------------
# REMOVE MISSING TARGET
# ---------------------------------------------------------

model_df = df.loc[
    df["major_escalation_6m"].notna()
].copy()


# ---------------------------------------------------------
# TEMPORAL SPLIT
# ---------------------------------------------------------

cutoff = pd.Timestamp("2020-01-01")

train = model_df[
    model_df["month"] < cutoff
].copy()

test = model_df[
    model_df["month"] >= cutoff
].copy()


print("\nTRAIN / TEST SPLIT\n")

print(
    f"Train: {len(train):,}"
)

print(
    f"Test:  {len(test):,}"
)

print(
    f"Train period: "
    f"{train['month'].min().date()} → "
    f"{train['month'].max().date()}"
)

print(
    f"Test period:  "
    f"{test['month'].min().date()} → "
    f"{test['month'].max().date()}"
)


# ---------------------------------------------------------
# X / y
# ---------------------------------------------------------

X_train = train[features]

y_train = train[
    "major_escalation_6m"
].astype(int)

X_test = test[features]

y_test = test[
    "major_escalation_6m"
].astype(int)


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)





# ---------------------------------------------------------
# FIT
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

y_prob = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_prob >= 0.5
).astype(int)

# ---------------------------------------------------------
# THRESHOLD ANALYSIS
# ---------------------------------------------------------

print("\n\nTHRESHOLD ANALYSIS\n")

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

for threshold in thresholds:

    pred = (
        y_prob >= threshold
    ).astype(int)

    precision_t = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall_t = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1_t = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    alerts = pred.sum()

    print(
        f"Threshold={threshold:.2f} | "
        f"Alerts={alerts:,} | "
        f"Precision={precision_t:.3f} | "
        f"Recall={recall_t:.3f} | "
        f"F1={f1_t:.3f}"
    )

# ---------------------------------------------------------
# RISK RANKING ANALYSIS
# ---------------------------------------------------------

print("\n\nRISK RANKING ANALYSIS\n")

ranking_df = test[
    ["country", "month", "major_escalation_6m"]
].copy()

ranking_df["predicted_risk"] = y_prob

ranking_df = ranking_df.sort_values(
    "predicted_risk",
    ascending=False
).reset_index(drop=True)


total_events = y_test.sum()

print(
    f"\nTotal major escalations in test: "
    f"{int(total_events):,}"
)


for pct in [0.01, 0.05, 0.10, 0.20, 0.30]:

    n = int(
        len(ranking_df) * pct
    )

    top = ranking_df.iloc[:n]

    events_found = top[
        "major_escalation_6m"
    ].sum()

    precision_top = (
        events_found / n
        if n > 0
        else 0
    )

    recall_top = (
        events_found / total_events
        if total_events > 0
        else 0
    )

    print(
        f"Top {pct:.0%}: "
        f"{n:,} observations | "
        f"Events={int(events_found):,} | "
        f"Precision={precision_top:.3f} | "
        f"Recall={recall_top:.3f}"
    )

# ---------------------------------------------------------
# GRADIENT BOOSTING
# ---------------------------------------------------------

print("\n\nGRADIENT BOOSTING MODEL\n")


gb_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ]
)


# ---------------------------------------------------------
# FIT
# ---------------------------------------------------------

gb_model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

gb_prob = gb_model.predict_proba(
    X_test
)[:, 1]


gb_pred = (
    gb_prob >= 0.5
).astype(int)


# ---------------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------------

gb_roc_auc = roc_auc_score(
    y_test,
    gb_prob
)

gb_pr_auc = average_precision_score(
    y_test,
    gb_prob
)

gb_precision = precision_score(
    y_test,
    gb_pred,
    zero_division=0
)

gb_recall = recall_score(
    y_test,
    gb_pred,
    zero_division=0
)

gb_f1 = f1_score(
    y_test,
    gb_pred,
    zero_division=0
)


print("\nMODEL PERFORMANCE\n")

print(
    f"ROC-AUC:  {gb_roc_auc:.4f}"
)

print(
    f"PR-AUC:   {gb_pr_auc:.4f}"
)

print(
    f"Precision: {gb_precision:.4f}"
)

print(
    f"Recall:    {gb_recall:.4f}"
)

print(
    f"F1:        {gb_f1:.4f}"
)


# ---------------------------------------------------------
# RISK RANKING
# ---------------------------------------------------------

print("\n\nGRADIENT BOOSTING RISK RANKING\n")


gb_ranking = test[
    ["country", "month", "major_escalation_6m"]
].copy()

gb_ranking["predicted_risk"] = gb_prob

gb_ranking = gb_ranking.sort_values(
    "predicted_risk",
    ascending=False
).reset_index(drop=True)


total_events = y_test.sum()


for pct in [0.01, 0.05, 0.10, 0.20, 0.30]:

    n = int(
        len(gb_ranking) * pct
    )

    top = gb_ranking.iloc[:n]

    events_found = top[
        "major_escalation_6m"
    ].sum()

    precision_top = (
        events_found / n
    )

    recall_top = (
        events_found / total_events
    )

    print(
        f"Top {pct:.0%}: "
        f"{n:,} observations | "
        f"Events={int(events_found):,} | "
        f"Precision={precision_top:.3f} | "
        f"Recall={recall_top:.3f}"
    )


# ---------------------------------------------------------
# PERMUTATION IMPORTANCE
# ---------------------------------------------------------

from sklearn.inspection import permutation_importance

print("\n\nPERMUTATION IMPORTANCE\n")

importance = permutation_importance(
    gb_model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
})

feature_importance = feature_importance.sort_values(
    "importance_mean",
    ascending=False
)

print(
    feature_importance.to_string(index=False)
)

# ---------------------------------------------------------
# ABLATION STUDY
# ---------------------------------------------------------

print("\n\nABLATION STUDY\n")

feature_sets = {

    "A_PERSISTENCE": [
        "fatalities_lag_1"
    ],

    "B_FATALITIES_HISTORY": [
        "fatalities_lag_1",
        "fatalities_lag_3",
        "fatalities_lag_6"
    ],

    "C_FATALITIES_VOLATILITY": [
        "fatalities_lag_1",
        "fatalities_lag_3",
        "fatalities_lag_6",
        "fatalities_std_6m",
        "fatalities_std_12m"
    ],

    "D_REGIME_DYNAMICS": [
        "fatalities_lag_1",
        "fatalities_lag_3",
        "fatalities_lag_6",
        "fatalities_std_6m",
        "fatalities_std_12m",
        "regime_lag_1",
        "regime_lag_3",
        "regime_lag_6",
        "regime_change_1m",
        "regime_change_3m"
    ],

    "E_FULL_UCDP": [
        "fatalities_lag_1",
        "fatalities_lag_3",
        "fatalities_lag_6",
        "regime_lag_1",
        "regime_lag_3",
        "regime_lag_6",
        "fatalities_mean_3m",
        "fatalities_mean_6m",
        "fatalities_mean_12m",
        "fatalities_std_6m",
        "fatalities_std_12m",
        "regime_change_1m",
        "regime_change_3m",
        "months_active_12m"
    ]
}

for name, features in feature_sets.items():

    print(f"\n{name}")

    X_train_ab = X_train[features]
    X_test_ab = X_test[features]

    model_ab = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ]
    )

    model_ab.fit(
        X_train_ab,
        y_train
    )

    prob_ab = model_ab.predict_proba(
        X_test_ab
    )[:, 1]

    roc_auc_ab = roc_auc_score(
        y_test,
        prob_ab
    )

    pr_auc_ab = average_precision_score(
        y_test,
        prob_ab
    )

    print(
        f"ROC-AUC={roc_auc_ab:.4f} | "
        f"PR-AUC={pr_auc_ab:.4f}"
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n\nMODEL PERFORMANCE\n")

print(
    f"ROC-AUC:  {roc_auc:.4f}"
)

print(
    f"PR-AUC:   {pr_auc:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1:        {f1:.4f}"
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

print("\n\nCONFUSION MATRIX\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ---------------------------------------------------------
# TEST TARGET RATE
# ---------------------------------------------------------

print("\n\nTEST TARGET RATE\n")

print(
    y_test.mean()
)

############


import pandas as pd

df = pd.read_csv(
    "data/processed/country_month_panel_full.csv"
)

print(df.columns.tolist())