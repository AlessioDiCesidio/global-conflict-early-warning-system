import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ---------------------------------------------------------
# LOAD UCDP FEATURES
# ---------------------------------------------------------

ucdp = pd.read_csv(
    "data/processed/country_month_features.csv"
)

ucdp["month"] = pd.to_datetime(
    ucdp["month"]
)


# ---------------------------------------------------------
# LOAD EXTERNAL FEATURES
# ---------------------------------------------------------

external = pd.read_csv(
    "data/processed/country_month_panel_external_features.csv"
)

external["month"] = pd.to_datetime(
    external["month"]
)


# ---------------------------------------------------------
# MERGE UCDP + EXTERNAL FEATURES
# ---------------------------------------------------------

df = ucdp.merge(
    external[
        [
            "country",
            "month",
            "fatalities_per_100k",
            "log_gdp_per_capita",
            "log_population",
            "population_lag_12",
            "gdp_per_capita_lag_12",
            "gdp_growth_lag_12"
        ]
    ],
    on=["country", "month"],
    how="left"
)


# ---------------------------------------------------------
# CHECK MERGE
# ---------------------------------------------------------

print("\nMERGED SHAPE")
print(df.shape)

print("\nEXTERNAL FEATURE COVERAGE")

print(
    df[
        [
            "fatalities_per_100k",
            "log_gdp_per_capita",
            "log_population",
            "population_lag_12",
            "gdp_per_capita_lag_12",
            "gdp_growth_lag_12"
        ]
    ].notna().mean()
)


# ---------------------------------------------------------
# FEATURE SET
# ---------------------------------------------------------

features = [

    # UCDP persistence
    "fatalities_lag_1",
    "fatalities_lag_3",
    "fatalities_lag_6",

    "regime_lag_1",
    "regime_lag_3",
    "regime_lag_6",

    # UCDP history
    "fatalities_mean_3m",
    "fatalities_mean_6m",
    "fatalities_mean_12m",

    # UCDP volatility
    "fatalities_std_6m",
    "fatalities_std_12m",

    # UCDP regime dynamics
    "regime_change_1m",
    "regime_change_3m",
    "months_active_12m",

    # External
    "fatalities_per_100k",
    "log_gdp_per_capita",
    "log_population",
    "population_lag_12",
    "gdp_per_capita_lag_12",
    "gdp_growth_lag_12"
]


target = "major_escalation_6m"


# ---------------------------------------------------------
# MODEL DATASET
# ---------------------------------------------------------

df_model = df[
    ["country", "month"] + features + [target]
].copy()

df_model = df_model[
    df_model[target].notna()
].copy()


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

train_mask = (
    df_model["month"] < "2020-01-01"
)

test_mask = (
    df_model["month"] >= "2020-01-01"
)

train = df_model.loc[
    train_mask
].copy()

test = df_model.loc[
    test_mask
].copy()


X_train = train[features]
y_train = train[target].astype(int)

X_test = test[features]
y_test = test[target].astype(int)


print("\nTRAIN / TEST SPLIT")

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
    f"Test period: "
    f"{test['month'].min().date()} → "
    f"{test['month'].max().date()}"
)


# ---------------------------------------------------------
# IMPUTATION
# ---------------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train_imp = imputer.fit_transform(
    X_train
)

X_test_imp = imputer.transform(
    X_test
)


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imp
)

X_test_scaled = scaler.transform(
    X_test_imp
)


logistic = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

logistic.fit(
    X_train_scaled,
    y_train
)

logistic_prob = logistic.predict_proba(
    X_test_scaled
)[:, 1]


print("\n\nLOGISTIC REGRESSION")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, logistic_prob):.4f}"
)

print(
    f"PR-AUC: "
    f"{average_precision_score(y_test, logistic_prob):.4f}"
)


# ---------------------------------------------------------
# GRADIENT BOOSTING
# ---------------------------------------------------------

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb.fit(
    X_train_imp,
    y_train
)

gb_prob = gb.predict_proba(
    X_test_imp
)[:, 1]


gb_pred = (
    gb_prob >= 0.5
).astype(int)


print("\n\nGRADIENT BOOSTING")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, gb_prob):.4f}"
)

print(
    f"PR-AUC: "
    f"{average_precision_score(y_test, gb_prob):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, gb_pred, zero_division=0):.4f}"
)

print(
    f"Recall: "
    f"{recall_score(y_test, gb_pred, zero_division=0):.4f}"
)

print(
    f"F1: "
    f"{f1_score(y_test, gb_pred, zero_division=0):.4f}"
)

print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        gb_pred
    )
)

# =========================================================
# MODEL CALIBRATION AUDIT
# =========================================================

from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("\n")
print("=" * 70)
print("MODEL CALIBRATION AUDIT")
print("=" * 70)


# ---------------------------------------------------------
# 1. BASIC METRICS
# ---------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    gb_prob
)

pr_auc = average_precision_score(
    y_test,
    gb_prob
)

baseline_pr = y_test.mean()

brier = brier_score_loss(
    y_test,
    gb_prob
)

print("\nMODEL METRICS")

print(
    f"ROC-AUC:       {roc_auc:.4f}"
)

print(
    f"PR-AUC:        {pr_auc:.4f}"
)

print(
    f"Baseline PR:   {baseline_pr:.4f}"
)

print(
    f"Brier Score:   {brier:.4f}"
)

print(
    f"Positive rate:  {y_test.mean():.4f}"
)


# ---------------------------------------------------------
# 2. PR-AUC IMPROVEMENT OVER BASELINE
# ---------------------------------------------------------

pr_lift = (
    pr_auc / baseline_pr
)

print("\nPR-AUC LIFT OVER BASELINE")

print(
    f"Model / baseline: {pr_lift:.2f}x"
)


# ---------------------------------------------------------
# 3. CALIBRATION CURVE
# ---------------------------------------------------------

prob_true, prob_pred = calibration_curve(
    y_test,
    gb_prob,
    n_bins=10,
    strategy="quantile"
)


calibration_df = pd.DataFrame({
    "Predicted Probability": prob_pred,
    "Observed Frequency": prob_true
})


print("\nCALIBRATION TABLE")

print(
    calibration_df.to_string(
        index=False,
        formatters={
            "Predicted Probability":
                "{:.3f}".format,
            "Observed Frequency":
                "{:.3f}".format
        }
    )
)


# ---------------------------------------------------------
# 4. CALIBRATION PLOT
# ---------------------------------------------------------

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    linewidth=2,
    label="Gradient Boosting"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration"
)

plt.xlabel(
    "Mean predicted probability"
)

plt.ylabel(
    "Observed event frequency"
)

plt.title(
    "Calibration Curve — Gradient Boosting"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "06_calibration_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 5. SAVE CALIBRATION DATA
# ---------------------------------------------------------

calibration_df.to_csv(
    "calibration_analysis.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "06_calibration_curve.png"
)

print(
    "calibration_analysis.csv"
)


# ---------------------------------------------------------
# 6. INTERPRETATION
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("CALIBRATION AUDIT SUMMARY")
print("=" * 70)

print(
    f"""
ROC-AUC: {roc_auc:.3f}

PR-AUC: {pr_auc:.3f}

Baseline PR-AUC: {baseline_pr:.3f}

PR-AUC improvement over baseline: {pr_lift:.2f}x

Brier Score: {brier:.4f}

Positive event rate: {baseline_pr:.3f}
"""
)

# =========================================================
# TEMPORAL STABILITY AUDIT
# =========================================================

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

print("\n")
print("=" * 70)
print("TEMPORAL STABILITY AUDIT")
print("=" * 70)


# ---------------------------------------------------------
# 1. BUILD TEMPORAL RESULTS DATAFRAME
# ---------------------------------------------------------

temporal_results = test[
    ["month"]
].copy()

temporal_results["actual"] = y_test.values
temporal_results["probability"] = gb_prob

temporal_results["year"] = (
    temporal_results["month"]
    .dt.year
)


# ---------------------------------------------------------
# 2. YEARLY PERFORMANCE
# ---------------------------------------------------------

yearly_results = []


for year, group in temporal_results.groupby("year"):

    y_true_year = group["actual"]
    y_prob_year = group["probability"]

    events = int(
        y_true_year.sum()
    )

    observations = len(
        group
    )

    prevalence = (
        y_true_year.mean()
    )

    # ROC-AUC requires both classes
    if y_true_year.nunique() == 2:

        roc = roc_auc_score(
            y_true_year,
            y_prob_year
        )

        pr = average_precision_score(
            y_true_year,
            y_prob_year
        )

    else:

        roc = np.nan
        pr = np.nan

    brier = brier_score_loss(
        y_true_year,
        y_prob_year
    )

    yearly_results.append({

        "Year": year,

        "Observations":
            observations,

        "Events":
            events,

        "Event Rate":
            prevalence,

        "ROC-AUC":
            roc,

        "PR-AUC":
            pr,

        "Brier":
            brier

    })


yearly_df = pd.DataFrame(
    yearly_results
)


# ---------------------------------------------------------
# 3. DISPLAY RESULTS
# ---------------------------------------------------------

print("\nYEARLY MODEL PERFORMANCE\n")

print(
    yearly_df.to_string(
        index=False,
        formatters={

            "Event Rate":
                "{:.3f}".format,

            "ROC-AUC":
                lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "NA",

            "PR-AUC":
                lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "NA",

            "Brier":
                "{:.4f}".format

        }
    )
)


# ---------------------------------------------------------
# 4. SAVE RESULTS
# ---------------------------------------------------------

yearly_df.to_csv(
    "temporal_stability_analysis.csv",
    index=False
)


# ---------------------------------------------------------
# 5. SUMMARY STATISTICS
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("TEMPORAL STABILITY SUMMARY")
print("=" * 70)

print(
    "\nMean yearly ROC-AUC:",
    yearly_df["ROC-AUC"].mean()
)

print(
    "Minimum yearly ROC-AUC:",
    yearly_df["ROC-AUC"].min()
)

print(
    "Maximum yearly ROC-AUC:",
    yearly_df["ROC-AUC"].max()
)

print(
    "\nMean yearly PR-AUC:",
    yearly_df["PR-AUC"].mean()
)

print(
    "Minimum yearly PR-AUC:",
    yearly_df["PR-AUC"].min()
)

print(
    "Maximum yearly PR-AUC:",
    yearly_df["PR-AUC"].max()
)


# ---------------------------------------------------------
# 6. TEMPORAL PERFORMANCE PLOT
# ---------------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    yearly_df["Year"],
    yearly_df["ROC-AUC"],
    marker="o",
    linewidth=2,
    label="ROC-AUC"
)

plt.plot(
    yearly_df["Year"],
    yearly_df["PR-AUC"],
    marker="o",
    linewidth=2,
    label="PR-AUC"
)

plt.xlabel(
    "Test Year"
)

plt.ylabel(
    "Performance"
)

plt.title(
    "Temporal Stability — Out-of-Sample Model Performance"
)

plt.xticks(
    yearly_df["Year"]
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "07_temporal_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nSaved:"
)

print(
    "temporal_stability_analysis.csv"
)

print(
    "07_temporal_stability.png"
)

# =========================================================
# ABLATION STUDY — CONFLICT HISTORY vs EXTERNAL FEATURES
# =========================================================

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

print("\n")
print("=" * 70)
print("ABLATION STUDY — INCREMENTAL VALUE OF EXTERNAL FEATURES")
print("=" * 70)


# ---------------------------------------------------------
# 1. DEFINE FEATURE GROUPS
# ---------------------------------------------------------

conflict_features = [

    # Conflict persistence
    "fatalities_lag_1",
    "fatalities_lag_3",
    "fatalities_lag_6",

    # Regime persistence
    "regime_lag_1",
    "regime_lag_3",
    "regime_lag_6",

    # Historical intensity
    "fatalities_mean_3m",
    "fatalities_mean_6m",
    "fatalities_mean_12m",

    # Volatility
    "fatalities_std_6m",
    "fatalities_std_12m",

    # Regime dynamics
    "regime_change_1m",
    "regime_change_3m",
    "months_active_12m"
]


external_features = [

    "fatalities_per_100k",
    "log_gdp_per_capita",
    "log_population",
    "population_lag_12",
    "gdp_per_capita_lag_12",
    "gdp_growth_lag_12"
]


full_features = (
    conflict_features +
    external_features
)


print("\nFEATURE COUNTS")

print(
    "Conflict history:",
    len(conflict_features)
)

print(
    "External:",
    len(external_features)
)

print(
    "Full model:",
    len(full_features)
)


# ---------------------------------------------------------
# 2. FUNCTION TO TRAIN AND EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_feature_set(
    feature_list,
    model_name
):

    X_train_subset = train[
        feature_list
    ].copy()

    X_test_subset = test[
        feature_list
    ].copy()


    # -----------------------------------------------------
    # Imputation
    # -----------------------------------------------------

    subset_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp_subset = (
        subset_imputer.fit_transform(
            X_train_subset
        )
    )

    X_test_imp_subset = (
        subset_imputer.transform(
            X_test_subset
        )
    )


    # -----------------------------------------------------
    # Gradient Boosting
    # -----------------------------------------------------

    model = GradientBoostingClassifier(

        n_estimators=200,

        learning_rate=0.05,

        max_depth=3,

        random_state=42

    )


    model.fit(
        X_train_imp_subset,
        y_train
    )


    probability = model.predict_proba(
        X_test_imp_subset
    )[:, 1]


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    roc = roc_auc_score(
        y_test,
        probability
    )

    pr = average_precision_score(
        y_test,
        probability
    )

    brier = brier_score_loss(
        y_test,
        probability
    )


    return {

        "Model":
            model_name,

        "Features":
            len(feature_list),

        "ROC-AUC":
            roc,

        "PR-AUC":
            pr,

        "Brier":
            brier,

        "Probability":
            probability

    }


# ---------------------------------------------------------
# 3. TRAIN THREE MODELS
# ---------------------------------------------------------

print("\n")
print("TRAINING CONFLICT HISTORY MODEL...")

conflict_result = evaluate_feature_set(
    conflict_features,
    "Conflict History Only"
)


print(
    "Training completed."
)


print("\n")
print("TRAINING EXTERNAL MODEL...")

external_result = evaluate_feature_set(
    external_features,
    "External Features Only"
)


print(
    "Training completed."
)


print("\n")
print("TRAINING FULL MODEL...")

full_result = evaluate_feature_set(
    full_features,
    "Full Model"
)


print(
    "Training completed."
)


# ---------------------------------------------------------
# 4. BUILD COMPARISON TABLE
# ---------------------------------------------------------

ablation_results = pd.DataFrame([

    {
        "Model":
            conflict_result["Model"],

        "Features":
            conflict_result["Features"],

        "ROC-AUC":
            conflict_result["ROC-AUC"],

        "PR-AUC":
            conflict_result["PR-AUC"],

        "Brier":
            conflict_result["Brier"]
    },

    {
        "Model":
            external_result["Model"],

        "Features":
            external_result["Features"],

        "ROC-AUC":
            external_result["ROC-AUC"],

        "PR-AUC":
            external_result["PR-AUC"],

        "Brier":
            external_result["Brier"]
    },

    {
        "Model":
            full_result["Model"],

        "Features":
            full_result["Features"],

        "ROC-AUC":
            full_result["ROC-AUC"],

        "PR-AUC":
            full_result["PR-AUC"],

        "Brier":
            full_result["Brier"]
    }

])


# ---------------------------------------------------------
# 5. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("ABLATION RESULTS")
print("=" * 70)

print(

    ablation_results.to_string(

        index=False,

        formatters={

            "ROC-AUC":
                "{:.4f}".format,

            "PR-AUC":
                "{:.4f}".format,

            "Brier":
                "{:.4f}".format

        }

    )

)


# ---------------------------------------------------------
# 6. INCREMENTAL VALUE OF EXTERNAL FEATURES
# ---------------------------------------------------------

conflict_roc = (
    conflict_result["ROC-AUC"]
)

conflict_pr = (
    conflict_result["PR-AUC"]
)

full_roc = (
    full_result["ROC-AUC"]
)

full_pr = (
    full_result["PR-AUC"]
)


roc_gain = (
    full_roc -
    conflict_roc
)

pr_gain = (
    full_pr -
    conflict_pr
)


print("\n")
print("=" * 70)
print("INCREMENTAL VALUE OF EXTERNAL FEATURES")
print("=" * 70)

print(
    f"\nConflict-only ROC-AUC: "
    f"{conflict_roc:.4f}"
)

print(
    f"Full-model ROC-AUC:     "
    f"{full_roc:.4f}"
)

print(
    f"ROC-AUC improvement:    "
    f"{roc_gain:+.4f}"
)


print(
    f"\nConflict-only PR-AUC: "
    f"{conflict_pr:.4f}"
)

print(
    f"Full-model PR-AUC:    "
    f"{full_pr:.4f}"
)

print(
    f"PR-AUC improvement:   "
    f"{pr_gain:+.4f}"
)


# ---------------------------------------------------------
# 7. RELATIVE PR-AUC IMPROVEMENT
# ---------------------------------------------------------

if conflict_pr > 0:

    relative_pr_gain = (
        (full_pr - conflict_pr)
        / conflict_pr
    )

else:

    relative_pr_gain = np.nan


print(
    f"\nRelative PR-AUC improvement: "
    f"{relative_pr_gain:+.2%}"
)


# ---------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------

ablation_results.to_csv(
    "ablation_model_comparison.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "ablation_model_comparison.csv"
)


# ---------------------------------------------------------
# 9. VISUAL COMPARISON
# ---------------------------------------------------------

import matplotlib.pyplot as plt


plt.figure(
    figsize=(10, 7)
)


x = np.arange(
    len(ablation_results)
)

width = 0.35


plt.bar(
    x - width / 2,
    ablation_results["ROC-AUC"],
    width,
    label="ROC-AUC"
)


plt.bar(
    x + width / 2,
    ablation_results["PR-AUC"],
    width,
    label="PR-AUC"
)


plt.xticks(
    x,
    ablation_results["Model"],
    rotation=15
)


plt.ylabel(
    "Score"
)


plt.title(
    "Ablation Study — Contribution of External Features"
)


plt.ylim(
    0,
    max(
        ablation_results[
            ["ROC-AUC", "PR-AUC"]
        ].max()
    ) * 1.15
)


plt.grid(
    axis="y",
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "08_ablation_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "Saved: 08_ablation_model_comparison.png"
)

# =========================================================
# FINAL ABLATION DIAGNOSTICS
# FEATURE IMPORTANCE + PREDICTION COMPARISON
# =========================================================

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

print("\n")
print("=" * 70)
print("FINAL ABLATION DIAGNOSTICS")
print("=" * 70)


# =========================================================
# 1. RE-TRAIN MODELS AND KEEP OBJECTS
# =========================================================

def train_model_with_features(feature_list):

    X_tr = train[
        feature_list
    ].copy()

    X_te = test[
        feature_list
    ].copy()


    model_imputer = SimpleImputer(
        strategy="median"
    )


    X_tr_imp = model_imputer.fit_transform(
        X_tr
    )

    X_te_imp = model_imputer.transform(
        X_te
    )


    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )


    model.fit(
        X_tr_imp,
        y_train
    )


    probabilities = model.predict_proba(
        X_te_imp
    )[:, 1]


    return (
        model,
        model_imputer,
        X_te_imp,
        probabilities
    )


print("\nTraining Conflict History model...")

(
    conflict_model,
    conflict_imputer,
    X_conflict_test,
    conflict_prob
) = train_model_with_features(
    conflict_features
)


print("Training External model...")

(
    external_model,
    external_imputer,
    X_external_test,
    external_prob
) = train_model_with_features(
    external_features
)


print("Training Full model...")

(
    full_model,
    full_imputer,
    X_full_test,
    full_prob
) = train_model_with_features(
    full_features
)


# =========================================================
# 2. PERMUTATION IMPORTANCE — CONFLICT MODEL
# =========================================================

print("\n")
print("=" * 70)
print("CONFLICT HISTORY — PERMUTATION IMPORTANCE")
print("=" * 70)


conflict_perm = permutation_importance(
    conflict_model,
    X_conflict_test,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


conflict_importance = pd.DataFrame({

    "feature":
        conflict_features,

    "importance_mean":
        conflict_perm.importances_mean,

    "importance_std":
        conflict_perm.importances_std

}).sort_values(
    "importance_mean",
    ascending=False
)


print(
    conflict_importance.to_string(
        index=False
    )
)


# =========================================================
# 3. PERMUTATION IMPORTANCE — EXTERNAL MODEL
# =========================================================

print("\n")
print("=" * 70)
print("EXTERNAL FEATURES — PERMUTATION IMPORTANCE")
print("=" * 70)


external_perm = permutation_importance(
    external_model,
    X_external_test,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


external_importance = pd.DataFrame({

    "feature":
        external_features,

    "importance_mean":
        external_perm.importances_mean,

    "importance_std":
        external_perm.importances_std

}).sort_values(
    "importance_mean",
    ascending=False
)


print(
    external_importance.to_string(
        index=False
    )
)


# =========================================================
# 4. PERMUTATION IMPORTANCE — FULL MODEL
# =========================================================

print("\n")
print("=" * 70)
print("FULL MODEL — PERMUTATION IMPORTANCE")
print("=" * 70)


full_perm = permutation_importance(
    full_model,
    X_full_test,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


full_importance = pd.DataFrame({

    "feature":
        full_features,

    "importance_mean":
        full_perm.importances_mean,

    "importance_std":
        full_perm.importances_std

}).sort_values(
    "importance_mean",
    ascending=False
)


print(
    full_importance.to_string(
        index=False
    )
)


# =========================================================
# 5. SAVE IMPORTANCE TABLES
# =========================================================

conflict_importance.to_csv(
    "conflict_history_feature_importance.csv",
    index=False
)


external_importance.to_csv(
    "external_feature_importance.csv",
    index=False
)


full_importance.to_csv(
    "full_model_feature_importance.csv",
    index=False
)


print("\nSaved:")

print(
    "conflict_history_feature_importance.csv"
)

print(
    "external_feature_importance.csv"
)

print(
    "full_model_feature_importance.csv"
)


# =========================================================
# 6. TOP FULL-MODEL FEATURES
# =========================================================

top_full_features = (
    full_importance
    .head(15)
    .sort_values(
        "importance_mean",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 8)
)


plt.barh(
    top_full_features["feature"],
    top_full_features["importance_mean"],
    xerr=top_full_features["importance_std"]
)


plt.xlabel(
    "Decrease in Average Precision"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Full Model — Permutation Feature Importance"
)


plt.grid(
    axis="x",
    alpha=0.25
)


plt.tight_layout()


plt.savefig(
    "09_full_model_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "Saved: 09_full_model_feature_importance.png"
)


# =========================================================
# 7. COMPARE MODEL PROBABILITIES
# =========================================================

prediction_comparison = pd.DataFrame({

    "actual":
        y_test.values,

    "conflict_history_probability":
        conflict_prob,

    "external_probability":
        external_prob,

    "full_model_probability":
        full_prob

})


prediction_comparison[
    "external_adjustment"
] = (
    prediction_comparison[
        "full_model_probability"
    ]
    -
    prediction_comparison[
        "conflict_history_probability"
    ]
)


prediction_comparison[
    "absolute_external_adjustment"
] = (
    prediction_comparison[
        "external_adjustment"
    ].abs()
)


# =========================================================
# 8. PROBABILITY COMPARISON SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("MODEL PROBABILITY COMPARISON")
print("=" * 70)


print("\nMEAN PREDICTED PROBABILITY")

print(
    prediction_comparison[
        [
            "conflict_history_probability",
            "external_probability",
            "full_model_probability"
        ]
    ].mean().to_string()
)


print("\nMEDIAN PREDICTED PROBABILITY")

print(
    prediction_comparison[
        [
            "conflict_history_probability",
            "external_probability",
            "full_model_probability"
        ]
    ].median().to_string()
)


print("\nFULL MODEL ADJUSTMENT RELATIVE TO CONFLICT HISTORY")

print(
    prediction_comparison[
        "external_adjustment"
    ].describe()
)


print("\nMEAN ABSOLUTE ADJUSTMENT:")

print(
    prediction_comparison[
        "absolute_external_adjustment"
    ].mean()
)


# =========================================================
# 9. COUNTRIES / OBSERVATIONS MOST AFFECTED BY EXTERNAL
# =========================================================

# Recover country and month from test
prediction_comparison[
    "country"
] = test[
    "country"
].values if "country" in test.columns else np.nan


prediction_comparison[
    "month"
] = test[
    "month"
].values


largest_adjustments = (
    prediction_comparison
    .sort_values(
        "absolute_external_adjustment",
        ascending=False
    )
    .head(20)
)


print("\n")
print("=" * 70)
print("LARGEST EXTERNAL-FEATURE ADJUSTMENTS")
print("=" * 70)


print(
    largest_adjustments[
        [
            "country",
            "month",
            "actual",
            "conflict_history_probability",
            "full_model_probability",
            "external_adjustment"
        ]
    ].to_string(
        index=False
    )
)


largest_adjustments[
    [
        "country",
        "month",
        "actual",
        "conflict_history_probability",
        "full_model_probability",
        "external_adjustment"
    ]
].to_csv(
    "largest_external_adjustments.csv",
    index=False
)


print(
    "\nSaved: largest_external_adjustments.csv"
)


# =========================================================
# 10. CORRELATION BETWEEN CONFLICT-ONLY AND FULL MODEL
# =========================================================

prediction_correlation = (
    prediction_comparison[
        [
            "conflict_history_probability",
            "full_model_probability"
        ]
    ]
    .corr()
    .iloc[0, 1]
)


print("\n")
print("=" * 70)
print("PREDICTION CORRELATION")
print("=" * 70)


print(
    f"\nCorrelation between Conflict-only "
    f"and Full-model probabilities: "
    f"{prediction_correlation:.4f}"
)


# =========================================================
# 11. SAVE COMPLETE PREDICTION COMPARISON
# =========================================================

prediction_comparison.to_csv(
    "ablation_prediction_comparison.csv",
    index=False
)


print(
    "\nSaved: ablation_prediction_comparison.csv"
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 70)
print("FINAL ABLATION DIAGNOSTICS COMPLETED")
print("=" * 70)



# =========================================================
# THRESHOLD ANALYSIS — EARLY WARNING SYSTEM
# =========================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n")
print("=" * 60)
print("THRESHOLD ANALYSIS — GRADIENT BOOSTING")
print("=" * 60)

thresholds = [
    0.50,
    0.40,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05
]

threshold_results = []

for threshold in thresholds:

    predictions = (
        gb_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Predicted Events": predictions.sum(),
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print(
    "\n"
    + threshold_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1": "{:.3f}".format
        }
    )
)


threshold_df.to_csv(
    "threshold_analysis.csv",
    index=False
)


print(
    "\nSaved: threshold_analysis.csv"
)


# =========================================================
# THRESHOLD STABILITY CHECK
# =========================================================

stability_df = pd.DataFrame({
    "actual": y_test.reset_index(drop=True),
    "probability": gb_prob
})

print("\nTHRESHOLD STABILITY CHECK")

print(
    stability_df.head()
)

print(
    stability_df.columns
)

print("\nTEST DATA COLUMNS:")

print(
    test.columns.tolist()
)

# =========================================================
# THRESHOLD SENSITIVITY — PRECISION / RECALL / F1
# =========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    linewidth=2,
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    linewidth=2,
    label="F1"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Threshold Sensitivity — Gradient Boosting Early-Warning Model"
)

plt.xticks(
    threshold_df["Threshold"]
)

plt.ylim(
    0,
    0.8
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "05_threshold_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: 05_threshold_sensitivity.png"
)

# =========================================================
# 0.50 vs 0.20 — EARLY WARNING COMPARISON
# =========================================================

comparison_thresholds = [0.50, 0.20]

comparison_results = []

for threshold in comparison_thresholds:

    predictions = (
        gb_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    comparison_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Alerts": predictions.sum()
    })


comparison_df = pd.DataFrame(
    comparison_results
)

print("\n")
print("=" * 60)
print("CONSERVATIVE vs EARLY-WARNING THRESHOLD")
print("=" * 60)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1": "{:.3f}".format
        }
    )
)

comparison_df.to_csv(
    "threshold_comparison.csv",
    index=False
)

print("\nSaved: threshold_comparison.csv")


# ---------------------------------------------------------
# PERMUTATION IMPORTANCE - EXTERNAL MODEL
# ---------------------------------------------------------

from sklearn.inspection import permutation_importance

# Use the same imputed data format used to fit the model
X_test_perm = X_test_imp.copy()
y_test_perm = y_test.copy()

print("\nPermutation importance sample:")
print("Observations:", len(X_test_perm))
print("Features:", X_test_perm.shape[1])

perm = permutation_importance(
    gb,
    X_test_perm,
    y_test_perm,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": features,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(
    "importance_mean",
    ascending=False
)

print("\n\nPERMUTATION IMPORTANCE - EXTERNAL MODEL\n")

print(
    importance_df.to_string(index=False)
)

# ---------------------------------------------------------
# RISK RANKING
# ---------------------------------------------------------

print("\n\nEXTERNAL MODEL RISK RANKING")

results = test[
    ["month"]
].copy()

results["actual"] = y_test.values

results["probability"] = gb_prob

results = results.sort_values(
    "probability",
    ascending=False
).reset_index(drop=True)


total_events = results["actual"].sum()

print(
    f"\nTotal major escalations in test: "
    f"{int(total_events):,}"
)


for pct in [
    0.01,
    0.05,
    0.10,
    0.20,
    0.30
]:

    n = max(
        1,
        int(len(results) * pct)
    )

    top = results.head(n)

    events = top["actual"].sum()

    precision = events / n

    recall = (
        events / total_events
        if total_events > 0
        else 0
    )

    print(
        f"Top {int(pct * 100)}%: "
        f"{n:,} observations | "
        f"Events={int(events):,} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f}"
    )

# =========================================================
# GLOBAL RISK MAP — LATEST AVAILABLE COUNTRY RISK
# =========================================================

print("\n\nBUILDING GLOBAL RISK MAP DATA")

# ---------------------------------------------------------
# 1. Recover country information
# ---------------------------------------------------------

# Load the original panel containing country information
country_panel = pd.read_csv(
    r"data\processed\country_month_panel_external_features.csv"
)
country_panel["month"] = pd.to_datetime(
    country_panel["month"]
)

# Make sure test month has the same format
test_month = pd.to_datetime(
    test["month"]
)

# ---------------------------------------------------------
# 2. Match test observations to original country-month data
# ---------------------------------------------------------

# The test set preserves the original row index.
# Use those indices to recover country information.

country_info = country_panel.loc[
    test.index,
    [
        "country",
        "country_id",
        "region",
        "month"
    ]
].copy()

country_info = country_info.reset_index(drop=True)

# ---------------------------------------------------------
# 3. Build CURRENT RISK MAP dataset
# ---------------------------------------------------------

print("\nBUILDING CURRENT RISK MAP")


# ---------------------------------------------------------
# Load complete feature datasets
# ---------------------------------------------------------

map_feature_data = pd.read_csv(
    "data/processed/country_month_features.csv"
)

map_feature_data["month"] = pd.to_datetime(
    map_feature_data["month"]
)


map_external_data = pd.read_csv(
    "data/processed/country_month_panel_external_features.csv"
)

map_external_data["month"] = pd.to_datetime(
    map_external_data["month"]
)


# ---------------------------------------------------------
# Merge feature datasets
# ---------------------------------------------------------

map_data = map_feature_data.merge(
    map_external_data[
        [
            "country",
            "month",
            "fatalities_per_100k",
            "log_gdp_per_capita",
            "log_population",
            "population_lag_12",
            "gdp_per_capita_lag_12",
            "gdp_growth_lag_12"
        ]
    ],
    on=[
        "country",
        "month"
    ],
    how="left"
)


# ---------------------------------------------------------
# Latest available observation for each country
# ---------------------------------------------------------

map_data = (
    map_data
    .sort_values("month")
    .groupby(
        "country",
        as_index=False
    )
    .tail(1)
    .copy()
)


print(
    "\nLatest map date:",
    map_data["month"].max()
)

print(
    "Countries available:",
    len(map_data)
)


# ---------------------------------------------------------
# Check model features
# ---------------------------------------------------------

missing_map_features = [
    col
    for col in features
    if col not in map_data.columns
]

if missing_map_features:

    raise ValueError(
        "Missing map features: "
        + str(missing_map_features)
    )


# ---------------------------------------------------------
# Prepare model input
# ---------------------------------------------------------

X_map = map_data[
    features
].copy()


X_map_imp = pd.DataFrame(
    imputer.transform(X_map),
    columns=features,
    index=map_data.index
)


# ---------------------------------------------------------
# Predict current six-month risk
# ---------------------------------------------------------

map_data["predicted_risk"] = (
    gb.predict_proba(
        X_map_imp
    )[:, 1]
)


# ---------------------------------------------------------
# Early-warning classification
# ---------------------------------------------------------

map_data["early_warning"] = (
    map_data["predicted_risk"] >= 0.20
).astype(int)


# ---------------------------------------------------------
# Check key countries
# ---------------------------------------------------------

print(
    "\nCURRENT RISK CHECK"
)

print(
    map_data[
        map_data["country"].isin(
            [
                "Ukraine",
                "Israel",
                "Iran"
            ]
        )
    ][
        [
            "country",
            "predicted_risk",
            "fatalities_per_100k",
            "month"
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Keep compatibility with the rest of the script
# ---------------------------------------------------------

map_latest = map_data.copy()

# ---------------------------------------------------------
# STANDARDIZE COUNTRY NAMES FOR PLOTLY
# ---------------------------------------------------------

map_plot = map_latest.copy()

map_plot["country_plot"] = (
    map_plot["country"]
    .replace({
        "Yemen (North Yemen)": "Yemen",
        "Russia (Soviet Union)": "Russia",
        "Myanmar (Burma)": "Myanmar",
        "DR Congo (Zaire)": "Democratic Republic of the Congo"
    })
)


# ---------------------------------------------------------
# 5. Sort by predicted risk
# ---------------------------------------------------------

map_latest = map_latest[
    [
        "country",
        "country_id",
        "region",
        "month",
        "predicted_risk",
        "fatalities_per_100k"
    ]
].sort_values(
    "predicted_risk",
    ascending=False
)

# ---------------------------------------------------------
# 6. Display top 20 countries
# ---------------------------------------------------------

print("\nTOP 20 COUNTRIES BY PREDICTED RISK\n")

print(
    map_latest.head(20).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 7. Save dataset
# ---------------------------------------------------------

map_latest.to_csv(
    "global_risk_map_data.csv",
    index=False
)

print(
    "\nSaved: global_risk_map_data.csv"
)

print(
    "Countries:",
    len(map_latest)
)

# =========================================================
# FINAL MODEL VALIDATION VISUALIZATIONS
# =========================================================

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

print("\n\nBUILDING FINAL MODEL VALIDATION PLOTS")


# ---------------------------------------------------------
# 1. ROC CURVE
# ---------------------------------------------------------

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    logistic_prob
)

fpr_gb, tpr_gb, _ = roc_curve(
    y_test,
    gb_prob
)

roc_lr = roc_auc_score(
    y_test,
    logistic_prob
)

roc_gb = roc_auc_score(
    y_test,
    gb_prob
)

plt.figure(figsize=(9, 7))

plt.plot(
    fpr_lr,
    tpr_lr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {roc_lr:.3f})"
)

plt.plot(
    fpr_gb,
    tpr_gb,
    linewidth=2,
    label=f"Gradient Boosting (AUC = {roc_gb:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve — Conflict Escalation Prediction"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "01_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 2. PRECISION–RECALL CURVE
# ---------------------------------------------------------

precision_lr, recall_lr, _ = precision_recall_curve(
    y_test,
    logistic_prob
)

precision_gb, recall_gb, _ = precision_recall_curve(
    y_test,
    gb_prob
)

pr_lr = average_precision_score(
    y_test,
    logistic_prob
)

pr_gb = average_precision_score(
    y_test,
    gb_prob
)

baseline = y_test.mean()

plt.figure(figsize=(9, 7))

plt.plot(
    recall_lr,
    precision_lr,
    linewidth=2,
    label=f"Logistic Regression (AP = {pr_lr:.3f})"
)

plt.plot(
    recall_gb,
    precision_gb,
    linewidth=2,
    label=f"Gradient Boosting (AP = {pr_gb:.3f})"
)

plt.axhline(
    baseline,
    linestyle="--",
    linewidth=1,
    label=f"Baseline = {baseline:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision–Recall Curve — Conflict Escalation Prediction"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "02_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 3. PERMUTATION IMPORTANCE
# ---------------------------------------------------------

top_features = (
    importance_df
    .sort_values(
        "importance_mean",
        ascending=True
    )
    .tail(15)
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"]
)

plt.xlabel(
    "Decrease in Average Precision"
)

plt.ylabel("Feature")

plt.title(
    "Gradient Boosting — Permutation Feature Importance"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "03_permutation_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 4. TOP 20 COUNTRY RISK RANKING
# ---------------------------------------------------------

top20 = (
    map_latest
    .sort_values(
        "predicted_risk",
        ascending=True
    )
    .tail(20)
)

plt.figure(figsize=(10, 9))

plt.barh(
    top20["country"],
    top20["predicted_risk"]
)

plt.xlabel(
    "Predicted Risk"
)

plt.ylabel("Country")

plt.title(
    "Top 20 Countries by Predicted Conflict Escalation Risk"
)

plt.xlim(
    0,
    max(top20["predicted_risk"]) * 1.15
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "04_country_risk_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nFINAL VALIDATION PLOTS SAVED:")
print("01_roc_curve.png")
print("02_precision_recall_curve.png")
print("03_permutation_importance.png")
print("04_country_risk_ranking.png")

# ---------------------------------------------------------
# GLOBAL CONFLICT MAPS - FINAL VERSION
# ---------------------------------------------------------

import plotly.express as px

print("\nBUILDING GLOBAL CONFLICT MAPS")


# =========================================================
# MAP 1 — PREDICTED ESCALATION RISK
# =========================================================

# ---------------------------------------------------------
# STANDARDIZE COUNTRY NAMES FOR PLOTLY
# ---------------------------------------------------------

map_plot = map_latest.copy()

map_plot["country_plot"] = map_plot["country"].replace({
    "Yemen (North Yemen)": "Yemen",
    "Russia (Soviet Union)": "Russia",
    "Myanmar (Burma)": "Myanmar",
    "DR Congo (Zaire)": "Democratic Republic of the Congo"
})

# =========================================================
# FINAL MAP DATA CHECK
# =========================================================

print("\n" + "=" * 70)
print("FINAL MAP DATA CHECK")
print("=" * 70)

print("\nMAP COLUMNS:")
print(map_plot.columns.tolist())

print("\nMAP SHAPE:")
print(map_plot.shape)

print("\nPREDICTED RISK:")
print(map_plot["predicted_risk"].describe())

print("\nFATALITIES PER 100K:")
print(map_plot["fatalities_per_100k"].describe())

print("\nCOUNTRIES:")
print(map_plot["country"].nunique())

print("\nMISSING VALUES:")
print(
    map_plot[
        [
            "country",
            "predicted_risk",
            "fatalities_per_100k",
            "month"
        ]
    ].isna().sum()
)

print("\nUKRAINE / ISRAEL / IRAN:")
print(
    map_plot[
        map_plot["country"].isin(
            ["Ukraine", "Israel", "Iran"]
        )
    ][
        [
            "country",
            "predicted_risk",
            "fatalities_per_100k",
            "month"
        ]
    ].to_string(index=False)
)

print("=" * 70)



fig_risk = px.choropleth(
    map_plot,
    locations="country_plot",
    locationmode="country names",
    color="predicted_risk",
    hover_name="country",
    hover_data={
        "predicted_risk": ":.3f",
        "fatalities_per_100k": ":.2f",
        "region": True,
        "month": True
    },
    color_continuous_scale=[
        [0.0, "white"],
        [0.25, "#fee5d9"],
        [0.5, "#fcae91"],
        [0.75, "#fb6a4a"],
        [1.0, "#99000d"]
    ],
    range_color=[0, 1],
    title="Global Conflict Escalation Risk"
)

fig_risk.update_geos(
    showframe=False,
    showcoastlines=True,
    showland=True,
    landcolor="lightgray",
    showcountries=True,
    countrycolor="white"
)

fig_risk.update_layout(
    geo=dict(
        projection_type="natural earth"
    ),
    coloraxis_colorbar=dict(
        title="Predicted<br>Risk"
    )
)

fig_risk.write_html(
    "global_conflict_risk_map.html"
)

print("Saved: global_conflict_risk_map.html")


# =========================================================
# MAP 2 — CURRENT CONFLICT INTENSITY
# =========================================================

fig_observed = px.choropleth(
    map_plot,
    locations="country_plot",
    locationmode="country names",
    color="fatalities_per_100k",
    hover_name="country",
    hover_data={
        "fatalities_per_100k": ":.2f",
        "predicted_risk": ":.3f",
        "region": True,
        "month": True
    },
    color_continuous_scale=[
        [0.0, "white"],
        [0.25, "#fee5d9"],
        [0.5, "#fcae91"],
        [0.75, "#fb6a4a"],
        [1.0, "#99000d"]
    ],
    range_color=[
        0,
        map_latest["fatalities_per_100k"].quantile(0.95)
    ],
    title="Current Conflict Intensity"
)

fig_observed.update_geos(
    showframe=False,
    showcoastlines=True,
    showland=True,
    landcolor="lightgray",
    showcountries=True,
    countrycolor="white"
)

fig_observed.update_layout(
    geo=dict(
        projection_type="natural earth"
    ),
    coloraxis_colorbar=dict(
        title="Fatalities<br>per 100k"
    )
)

fig_observed.write_html(
    "global_conflict_observed_map.html"
)

print("Saved: global_conflict_observed_map.html")

print("\nGLOBAL CONFLICT MAPS COMPLETED")

# =========================================================
# MAP 4 — CONSERVATIVE RISK ALERT
# THRESHOLD 0.50
# =========================================================

print("\nBUILDING CONSERVATIVE RISK MAP")

map_conservative = map_plot.copy()

map_conservative["conservative_alert"] = (
    map_conservative["predicted_risk"] >= 0.50
).astype(int)

fig_conservative = px.choropleth(
    map_conservative,
    locations="country_plot",
    locationmode="country names",
    color="conservative_alert",
    hover_name="country",
    hover_data={
        "conservative_alert": True,
        "predicted_risk": ":.3f",
        "region": True,
        "month": True
    },
    color_continuous_scale=[
        [0.0, "white"],
        [1.0, "#99000d"]
    ],
    range_color=[0, 1],
    title="Conservative Conflict Risk Alert — Threshold 0.50"
)

fig_conservative.update_geos(
    showframe=False,
    showcoastlines=True,
    showland=True,
    landcolor="lightgray",
    showcountries=True,
    countrycolor="white"
)

fig_conservative.update_layout(
    geo=dict(
        projection_type="natural earth"
    ),
    coloraxis_colorbar=dict(
        title="Conservative<br>Alert"
    )
)

fig_conservative.write_html(
    "global_conflict_conservative_risk_map.html"
)

print(
    "Saved: global_conflict_conservative_risk_map.html"
)


# =========================================================
# MAP 5 — EARLY-WARNING RISK ALERT
# THRESHOLD 0.20
# =========================================================

print("\nBUILDING EARLY-WARNING RISK MAP")

map_early = map_plot.copy()

map_early["early_warning_alert"] = (
    map_early["predicted_risk"] >= 0.20
).astype(int)

fig_early_warning = px.choropleth(
    map_early,
    locations="country_plot",
    locationmode="country names",
    color="early_warning_alert",
    hover_name="country",
    hover_data={
        "early_warning_alert": True,
        "predicted_risk": ":.3f",
        "region": True,
        "month": True
    },
    color_continuous_scale=[
        [0.0, "white"],
        [1.0, "#99000d"]
    ],
    range_color=[0, 1],
    title="Early-Warning Conflict Risk Alert — Threshold 0.20"
)

fig_early_warning.update_geos(
    showframe=False,
    showcoastlines=True,
    showland=True,
    landcolor="lightgray",
    showcountries=True,
    countrycolor="white"
)

fig_early_warning.update_layout(
    geo=dict(
        projection_type="natural earth"
    ),
    coloraxis_colorbar=dict(
        title="Early-Warning<br>Alert"
    )
)

fig_early_warning.write_html(
    "global_conflict_early_warning_risk_map.html"
)

print(
    "Saved: global_conflict_early_warning_risk_map.html"
)

# =========================================================
# MAP — 6-MONTH EARLY-WARNING CONFLICT RISK
# =========================================================

print("\nBUILDING 6-MONTH EARLY-WARNING RISK MAP")

map_6m_early = map_plot.copy()

print(
    "\nUKRAINE / ISRAEL / IRAN RISK CHECK"
)

print(
    map_6m_early[
        map_6m_early["country"].isin(
            ["Ukraine", "Israel", "Iran"]
        )
    ][
        [
            "country",
            "predicted_risk",
            "month"
        ]
    ].to_string(index=False)
)

print("\nPREDICTED RISK CHECK")

print(
    map_plot[
        map_plot["country"].isin(
            ["Ukraine", "Israel", "Iran"]
        )
    ][
        [
            "country",
            "predicted_risk",
            "fatalities_per_100k",
            "month"
        ]
    ].to_string(index=False)
)

fig_6m_early = px.choropleth(
    map_6m_early,
    locations="country_plot",
    locationmode="country names",
    color="predicted_risk",
    hover_name="country",
    hover_data={
        "predicted_risk": ":.3f",
        "fatalities_per_100k": ":.3f",
        "region": True,
        "month": True
    },
    color_continuous_scale=[
        [0.00, "white"],
        [0.20, "#fdae6b"],
        [0.50, "#fb6a4a"],
        [1.00, "#99000d"]
    ],
    range_color=[0, 1],
    title="Predicted Major Escalation Risk — Next 6 Months"
)

fig_6m_early.update_geos(
    showframe=False,
    showcoastlines=True,
    showland=True,
    landcolor="lightgray",
    showcountries=True,
    countrycolor="white"
)

fig_6m_early.update_layout(
    geo=dict(
        projection_type="natural earth"
    ),
    coloraxis_colorbar=dict(
        title="Predicted<br>Risk",
        tickformat=".0%"
    )
)

fig_6m_early.write_html(
    "global_conflict_6m_early_warning_map.html"
)

import plotly.io as pio

six_month_early_map_html = pio.to_html(
    fig_6m_early,
    full_html=False,
    include_plotlyjs=False
)

print(
    "Saved: global_conflict_6m_early_warning_map.html"
)

# =========================================================
# FINAL INTERACTIVE DASHBOARD
# =========================================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import html

print("\n")
print("=" * 60)
print("BUILDING FINAL INTERACTIVE DASHBOARD")
print("=" * 60)


# ---------------------------------------------------------
# 1. MODEL METRICS
# ---------------------------------------------------------

roc_lr = roc_auc_score(
    y_test,
    logistic_prob
)

roc_gb = roc_auc_score(
    y_test,
    gb_prob
)

pr_lr = average_precision_score(
    y_test,
    logistic_prob
)

pr_gb = average_precision_score(
    y_test,
    gb_prob
)


# ---------------------------------------------------------
# 2. ROC CURVE
# ---------------------------------------------------------

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    logistic_prob
)

fpr_gb, tpr_gb, _ = roc_curve(
    y_test,
    gb_prob
)

roc_fig = go.Figure()

roc_fig.add_trace(
    go.Scatter(
        x=fpr_lr,
        y=tpr_lr,
        mode="lines",
        name=f"Logistic Regression — AUC {roc_lr:.3f}"
    )
)

roc_fig.add_trace(
    go.Scatter(
        x=fpr_gb,
        y=tpr_gb,
        mode="lines",
        name=f"Gradient Boosting — AUC {roc_gb:.3f}"
    )
)

roc_fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random classifier",
        line=dict(dash="dash")
    )
)

roc_fig.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template="plotly_white",
    height=500
)


# ---------------------------------------------------------
# 3. PRECISION–RECALL CURVE
# ---------------------------------------------------------

precision_lr, recall_lr, _ = precision_recall_curve(
    y_test,
    logistic_prob
)

precision_gb, recall_gb, _ = precision_recall_curve(
    y_test,
    gb_prob
)

baseline = y_test.mean()

pr_fig = go.Figure()

pr_fig.add_trace(
    go.Scatter(
        x=recall_lr,
        y=precision_lr,
        mode="lines",
        name=f"Logistic Regression — AP {pr_lr:.3f}"
    )
)

pr_fig.add_trace(
    go.Scatter(
        x=recall_gb,
        y=precision_gb,
        mode="lines",
        name=f"Gradient Boosting — AP {pr_gb:.3f}"
    )
)

pr_fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode="lines",
        name=f"Baseline — {baseline:.3f}",
        line=dict(dash="dash")
    )
)

pr_fig.update_layout(
    title="Precision–Recall Curve",
    xaxis_title="Recall",
    yaxis_title="Precision",
    template="plotly_white",
    height=500
)


# ---------------------------------------------------------
# 4. PERMUTATION IMPORTANCE
# ---------------------------------------------------------

top_features = (
    importance_df
    .sort_values(
        "importance_mean",
        ascending=True
    )
    .tail(15)
)

importance_fig = go.Figure()

importance_fig.add_trace(
    go.Bar(
        x=top_features["importance_mean"],
        y=top_features["feature"],
        orientation="h",
        error_x=dict(
            type="data",
            array=top_features["importance_std"]
        )
    )
)

importance_fig.update_layout(
    title="Gradient Boosting — Predictive Features",
    xaxis_title="Decrease in Average Precision",
    yaxis_title="",
    template="plotly_white",
    height=600
)


# ---------------------------------------------------------
# 5. TOP 20 COUNTRY RISK
# ---------------------------------------------------------

top20 = (
    map_latest
    .sort_values(
        "predicted_risk",
        ascending=False
    )
    .head(20)
    .sort_values(
        "predicted_risk",
        ascending=True
    )
)

country_fig = go.Figure()

country_fig.add_trace(
    go.Bar(
        x=top20["predicted_risk"],
        y=top20["country"],
        orientation="h",
        text=top20["predicted_risk"].round(3),
        textposition="outside"
    )
)

country_fig.update_layout(
    title="Top 20 Countries by Predicted Risk",
    xaxis_title="Predicted Probability of Major Escalation",
    yaxis_title="",
    template="plotly_white",
    height=650
)


# ---------------------------------------------------------
# 6. EARLY WARNING / CUMULATIVE RECALL
# ---------------------------------------------------------

ranking = pd.DataFrame({
    "actual": y_test.values,
    "probability": gb_prob
})

ranking = (
    ranking
    .sort_values(
        "probability",
        ascending=False
    )
    .reset_index(drop=True)
)

ranking["cumulative_events"] = (
    ranking["actual"].cumsum()
)

total_events = ranking["actual"].sum()

ranking["cumulative_recall"] = (
    ranking["cumulative_events"] /
    total_events
)

ranking["population_percentage"] = (
    np.arange(1, len(ranking) + 1) /
    len(ranking)
)

early_fig = go.Figure()

early_fig.add_trace(
    go.Scatter(
        x=ranking["population_percentage"] * 100,
        y=ranking["cumulative_recall"] * 100,
        mode="lines",
        name="Cumulative recall"
    )
)

# Reference points
early_percentages = [1, 5, 10, 20, 30]

early_values = []

for pct in early_percentages:

    n = max(
        1,
        int(len(ranking) * pct / 100)
    )

    recall = (
        ranking.iloc[n - 1]["cumulative_recall"] *
        100
    )

    early_values.append(recall)

early_fig.add_trace(
    go.Scatter(
        x=early_percentages,
        y=early_values,
        mode="markers+text",
        text=[
            f"{x:.1f}%"
            for x in early_values
        ],
        textposition="top center",
        name="Key thresholds"
    )
)

early_fig.update_layout(
    title="Early-Warning Performance",
    xaxis_title="Top % of observations ranked by predicted risk",
    yaxis_title="Cumulative Recall of Major Escalations (%)",
    template="plotly_white",
    height=500
)


# ---------------------------------------------------------
# 7. CONVERT EXISTING MAPS TO HTML
# ---------------------------------------------------------

risk_map_html = fig_risk.to_html(
    full_html=False,
    include_plotlyjs=False
)

observed_map_html = fig_observed.to_html(
    full_html=False,
    include_plotlyjs=False
)


# ---------------------------------------------------------
# 8. CONVERT CHARTS TO HTML
# ---------------------------------------------------------

roc_html = roc_fig.to_html(
    full_html=False,
    include_plotlyjs=False
)

pr_html = pr_fig.to_html(
    full_html=False,
    include_plotlyjs=False
)

importance_html = importance_fig.to_html(
    full_html=False,
    include_plotlyjs=False
)

country_html = country_fig.to_html(
    full_html=False,
    include_plotlyjs=False
)

early_html = early_fig.to_html(
    full_html=False,
    include_plotlyjs=False
)


# ---------------------------------------------------------
# 9. EARLY WARNING NUMBERS
# ---------------------------------------------------------

early_results = []

for pct in [1, 5, 10, 20, 30]:

    n = max(
        1,
        int(len(ranking) * pct / 100)
    )

    events = ranking.head(n)["actual"].sum()

    precision = events / n

    recall = (
        events / total_events
        if total_events > 0
        else 0
    )

    early_results.append(
        (pct, n, int(events), precision, recall)
    )

# ---------------------------------------------------------
# 10. CONSERVATIVE vs EARLY-WARNING THRESHOLDS
# ---------------------------------------------------------

threshold_comparison = []

for threshold, label in [
    (0.50, "Conservative"),
    (0.20, "Early Warning")
]:

    predictions = (
        gb_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    threshold_comparison.append({
        "Mode": label,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Alerts": predictions.sum()
    })


threshold_comparison_df = pd.DataFrame(
    threshold_comparison
)


# Interactive comparison chart

threshold_fig = go.Figure()

threshold_fig.add_trace(
    go.Bar(
        name="Precision",
        x=threshold_comparison_df["Mode"],
        y=threshold_comparison_df["Precision"]
    )
)

threshold_fig.add_trace(
    go.Bar(
        name="Recall",
        x=threshold_comparison_df["Mode"],
        y=threshold_comparison_df["Recall"]
    )
)

threshold_fig.add_trace(
    go.Bar(
        name="F1",
        x=threshold_comparison_df["Mode"],
        y=threshold_comparison_df["F1"]
    )
)

threshold_fig.update_layout(
    title="Conservative vs Early-Warning Threshold",
    xaxis_title="Operational Mode",
    yaxis_title="Score",
    barmode="group",
    yaxis=dict(
        range=[0, 0.75]
    ),
    template="plotly_white",
    height=500
)


threshold_html = threshold_fig.to_html(
    full_html=False,
    include_plotlyjs=False
)

# =========================================================
# FIVE-YEAR INTERACTIVE CONFLICT RISK PROJECTION
# =========================================================

import plotly.express as px
import pandas as pd
import numpy as np

print("\n")
print("=" * 60)
print("BUILDING FIVE-YEAR CONFLICT RISK PROJECTION")
print("=" * 60)


# ---------------------------------------------------------
# 1. LOAD MODEL FEATURE DATASETS
# ---------------------------------------------------------

# Engineered temporal/conflict features
feature_data = pd.read_csv(
    "data/processed/country_month_features.csv"
)

feature_data["month"] = pd.to_datetime(
    feature_data["month"]
)


# External socioeconomic/demographic features
external_data = pd.read_csv(
    "data/processed/country_month_panel_external_features.csv"
)

external_data["month"] = pd.to_datetime(
    external_data["month"]
)


# ---------------------------------------------------------
# 2. MERGE THE TWO FEATURE SOURCES
# ---------------------------------------------------------

projection_data = feature_data.merge(
    external_data[
        [
            "country",
            "month",
            "fatalities_per_100k",
            "log_gdp_per_capita",
            "log_population",
            "population_lag_12",
            "gdp_per_capita_lag_12",
            "gdp_growth_lag_12"
        ]
    ],
    on=["country", "month"],
    how="left"
)


# ---------------------------------------------------------
# 3. USE LATEST AVAILABLE OBSERVATION
# ---------------------------------------------------------

latest_projection = (
    projection_data
    .sort_values("month")
    .groupby("country", as_index=False)
    .tail(1)
    .copy()
)


print(
    "\nCountries available for projection:",
    len(latest_projection)
)


# Check that every model feature is available
missing_projection_features = [
    col
    for col in features
    if col not in latest_projection.columns
]

if missing_projection_features:

    raise ValueError(
        "Missing projection features: "
        + str(missing_projection_features)
    )


# ---------------------------------------------------------
# 4. MODEL FEATURES
# ---------------------------------------------------------

X_projection = latest_projection[
    features
].copy()


# ---------------------------------------------------------
# 5. APPLY SAME IMPUTER USED BY THE MODEL
# ---------------------------------------------------------

X_projection_imp = pd.DataFrame(
    imputer.transform(X_projection),
    columns=features,
    index=latest_projection.index
)


# ---------------------------------------------------------
# 6. CURRENT SIX-MONTH RISK
# ---------------------------------------------------------

current_risk = gb.predict_proba(
    X_projection_imp
)[:, 1]


latest_projection["six_month_risk"] = (
    current_risk
)

# ---------------------------------------------------------
# 5. EARLY-WARNING ALERT
# ---------------------------------------------------------

latest_projection["early_warning"] = (
    latest_projection["six_month_risk"] >= 0.20
).astype(int)




# ---------------------------------------------------------
# 6. BUILD FIVE-YEAR BASELINE SCENARIO
# ---------------------------------------------------------
#
# The model estimates the probability of a major
# escalation within the next six months.
#
# Future explanatory variables are unknown.
# Therefore this projection keeps the latest observed
# conditions constant as a baseline scenario.
#
# Five years = 10 consecutive six-month periods.
# ---------------------------------------------------------

projection_results = []


start_year = 2026

years = list(
    range(
        start_year,
        start_year + 5
    )
)


for year in years:

    year_data = latest_projection.copy()

    year_data["projection_year"] = year

    # Number of six-month periods elapsed
    periods = (
        (year - start_year + 1) * 2
    )

    # Probability of at least one escalation
    # over the cumulative six-month periods
    year_data["cumulative_risk"] = (
        1 -
        (
            1 -
            year_data["six_month_risk"]
        ) ** periods
    )

    projection_results.append(
        year_data[
            [
                "country",
                "country_id",
                "region",
                "projection_year",
                "six_month_risk",
                "cumulative_risk",
                "early_warning"
            ]
        ]
    )


projection_df = pd.concat(
    projection_results,
    ignore_index=True
)


# ---------------------------------------------------------
# 7. STANDARDIZE COUNTRY NAMES
# ---------------------------------------------------------

projection_df["country_plot"] = (
    projection_df["country"]
    .replace({
        "Yemen (North Yemen)": "Yemen",
        "Russia (Soviet Union)": "Russia",
        "Myanmar (Burma)": "Myanmar",
        "DR Congo (Zaire)":
            "Democratic Republic of the Congo"
    })
)


# ---------------------------------------------------------
# 8. INTERACTIVE FIVE-YEAR MAP
# ---------------------------------------------------------

fig_projection = px.choropleth(
    projection_df,

    locations="country_plot",

    locationmode="country names",

    color="cumulative_risk",

    animation_frame="projection_year",

    hover_name="country",

    hover_data={
        "cumulative_risk": ":.1%",
        "six_month_risk": ":.1%",
        "early_warning": True,
        "region": True,
        "projection_year": True,
        "country_plot": False
    },

    color_continuous_scale=[
        [0.00, "white"],
        [0.20, "#fee5d9"],
        [0.40, "#fcae91"],
        [0.60, "#fb6a4a"],
        [0.80, "#de2d26"],
        [1.00, "#99000d"]
    ],

    range_color=[0, 1],

    title=(
        "Projected Conflict Escalation Risk — "
        "2026–2030"
    )
)


fig_projection.update_geos(
    showframe=False,
    showcoastlines=True,
    showland=True,
    landcolor="lightgray",
    showcountries=True,
    countrycolor="white"
)


fig_projection.update_layout(
    geo=dict(
        projection_type="natural earth"
    ),

    coloraxis_colorbar=dict(
        title="Cumulative<br>Risk",
        tickformat=".0%"
    ),

    height=700
)


# ---------------------------------------------------------
# 9. SAVE INTERACTIVE MAP
# ---------------------------------------------------------

fig_projection.write_html(
    "global_conflict_five_year_projection.html"
)

five_year_projection_map_html = pio.to_html(
    fig_projection,
    full_html=False,
    include_plotlyjs=False
)

print(
    "\nSaved:"
    "\nglobal_conflict_five_year_projection.html"
)

# ---------------------------------------------------------
# 10. TOP COUNTRIES — FINAL YEAR
# ---------------------------------------------------------

final_year = (
    projection_df[
        projection_df["projection_year"] == 2030
    ]
    .sort_values(
        "cumulative_risk",
        ascending=False
    )
    .head(20)
)


print(
    "\nTOP 20 PROJECTED RISK — 2030\n"
)

print(
    final_year[
        [
            "country",
            "cumulative_risk",
            "six_month_risk",
            "early_warning"
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 60)
print("FIVE-YEAR PROJECTION COMPLETED")
print("=" * 60)

# =========================================================
# DASHBOARD — ADDITIONAL MODEL DIAGNOSTICS
# =========================================================

# ---------------------------------------------------------
# 1. CALIBRATION
# ---------------------------------------------------------

calibration_html = calibration_df.to_html(
    index=False,
    classes="dashboard-table",
    float_format=lambda x: f"{x:.3f}"
)


# ---------------------------------------------------------
# 2. TEMPORAL STABILITY
# ---------------------------------------------------------

temporal_plot = go.Figure()

temporal_plot.add_trace(
    go.Scatter(
        x=yearly_df["Year"],
        y=yearly_df["ROC-AUC"],
        mode="lines+markers",
        name="ROC-AUC"
    )
)

temporal_plot.add_trace(
    go.Scatter(
        x=yearly_df["Year"],
        y=yearly_df["PR-AUC"],
        mode="lines+markers",
        name="PR-AUC"
    )
)

temporal_plot.update_layout(
    title="Out-of-Sample Performance by Year",
    xaxis_title="Test Year",
    yaxis_title="Performance",
    template="plotly_white",
    height=450
)

temporal_html = (
    temporal_plot
    .to_html(
        full_html=False,
        include_plotlyjs=False
    )
)


# ---------------------------------------------------------
# 3. ABLATION STUDY
# ---------------------------------------------------------

ablation_display = ablation_results.copy()

ablation_display[
    "ROC-AUC"
] = ablation_display[
    "ROC-AUC"
].map(
    lambda x: f"{x:.3f}"
)

ablation_display[
    "PR-AUC"
] = ablation_display[
    "PR-AUC"
].map(
    lambda x: f"{x:.3f}"
)

ablation_display[
    "Brier"
] = ablation_display[
    "Brier"
].map(
    lambda x: f"{x:.4f}"
)

ablation_html = ablation_display.to_html(
    index=False,
    classes="dashboard-table"
)


# ---------------------------------------------------------
# 4. TOP FULL MODEL FEATURES
# ---------------------------------------------------------

top_features_display = (
    full_importance
    .head(10)
    .copy()
)

top_features_display[
    "importance_mean"
] = top_features_display[
    "importance_mean"
].map(
    lambda x: f"{x:.4f}"
)

top_features_display[
    "importance_std"
] = top_features_display[
    "importance_std"
].map(
    lambda x: f"{x:.4f}"
)

top_features_html = top_features_display.to_html(
    index=False,
    classes="dashboard-table"
)


# ---------------------------------------------------------
# 5. CALIBRATION METRICS
# ---------------------------------------------------------

calibration_lift = (
    pr_auc / baseline_pr
)

calibration_brier = brier


# ---------------------------------------------------------
# 6. TEMPORAL SUMMARY
# ---------------------------------------------------------

mean_yearly_roc = yearly_df[
    "ROC-AUC"
].mean()

min_yearly_roc = yearly_df[
    "ROC-AUC"
].min()

max_yearly_roc = yearly_df[
    "ROC-AUC"
].max()

mean_yearly_pr = yearly_df[
    "PR-AUC"
].mean()


# ---------------------------------------------------------
# 7. EXTERNAL FEATURE CONTRIBUTION
# ---------------------------------------------------------

external_roc_gain = (
    full_result["ROC-AUC"]
    -
    conflict_result["ROC-AUC"]
)

external_pr_gain = (
    full_result["PR-AUC"]
    -
    conflict_result["PR-AUC"]
)

# ---------------------------------------------------------
# CALIBRATION CURVE FOR DASHBOARD
# ---------------------------------------------------------

calibration_fig = go.Figure()

calibration_fig.add_trace(
    go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode="lines+markers",
        name="Gradient Boosting"
    )
)

calibration_fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(
            dash="dash"
        )
    )
)

calibration_fig.update_layout(
    title="Calibration Curve",
    xaxis_title="Mean predicted probability",
    yaxis_title="Observed event frequency",
    template="plotly_white",
    height=450
)

calibration_curve_html = (
    calibration_fig
    .to_html(
        full_html=False,
        include_plotlyjs=False
    )
)

# ---------------------------------------------------------
# 11. DASHBOARD HTML
# ---------------------------------------------------------

dashboard_html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Global Conflict Early-Warning System
</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>

body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f5f6f8;
    color: #1f2937;
}}

.header {{
    background: #111827;
    color: white;
    padding: 45px 8%;
}}

.header h1 {{
    margin: 0;
    font-size: 38px;
}}

.header p {{
    margin-top: 12px;
    font-size: 17px;
    color: #d1d5db;
}}

.container {{
    width: 84%;
    margin: 30px auto;
}}

.section-title {{
    margin-top: 45px;
    margin-bottom: 18px;
    font-size: 25px;
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(200px, 1fr));
    gap: 18px;
}}

.card {{
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}}

.card .label {{
    font-size: 14px;
    color: #6b7280;
}}

.card .value {{
    margin-top: 8px;
    font-size: 32px;
    font-weight: bold;
}}

.card .description {{
    margin-top: 6px;
    font-size: 13px;
    color: #6b7280;
}}

.chart {{
    background: white;
    margin-bottom: 25px;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}}

.two-columns {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(450px, 1fr));
    gap: 25px;
}}

.map {{
    background: white;
    margin-bottom: 30px;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
}}

th, td {{
    padding: 12px;
    text-align: center;
    border-bottom: 1px solid #e5e7eb;
}}

th {{
    background: #111827;
    color: white;
}}

.footer {{
    margin-top: 60px;
    padding: 30px;
    text-align: center;
    color: #6b7280;
    font-size: 13px;
}}

</style>

</head>


<body>


<div class="header">

<h1>
Global Conflict Early-Warning System
</h1>

<p>
Machine Learning for Conflict Escalation Risk Assessment
</p>

<p>
Out-of-sample validation • Gradient Boosting • Early-Warning Analytics • Geospatial Risk Mapping</p>

</div>


<div class="container">


<!-- ================================================= -->
<!-- MODEL PERFORMANCE -->
<!-- ================================================= -->

<h2 class="section-title">
Model Performance
</h2>

<p style="color: #6b7280; max-width: 900px;">
The system estimates the probability that a country-month
observation will experience a major conflict escalation within
the following six months. Model performance is evaluated on an
out-of-sample test period covering 2020-2025, while the model is
trained exclusively on observations from 1989-2019.
</p>

<div class="cards">


<div class="card">

<div class="label">
Gradient Boosting ROC-AUC
</div>

<div class="value">
{roc_gb:.3f}
</div>

<div class="description">
Out-of-sample discrimination
</div>

</div>


<div class="card">

<div class="label">
Gradient Boosting PR-AUC
</div>

<div class="value">
{pr_gb:.3f}
</div>

<div class="description">
Performance under class imbalance
</div>

</div>


<div class="card">

<div class="label">
Logistic Regression ROC-AUC
</div>

<div class="value">
{roc_lr:.3f}
</div>

<div class="description">
Benchmark model
</div>

</div>


<div class="card">

<div class="label">
Major Escalations
</div>

<div class="card">

<div class="label">
Brier Score
</div>

<div class="value">
{calibration_brier:.3f}
</div>

<div class="description">
Probability calibration
</div>

</div>


<div class="card">

<div class="label">
PR-AUC Lift
</div>

<div class="value">
{calibration_lift:.2f}×
</div>

<div class="description">
Improvement over prevalence baseline
</div>

</div>

<div class="value">
{int(total_events):,}
</div>

<div class="description">
Observed in the test set
</div>

</div>


</div>


<!-- ================================================= -->
<!-- ROC + PR -->
<!-- ================================================= -->

<h2 class="section-title">
Model Validation
</h2>

<p style="color: #6b7280; max-width: 900px;">
Performance is assessed using ROC-AUC and Precision-Recall AUC.
ROC-AUC measures the model's ability to discriminate between
country-month observations that subsequently experience a major
escalation and those that do not. PR-AUC is particularly relevant
because major escalation events are relatively rare in the dataset.
The logistic regression model provides a simpler benchmark against
which the Gradient Boosting model is compared.
</p>


<div class="two-columns">

<div class="chart">
{roc_html}
</div>

<div class="chart">
{pr_html}
</div>

</div>

<!-- ================================================= -->
<!-- MODEL CALIBRATION -->
<!-- ================================================= -->

<h2 class="section-title">
Model Calibration
</h2>

<p style="color: #6b7280; max-width: 900px;">
Calibration evaluates whether the probabilities produced by the
model correspond to the observed frequency of major escalation
events. A well-calibrated model should produce probabilities that
approximately match the empirical frequency of events.
</p>

<div class="cards">

<div class="card">

<div class="label">
Brier Score
</div>

<div class="value">
{calibration_brier:.3f}
</div>

<div class="description">
Lower values indicate better probabilistic accuracy
</div>

</div>

<div class="card">

<div class="label">
Baseline PR-AUC
</div>

<div class="value">
{baseline_pr:.3f}
</div>

<div class="description">
Observed event prevalence
</div>

</div>

<div class="card">

<div class="label">
PR-AUC Lift
</div>

<div class="value">
{calibration_lift:.2f}×
</div>

<div class="description">
Model performance relative to baseline
</div>

</div>

</div>

<div class="chart">

{calibration_curve_html}

</div>

<h3>
Calibration Table
</h3>

<div class="chart">

{calibration_html}

</div>


<!-- ================================================= -->
<!-- TEMPORAL STABILITY -->
<!-- ================================================= -->

<h2 class="section-title">
Temporal Stability
</h2>

<p style="color: #6b7280; max-width: 900px;">
The model is evaluated separately across each year of the
out-of-sample period to assess whether predictive performance
remains reasonably stable over time. This is important because
relationships between conflict indicators and escalation may
change across geopolitical periods.
</p>

<div class="cards">

<div class="card">

<div class="label">
Mean Yearly ROC-AUC
</div>

<div class="value">
{mean_yearly_roc:.3f}
</div>

<div class="description">
Average out-of-sample discrimination
</div>

</div>

<div class="card">

<div class="label">
Minimum Yearly ROC-AUC
</div>

<div class="value">
{min_yearly_roc:.3f}
</div>

<div class="description">
Lowest annual performance
</div>

</div>

<div class="card">

<div class="label">
Maximum Yearly ROC-AUC
</div>

<div class="value">
{max_yearly_roc:.3f}
</div>

<div class="description">
Highest annual performance
</div>

</div>

<div class="card">

<div class="label">
Mean Yearly PR-AUC
</div>

<div class="value">
{mean_yearly_pr:.3f}
</div>

<div class="description">
Average precision-recall performance
</div>

</div>

</div>

<div class="chart">

{temporal_html}

</div>

<div class="chart">

{yearly_df.to_html(
    index=False,
    classes="dashboard-table",
    float_format=lambda x: f"{x:.3f}"
)}

</div>

<!-- ================================================= -->
<!-- ABLATION STUDY -->
<!-- ================================================= -->

<h2 class="section-title">
Incremental Value of External Features
</h2>

<p style="color: #6b7280; max-width: 900px;">
The ablation study evaluates whether socioeconomic and demographic
variables provide predictive information beyond the historical
conflict characteristics already contained in the model.
Three specifications are compared: conflict history only,
external features only, and the full model.
</p>

<div class="cards">

<div class="card">

<div class="label">
ROC-AUC Gain
</div>

<div class="value">
{external_roc_gain:+.3f}
</div>

<div class="description">
Full model vs conflict history only
</div>

</div>

<div class="card">

<div class="label">
PR-AUC Gain
</div>

<div class="value">
{external_pr_gain:+.3f}
</div>

<div class="description">
Full model vs conflict history only
</div>

</div>

</div>

<div class="chart">

{ablation_html}

</div>

<p style="color: #6b7280; max-width: 900px;">
The comparison should be interpreted as evidence of incremental
predictive value rather than causal importance. A positive
improvement indicates that the external variables contribute
additional predictive information when combined with conflict
history.
</p>

<!-- ================================================= -->
<!-- FEATURE IMPORTANCE -->
<!-- ================================================= -->

<h2 class="section-title">
What Drives the Predictions?
</h2>



<p style="color: #6b7280; max-width: 900px;">
Feature importance indicates which variables contribute most to
the model's predictions. These measures describe predictive
association rather than causal effects: a highly important
variable should not automatically be interpreted as a direct
cause of conflict escalation.
</p>


<div class="chart">

{importance_html}

</div>

<h3>
Top Features by Permutation Importance
</h3>

<div class="chart">

{top_features_html}

</div>


<!-- ================================================= -->
<!-- EARLY WARNING -->
<!-- ================================================= -->

<h2 class="section-title">
Early-Warning Performance
</h2>

<p style="color: #6b7280; max-width: 900px;">
The early-warning analysis evaluates how effectively the model
concentrates observed escalation events among the highest-risk
country-month observations. The table reports performance when
analysts focus only on the highest-risk 1%, 5%, 10%, 20%, and 30%
of observations.
</p>

<p style="color: #6b7280; max-width: 900px;">
This ranking perspective is particularly relevant for operational
early-warning systems, where analytical resources may be limited
and attention must be concentrated on a smaller set of high-risk
observations.
</p>

<div class="chart">

{early_html}

</div>


<table>

<tr>

<th>
Risk-ranked population
</th>

<th>
Observations
</th>

<th>
Observed events
</th>

<th>
Precision
</th>

<th>
Recall
</th>

</tr>
"""


for pct, n, events, precision, recall in early_results:

    dashboard_html += f"""

<tr>

<td>
Top {pct}%
</td>

<td>
{n:,}
</td>

<td>
{events:,}
</td>

<td>
{precision:.1%}
</td>

<td>
{recall:.1%}
</td>

</tr>

"""


dashboard_html += """

</table>

<!-- ================================================= -->
<!-- OPERATIONAL THRESHOLDS -->
<!-- ================================================= -->

<h2 class="section-title">
Operational Thresholds
</h2>

<div class="chart">

<h3 style="margin-left: 20px;">
Conservative vs Early-Warning Mode
</h3>

<p style="margin-left: 20px; margin-right: 20px; color: #6b7280;">
The model produces a continuous probability score. Operational
decisions require converting this score into an alert using a
threshold.

The conservative mode uses a threshold of 0.50. It generates
fewer alerts and therefore limits false positives, but it detects
only a small proportion of observed escalations.

The early-warning mode uses a threshold of 0.20. This lower
threshold deliberately prioritizes recall: more potential
escalations are flagged, at the cost of generating more false
alarms.

The appropriate threshold therefore depends on the operational
cost of missed escalations versus false warnings.
</p>

"""

dashboard_html += threshold_html

dashboard_html += """

</div>


<table>

<tr>

<th>
Operational Mode
</th>

<th>
Threshold
</th>

<th>
Precision
</th>

<th>
Recall
</th>

<th>
F1
</th>

<th>
True Positives
</th>

<th>
False Positives
</th>

<th>
False Negatives
</th>

<th>
Alerts
</th>

</tr>
"""
for _, row in threshold_comparison_df.iterrows():

    dashboard_html += f"""

<tr>

<td>
<strong>{row["Mode"]}</strong>
</td>

<td>
{row["Threshold"]:.2f}
</td>

<td>
{row["Precision"]:.1%}
</td>

<td>
{row["Recall"]:.1%}
</td>

<td>
{row["F1"]:.3f}
</td>

<td>
{int(row["True Positives"]):,}
</td>

<td>
{int(row["False Positives"]):,}
</td>

<td>
{int(row["False Negatives"]):,}
</td>

<td>
{int(row["Alerts"]):,}
</td>

</tr>

"""


dashboard_html += """

</table>


"""
dashboard_html += """

<!-- ================================================= -->
<!-- COUNTRY RISK -->
<!-- ================================================= -->

<h2 class="section-title">
Countries Ranked by Predicted Risk
</h2>

<p style="color: #6b7280; max-width: 900px;">
Countries are ranked according to the model's predicted
six-month escalation probability using their latest available
observation. A higher score indicates that the observed
characteristics of the country-month resemble patterns that the
model associates with subsequent major escalation.
</p>

<p style="color: #6b7280; max-width: 900px;">
The ranking should be interpreted as a prioritization tool rather
than as a deterministic forecast. A high predicted probability
does not imply that an escalation will necessarily occur.
</p>

<div class="chart">

"""


dashboard_html += country_html


dashboard_html += """

</div>


<!-- ================================================= -->
<!-- GLOBAL MAPS -->
<!-- ================================================= -->

<h2 class="section-title">
Global Conflict Risk
</h2>

<div class="map">

<p style="color: #6b7280; max-width: 900px;">
This map shows the model's predicted probability of major conflict escalationù
within the next six months using the latest available country-level observation in the dataset.
It represents a forward-looking model signal rather than a measure
of current conflict intensity.
</p>


"""

dashboard_html += risk_map_html

dashboard_html += """

</div>


<!-- ================================================= -->
<!-- 6-MONTH EARLY-WARNING MAP -->
<!-- ================================================= -->

<h2 class="section-title">
6-Month Early-Warning Risk
</h2>

<div class="map">

<p style="margin: 20px; color: #6b7280;">
The map shows the model's predicted probability of a major
conflict escalation within the next six months.

The 0.20 threshold is used as the operational early-warning
threshold. Country-month observations at or above this threshold are treated 
as potential early-warning alerts.

The map should not be interpreted as a map of current conflict
severity. A country can have high current conflict intensity but
low predicted escalation risk, or relatively low current intensity
but elevated predicted risk.
</p>

"""

dashboard_html += six_month_early_map_html

dashboard_html += """

</div>

<!-- ================================================= -->
<!-- FIVE-YEAR PROJECTION -->
<!-- ================================================= -->

<h2 class="section-title">
Five-Year Conflict Risk Projection
</h2>

<div class="map">

<p style="margin: 20px; color: #6b7280;">
This interactive map presents a five-year baseline scenario for
2026-2030.

The model's latest available country-level characteristics are
held constant throughout the projection horizon. The resulting
values therefore represent a scenario analysis of how the model's
estimated escalation risk would accumulate over repeated
six-month periods if current conditions remained unchanged.

The projection is not a direct forecast of geopolitical events
and should not be interpreted as a prediction that today's risk
conditions will actually persist for five years.

The cumulative risk represents the estimated probability of at
least one escalation occurring over the corresponding projection
horizon under this constant-conditions scenario.
</p>

"""

dashboard_html += five_year_projection_map_html

dashboard_html += """

</div>

<!-- ================================================= -->
<!-- OBSERVED MAJOR ESCALATION -->
<!-- ================================================= -->

<h2 class="section-title">
Observed Major Escalation
</h2>

<div class="map">

<p style="color: #6b7280; max-width: 900px;">
This map represents observed conflict intensity rather than
model-predicted escalation risk. Fatalities per 100,000 inhabitants
are used to provide a population-adjusted measure of conflict
intensity.

This measure answers a different question from the early-warning
map: it describes the intensity of conflict already observed in
the data rather than the probability of a future escalation.
</p>

"""

dashboard_html += observed_map_html

dashboard_html += """

</div>
"""


dashboard_html += f"""

<!-- ================================================= -->
<!-- METHODOLOGICAL NOTES -->
<!-- ================================================= -->

<h2 class="section-title">
Methodological Notes
</h2>

<div class="card">

<h3>Unit of analysis</h3>

<p>
The model operates at the <strong>country-month</strong> level.
Each observation represents one country in one calendar month.
The dataset combines conflict-related, socioeconomic and
demographic indicators to characterize the conditions observed
at that point in time.
</p>


<h3>Prediction target</h3>

<p>
The prediction target identifies whether a
<strong>major conflict escalation</strong> occurs within the
subsequent six-month period. The model therefore uses information
available at a country-month observation to estimate the
probability of a subsequent escalation.
</p>


<h3>Temporal validation</h3>

<p>
The model is evaluated using a chronological train-test split
rather than a random split. The training period covers
<strong>1989-2019</strong>, while the out-of-sample test period
covers <strong>2020-2025</strong>.
</p>

<p>
This temporal design prevents observations from the future test
period from being used to train the model and provides a more
realistic assessment of prospective predictive performance.
</p>


<h3>Models</h3>

<p>
The primary model is a
<strong>Gradient Boosting Classifier</strong>. A
<strong>Logistic Regression</strong> model is used as a benchmark
to determine whether the nonlinear Gradient Boosting approach
provides additional predictive value.
</p>


<h3>Evaluation metrics</h3>

<p>
Model discrimination is evaluated using
<strong>ROC-AUC</strong> and <strong>PR-AUC</strong>.
ROC-AUC measures how effectively the model ranks observations
that subsequently experience escalation above those that do not.
PR-AUC is particularly informative in this application because
major escalation events represent a minority of country-month
observations.
</p>


<h3>Operational thresholds</h3>

<p>
The model produces a continuous probability score. Two
operational thresholds are presented.
</p>

<p>
The <strong>Conservative</strong> mode uses a threshold of
<strong>0.50</strong>, producing fewer alerts and prioritizing
precision. The <strong>Early-Warning</strong> mode uses a
threshold of <strong>0.20</strong>, prioritizing the detection
of a larger proportion of potential escalations at the cost of
more false alarms.
</p>


<h3>Observed conflict intensity</h3>

<p>
Observed conflict intensity is measured using
<strong>fatalities per 100,000 inhabitants</strong>. This measure
is displayed separately from predicted escalation risk because
current conflict intensity and future escalation risk represent
different concepts.
</p>


<h3>Five-year scenario projection</h3>

<p>
The five-year projection extends the model across ten consecutive
six-month periods covering <strong>2026–2030</strong>.
Because future explanatory variables are unknown, the latest
available country-level conditions are held constant throughout
the baseline scenario.
</p>

<p>
Cumulative risk represents the estimated probability of
experiencing <strong>at least one escalation</strong> over the
corresponding sequence of six-month periods, assuming that the
six-month risk remains constant and the periods are treated as
conditionally independent.
</p>

<p>
The five-year projection is therefore a
<strong>scenario-based extrapolation</strong>, not a literal
forecast of future geopolitical, political, economic or military
conditions.
</p>


<h3>Limitations</h3>

<p>
The model is based on historical relationships between observed
country-month characteristics and subsequent conflict
escalations. These relationships may not remain stable over
time.
</p>

<p>
Future conflicts may involve political, military, economic,
technological or social conditions that are not adequately
represented in the historical training data.
</p>

<p>
The system should therefore be considered an
<strong>early-warning and analytical decision-support tool</strong>,
rather than an autonomous conflict prediction system.
Predicted probabilities should be interpreted as model-based
risk estimates and ranking signals, not deterministic predictions.
</p>

</div>


<!-- ================================================= -->
<!-- HOW TO READ THE RESULTS -->
<!-- ================================================= -->

<h2 class="section-title">
How to Read the Results
</h2>

<div class="card">

<h3>Model performance</h3>

<p>
The Gradient Boosting model achieves a ROC-AUC of
<strong>{roc_gb:.3f}</strong>, compared with
<strong>{roc_lr:.3f}</strong> for the Logistic Regression
benchmark. This indicates stronger out-of-sample discrimination
between country-month observations that subsequently experience
a major conflict escalation and those that do not.
</p>

<p>
The Gradient Boosting model also achieves a PR-AUC of
<strong>{pr_gb:.3f}</strong>. Because escalation events are
relatively rare, PR-AUC provides an important complementary
measure of model performance.
</p>


<h3>Risk ranking and early warning</h3>

<p>
The model assigns each country-month observation a predicted
probability of major conflict escalation within the subsequent
six months. These probabilities are primarily useful as a
<strong>risk-ranking signal</strong>: higher values indicate that
the observed characteristics more closely resemble historical
patterns associated with subsequent escalation.
</p>

<p>
The early-warning analysis evaluates how effectively observed
escalation events are concentrated within the highest-risk
observations. Expanding the alert population generally increases
recall while reducing precision.
</p>


<h3>Operational thresholds</h3>

<p>
The <strong>Conservative</strong> mode uses a 0.50 threshold and
generates fewer alerts. The <strong>Early-Warning</strong> mode
uses a 0.20 threshold and generates more alerts in order to
identify a larger proportion of potential escalations.
</p>

<p>
The two modes therefore represent different operational
trade-offs between <strong>false alarms</strong> and
<strong>missed escalations</strong>. The lower threshold is
appropriate when missing a potential escalation is considered
more costly than investigating additional alerts.
</p>


<h3>Global Conflict Risk</h3>

<p>
The <strong>Global Conflict Risk</strong> map represents the
model's predicted probability of a subsequent major escalation
using the latest available country-level observation.
</p>

<p>
This is a <strong>forward-looking model signal</strong>, not a
direct measure of the amount of violence currently occurring in
a country.
</p>


<h3>Observed Conflict Intensity</h3>

<p>
The <strong>Observed Major Escalation</strong> map represents
observed conflict intensity using fatalities per 100,000
inhabitants.
</p>

<p>
It therefore describes the level of conflict-related violence
recorded in the data rather than the model's assessment of future
escalation risk.
</p>

<p>
The observed and predicted maps should not be expected to look
the same. They answer different questions:
<strong>observed intensity describes what has been recorded,
while predicted risk describes what the model considers more
likely to escalate.</strong>
</p>


<h3>Six-month early-warning risk</h3>

<p>
The <strong>6-Month Early-Warning Risk</strong> map shows the
model's estimated probability of a major escalation occurring
within the following six months.
</p>

<p>
The operational threshold is set at <strong>0.20</strong>.
Country-month observations at or above this threshold are treated
as potential early-warning alerts.
</p>

<p>
The threshold is an operational choice rather than a universally
optimal or universally calibrated probability boundary.
Accordingly, the map should be interpreted as an alerting tool
rather than as a definitive classification of countries as
"safe" or "unsafe".
</p>


<h3>Five-year conflict risk projection</h3>

<p>
The <strong>Five-Year Conflict Risk Projection</strong> presents
a baseline scenario covering <strong>2026–2030</strong>.
</p>

<p>
Because future explanatory variables are unknown, the projection
holds the latest available country-level conditions constant.
It therefore answers a hypothetical question:
<strong>what would the model's cumulative escalation risk look
like if the current observed conditions were maintained?</strong>
</p>

<p>
The cumulative probability represents the estimated probability
of experiencing <strong>at least one escalation</strong> across
the sequence of six-month periods.
</p>

<p>
Cumulative risk naturally increases as the projection horizon
becomes longer. Therefore, a country becoming darker over the
five-year horizon does <strong>not necessarily mean that its
underlying six-month risk is increasing</strong>. It can simply
reflect the accumulation of risk over multiple periods.
</p>

<p>
The five-year map should consequently be interpreted as a
<strong>scenario analysis</strong>, not as a literal prediction
of geopolitical events between 2026 and 2030.
</p>


<h3>How the maps should be interpreted together</h3>

<p>
The four geographical views provide complementary information.
The <strong>Global Conflict Risk</strong> map shows the model's
predicted near-term escalation risk. The
<strong>Observed Conflict Intensity</strong> map shows recorded
conflict intensity. The <strong>6-Month Early-Warning Risk</strong>
map converts predicted risk into an operational alert using the
0.20 threshold. The <strong>Five-Year Projection</strong>
illustrates how estimated risk could accumulate over a longer
horizon under a constant-conditions scenario.
</p>

<p>
Differences between these maps are therefore expected and do not
necessarily indicate inconsistencies. A country experiencing
substantial current violence may receive relatively low
additional escalation risk, while a country with limited
currently observed violence may receive elevated predicted risk
if its characteristics resemble historical pre-escalation
patterns.
</p>


<h3>Final interpretation</h3>

<p>
The system is designed to help identify
<strong>countries and periods that warrant closer analytical
attention</strong>. It is not designed to determine whether a
conflict will occur with certainty.
</p>

<p>
Model outputs should therefore be interpreted alongside
qualitative geopolitical analysis, expert assessment and other
relevant information. The value of the system lies primarily in
<strong>systematic risk ranking, early identification of
potential escalation and structured monitoring</strong>.
</p>

</div>


<!-- ================================================= -->
<!-- FOOTER -->
<!-- ================================================= -->

<div class="footer">

Global Conflict Early-Warning System<br>
Machine Learning • Statistical Modelling • Geospatial Analytics<br>
Risk Assessment • Early Warning • Conflict Monitoring • Scenario Analysis

</div>


</div>

</body>

</html>
"""


# ---------------------------------------------------------
# 11. SAVE DASHBOARD
# ---------------------------------------------------------

with open(
    "global_conflict_dashboard.html",
    "w",
    encoding="utf-8"
) as f:

    f.write(dashboard_html)


print("\\n")
print("=" * 60)
print("DASHBOARD COMPLETED")
print("=" * 60)

print(
    "\\nSaved:"
    "\\nglobal_conflict_dashboard.html"
)

