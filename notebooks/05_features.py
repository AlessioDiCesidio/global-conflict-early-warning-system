import pandas as pd
import numpy as np


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/country_month_panel_full.csv"
)

df["month"] = pd.to_datetime(df["month"])

df = df.sort_values(
    ["country", "month"]
).reset_index(drop=True)


# ---------------------------------------------------------
# VIOLENCE REGIME
# ---------------------------------------------------------

def violence_regime(x):

    if x == 0:
        return 0
    elif x <= 56:
        return 1
    elif x <= 179:
        return 2
    elif x <= 1000:
        return 3
    else:
        return 4


df["regime"] = df["fatalities"].apply(
    violence_regime
)


# ---------------------------------------------------------
# FUTURE MAXIMUM REGIME
# NEXT 6 MONTHS
# ---------------------------------------------------------

def future_max_6m(x):

    return (
        x.shift(-1)
         .iloc[::-1]
         .rolling(6, min_periods=6)
         .max()
         .iloc[::-1]
    )


df["future_max_regime_6m"] = (
    df.groupby("country")["regime"]
      .transform(future_max_6m)
)


# ---------------------------------------------------------
# MAJOR ESCALATION TARGET
# ---------------------------------------------------------

df["major_escalation_6m"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="boolean"
)

valid = df["future_max_regime_6m"].notna()

df.loc[valid, "major_escalation_6m"] = (
    df.loc[valid, "future_max_regime_6m"]
    >= df.loc[valid, "regime"] + 2
)


# ---------------------------------------------------------
# LAG FEATURES
# ---------------------------------------------------------

group = df.groupby("country")

df["fatalities_lag_1"] = (
    group["fatalities"].shift(1)
)

df["fatalities_lag_3"] = (
    group["fatalities"].shift(3)
)

df["fatalities_lag_6"] = (
    group["fatalities"].shift(6)
)


df["regime_lag_1"] = (
    group["regime"].shift(1)
)

df["regime_lag_3"] = (
    group["regime"].shift(3)
)

df["regime_lag_6"] = (
    group["regime"].shift(6)
)


# ---------------------------------------------------------
# ROLLING FATALITIES
# ONLY PAST INFORMATION
# ---------------------------------------------------------

df["fatalities_mean_3m"] = (
    group["fatalities"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

df["fatalities_mean_6m"] = (
    group["fatalities"]
    .transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
)

df["fatalities_mean_12m"] = (
    group["fatalities"]
    .transform(
        lambda x: x.shift(1).rolling(12).mean()
    )
)


# ---------------------------------------------------------
# VOLATILITY
# ---------------------------------------------------------

df["fatalities_std_6m"] = (
    group["fatalities"]
    .transform(
        lambda x: x.shift(1).rolling(6).std()
    )
)

df["fatalities_std_12m"] = (
    group["fatalities"]
    .transform(
        lambda x: x.shift(1).rolling(12).std()
    )
)


# ---------------------------------------------------------
# RECENT REGIME CHANGE
# ---------------------------------------------------------

df["regime_change_1m"] = (
    df["regime"] -
    df["regime_lag_1"]
)

df["regime_change_3m"] = (
    df["regime"] -
    df["regime_lag_3"]
)


# ---------------------------------------------------------
# MONTHS ACTIVE IN PREVIOUS 12 MONTHS
# ---------------------------------------------------------

df["months_active_12m"] = (
    group["fatalities"]
    .transform(
        lambda x:
        (x.shift(1) > 0)
        .rolling(12)
        .sum()
    )
)


# ---------------------------------------------------------
# FEATURE SUMMARY
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

print("\nFEATURE SUMMARY\n")

print(
    df[features].describe().T
)


# ---------------------------------------------------------
# TARGET SUMMARY
# ---------------------------------------------------------

print("\n\nMAJOR ESCALATION TARGET\n")

print(
    df["major_escalation_6m"]
    .value_counts(dropna=False)
)

print("\n\nMAJOR ESCALATION RATE\n")

print(
    df.loc[
        df["major_escalation_6m"].notna(),
        "major_escalation_6m"
    ].mean()
)

# ---------------------------------------------------------
# TEMPORAL CONSISTENCY CHECK
# ---------------------------------------------------------

print("\n\nTEMPORAL CONSISTENCY CHECK\n")

df["month_diff"] = (
    df.groupby("country")["month"]
      .diff()
      .dt.days
)

print("\nMonth gap distribution:\n")

print(
    df["month_diff"]
      .value_counts()
      .sort_index()
      .head(20)
)


# Countries with gaps larger than expected
gaps = df.loc[
    df["month_diff"] > 31,
    ["country", "month", "month_diff"]
]

print(
    f"\nNumber of observations with gaps > 31 days: "
    f"{len(gaps):,}"
)

print("\nExamples:\n")

print(
    gaps.head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_path = (
    "data/processed/"
    "country_month_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved features to: {output_path}"
)