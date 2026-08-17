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
# EXACTLY THE NEXT 6 MONTHS
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
# ESCALATION
# ---------------------------------------------------------

df["escalation_6m"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="boolean"
)

valid = df["future_max_regime_6m"].notna()

# ---------------------------------------------------------
# ALTERNATIVE ESCALATION DEFINITIONS
# ---------------------------------------------------------

df["major_escalation_6m"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="boolean"
)

df["severe_escalation_6m"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="boolean"
)


df.loc[valid, "major_escalation_6m"] = (
    df.loc[valid, "future_max_regime_6m"]
    >= df.loc[valid, "regime"] + 2
)


df.loc[valid, "severe_escalation_6m"] = (
    df.loc[valid, "future_max_regime_6m"]
    >= 3
)

valid = df["future_max_regime_6m"].notna()

df.loc[valid, "escalation_6m"] = (
    df.loc[valid, "future_max_regime_6m"]
    > df.loc[valid, "regime"]
)


# ---------------------------------------------------------
# REGIME DISTRIBUTION
# ---------------------------------------------------------

print("\nVIOLENCE REGIME DISTRIBUTION\n")

print(
    df["regime"]
      .value_counts()
      .sort_index()
)


# ---------------------------------------------------------
# ESCALATION RESULTS
# ---------------------------------------------------------

print("\n\nESCALATION EVENTS\n")

print(
    df["escalation_6m"]
      .value_counts(dropna=False)
)


print("\n\nESCALATION RATE\n")

valid_target = df["escalation_6m"].notna()

print(
    df.loc[
        valid_target,
        "escalation_6m"
    ].mean()
)


# ---------------------------------------------------------
# TRANSITIONS
# ---------------------------------------------------------

print("\n\nCURRENT REGIME → FUTURE MAX REGIME\n")

transitions = (
    df.loc[
        valid_target,
        ["regime", "future_max_regime_6m"]
    ]
    .value_counts()
    .sort_index()
)

print(transitions)


# ---------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------

print("\n\nFIRST 30 ESCALATION EXAMPLES\n")

print(
    df.loc[
        df["escalation_6m"] == True,
        [
            "country",
            "month",
            "fatalities",
            "regime",
            "future_max_regime_6m"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

# ---------------------------------------------------------
# COMPARE ESCALATION DEFINITIONS
# ---------------------------------------------------------

print("\n\nESCALATION DEFINITION COMPARISON\n")

for name in [
    "escalation_6m",
    "major_escalation_6m",
    "severe_escalation_6m"
]:

    rate = df.loc[
        df[name].notna(),
        name
    ].mean()

    count = df[name].sum()

    print(
        f"{name}: "
        f"{int(count):,} events | "
        f"{rate:.2%}"
    )

# ---------------------------------------------------------
# MAJOR ESCALATION RATE BY CURRENT REGIME
# ---------------------------------------------------------

print("\n\nMAJOR ESCALATION RATE BY CURRENT REGIME\n")

major_by_regime = (
    df.loc[
        df["major_escalation_6m"].notna()
    ]
    .groupby("regime")["major_escalation_6m"]
    .agg(
        events="sum",
        observations="count",
        rate="mean"
    )
)

print(major_by_regime)   