import pandas as pd


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/country_month_panel_full.csv"
)

df["month"] = pd.to_datetime(df["month"])

df = df.sort_values(
    ["country", "month"]
)


# ---------------------------------------------------------
# LAGGED VARIABLES
# ---------------------------------------------------------

df["fatalities_lag1"] = (
    df.groupby("country")["fatalities"]
      .shift(1)
)

df["fatalities_lag3"] = (
    df.groupby("country")["fatalities"]
      .shift(3)
)

df["fatalities_lag6"] = (
    df.groupby("country")["fatalities"]
      .shift(6)
)


# ---------------------------------------------------------
# ROLLING BASELINE
# ---------------------------------------------------------

df["fatalities_mean_3m"] = (
    df.groupby("country")["fatalities"]
      .transform(
          lambda x: x.shift(1).rolling(3).mean()
      )
)

df["fatalities_mean_6m"] = (
    df.groupby("country")["fatalities"]
      .transform(
          lambda x: x.shift(1).rolling(6).mean()
      )
)

df["fatalities_mean_12m"] = (
    df.groupby("country")["fatalities"]
      .transform(
          lambda x: x.shift(1).rolling(12).mean()
      )
)


# ---------------------------------------------------------
# CHANGE FROM PREVIOUS MONTH
# ---------------------------------------------------------

df["fatality_change"] = (
    df["fatalities"]
    - df["fatalities_lag1"]
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\nFATALITY CHANGE DISTRIBUTION\n")

print(
    df["fatality_change"]
    .describe(
        percentiles=[
            .01,
            .05,
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99
        ]
    )
)


print("\n\nLARGEST MONTH-TO-MONTH INCREASES\n")

print(
    df.nlargest(
        30,
        "fatality_change"
    )[
        [
            "country",
            "month",
            "fatalities_lag1",
            "fatalities",
            "fatality_change"
        ]
    ]
    .to_string(index=False)
)