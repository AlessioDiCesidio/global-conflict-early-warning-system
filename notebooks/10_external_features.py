import pandas as pd
import numpy as np


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/country_month_panel_external.csv"
)

df["month"] = pd.to_datetime(df["month"])

df = df.sort_values(
    ["country", "month"]
).reset_index(drop=True)


# ---------------------------------------------------------
# BASIC EXTERNAL FEATURES
# ---------------------------------------------------------

# Fatalities relative to population
df["fatalities_per_100k"] = (
    df["fatalities"] /
    df["population"] *
    100000
)


# GDP per capita is already provided by World Bank
df["log_gdp_per_capita"] = np.log1p(
    df["gdp_per_capita"]
)


# Log population
df["log_population"] = np.log1p(
    df["population"]
)

# ---------------------------------------------------------
# PREVIOUS-YEAR EXTERNAL FEATURES
# ---------------------------------------------------------

df["population_lag_12"] = (
    df.groupby("country")["population"]
      .shift(12)
)

df["gdp_per_capita_lag_12"] = (
    df.groupby("country")["gdp_per_capita"]
      .shift(12)
)

df["gdp_growth_lag_12"] = (
    df.groupby("country")["gdp_growth"]
      .shift(12)
)


# ---------------------------------------------------------
# CHECK
# ---------------------------------------------------------

external_features = [
    "population",
    "gdp",
    "gdp_per_capita",
    "gdp_growth",
    "fatalities_per_100k",
    "log_gdp_per_capita",
    "log_population",
    "population_lag_12",
    "gdp_per_capita_lag_12",
    "gdp_growth_lag_12"
]

print("\n\nEXTERNAL FEATURE SUMMARY\n")

print(
    df[external_features]
    .describe()
    .T
)


print("\n\nMISSING VALUES\n")

print(
    df[external_features]
    .isna()
    .sum()
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_path = (
    "data/processed/"
    "country_month_panel_external_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)