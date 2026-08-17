import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "GEDEvent_v26_1.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "country_month_panel.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading UCDP GED...")

df = pd.read_csv(
    RAW_FILE,
    usecols=[
        "country",
        "country_id",
        "region",
        "year",
        "date_start",
        "best",
        "deaths_a",
        "deaths_b",
        "deaths_civilians",
        "deaths_unknown",
    ],
)

print(f"Events loaded: {len(df):,}")


# ---------------------------------------------------------
# DATE
# ---------------------------------------------------------

df["date_start"] = pd.to_datetime(
    df["date_start"],
    errors="coerce"
)

df = df.dropna(subset=["date_start"])

df["month"] = (
    df["date_start"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ---------------------------------------------------------
# NUMERIC VARIABLES
# ---------------------------------------------------------

fatality_columns = [
    "best",
    "deaths_a",
    "deaths_b",
    "deaths_civilians",
    "deaths_unknown",
]

for col in fatality_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# ---------------------------------------------------------
# AGGREGATION
# Country × Month
# ---------------------------------------------------------

panel = (
    df.groupby(
        [
            "country",
            "country_id",
            "region",
            "month",
        ],
        as_index=False
    )
    .agg(
        events=("country", "size"),
        fatalities=("best", "sum"),
        deaths_a=("deaths_a", "sum"),
        deaths_b=("deaths_b", "sum"),
        civilian_fatalities=("deaths_civilians", "sum"),
        unknown_fatalities=("deaths_unknown", "sum"),
    )
)


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

panel = panel.sort_values(
    ["country", "month"]
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

panel.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("Panel created successfully.")
print(f"Rows: {len(panel):,}")
print(f"Countries: {panel['country'].nunique():,}")
print(
    f"Period: "
    f"{panel['month'].min().date()} → "
    f"{panel['month'].max().date()}"
)

print()
print(panel.head(10))
# ---------------------------------------------------------
# COMPLETE COUNTRY × MONTH PANEL
# ---------------------------------------------------------

print()
print("Creating complete country-month panel...")

countries = panel[
    ["country", "country_id", "region"]
].drop_duplicates()

all_months = pd.date_range(
    start="1989-01-01",
    end="2025-12-01",
    freq="MS"
)

countries["key"] = 1

months = pd.DataFrame({
    "month": all_months,
    "key": 1
})

full_panel = countries.merge(
    months,
    on="key"
).drop(columns="key")


full_panel = full_panel.merge(
    panel,
    on=[
        "country",
        "country_id",
        "region",
        "month"
    ],
    how="left"
)


numeric_columns = [
    "events",
    "fatalities",
    "deaths_a",
    "deaths_b",
    "civilian_fatalities",
    "unknown_fatalities",
]

full_panel[numeric_columns] = (
    full_panel[numeric_columns]
    .fillna(0)
)


full_panel = full_panel.sort_values(
    ["country", "month"]
)


FULL_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "country_month_panel_full.csv"
)


full_panel.to_csv(
    FULL_OUTPUT,
    index=False
)


print()
print("FULL PANEL CREATED")
print(f"Rows: {len(full_panel):,}")
print(f"Countries: {full_panel['country'].nunique():,}")
print(
    f"Period: "
    f"{full_panel['month'].min().date()} → "
    f"{full_panel['month'].max().date()}"
)

print()
print(full_panel.head())