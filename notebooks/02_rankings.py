import pandas as pd


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/country_month_panel_full.csv"
)

df["month"] = pd.to_datetime(df["month"])


# ---------------------------------------------------------
# TOP COUNTRY-MONTHS
# ---------------------------------------------------------

top_months = (
    df.nlargest(
        30,
        "fatalities"
    )
    [
        [
            "country",
            "month",
            "events",
            "fatalities",
            "civilian_fatalities"
        ]
    ]
)

print("\nTOP 30 COUNTRY-MONTHS BY FATALITIES\n")
print(
    top_months.to_string(index=False)
)


# ---------------------------------------------------------
# TOTAL FATALITIES BY COUNTRY
# ---------------------------------------------------------

country_totals = (
    df.groupby(
        "country",
        as_index=False
    )
    .agg(
        fatalities=("fatalities", "sum"),
        events=("events", "sum")
    )
    .sort_values(
        "fatalities",
        ascending=False
    )
)

print("\n\nTOP 30 COUNTRIES BY TOTAL FATALITIES\n")

print(
    country_totals.head(30)
    .to_string(index=False)
)


# ---------------------------------------------------------
# TOP COUNTRIES BY CIVILIAN FATALITIES
# ---------------------------------------------------------

civilian_totals = (
    df.groupby(
        "country",
        as_index=False
    )
    .agg(
        civilian_fatalities=(
            "civilian_fatalities",
            "sum"
        )
    )
    .sort_values(
        "civilian_fatalities",
        ascending=False
    )
)

print(
    "\n\nTOP 30 COUNTRIES BY CIVILIAN FATALITIES\n"
)

print(
    civilian_totals.head(30)
    .to_string(index=False)
)