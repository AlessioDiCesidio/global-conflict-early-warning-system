import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/country_month_panel_full.csv"
)

df["month"] = pd.to_datetime(df["month"])


# ---------------------------------------------------------
# GLOBAL MONTHLY FATALITIES
# ---------------------------------------------------------

monthly = (
    df.groupby("month", as_index=False)
      .agg(
          fatalities=("fatalities", "sum"),
          events=("events", "sum")
      )
)


# ---------------------------------------------------------
# GRAPH 1 — LINEAR SCALE
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    monthly["month"],
    monthly["fatalities"]
)

plt.title(
    "Global Recorded Fatalities from Organized Violence"
)

plt.xlabel("Year")
plt.ylabel("Fatalities")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/global_fatalities_linear.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# GRAPH 2 — LOG SCALE
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    monthly["month"],
    monthly["fatalities"]
)

plt.yscale("log")

plt.title(
    "Global Recorded Fatalities from Organized Violence — Log Scale"
)

plt.xlabel("Year")
plt.ylabel("Fatalities (log scale)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/global_fatalities_log.png",
    dpi=300
)

plt.show()