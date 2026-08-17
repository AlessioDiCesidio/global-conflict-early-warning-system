import pandas as pd
import requests
from pathlib import Path


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp": "NY.GDP.MKTP.CD",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
}

START_YEAR = 1989
END_YEAR = 2025


# ---------------------------------------------------------
# DOWNLOAD WORLD BANK DATA
# ---------------------------------------------------------

def download_world_bank(indicator):

    url = (
        f"https://api.worldbank.org/v2/country/all/"
        f"indicator/{indicator}"
        f"?format=json"
        f"&per_page=20000"
        f"&date={START_YEAR}:{END_YEAR}"
    )

    response = requests.get(url)

    response.raise_for_status()

    data = response.json()[1]

    return pd.DataFrame(data)


# ---------------------------------------------------------
# DOWNLOAD ALL INDICATORS
# ---------------------------------------------------------

all_data = []

for name, indicator in INDICATORS.items():

    print(f"Downloading {name}...")

    temp = download_world_bank(indicator)

    temp = temp[
        [
            "countryiso3code",
            "date",
            "value"
        ]
    ]

    temp = temp[
        temp["countryiso3code"].notna()
    ]

    temp["countryiso3code"] = (temp["countryiso3code"].str.strip())

    temp = temp[temp["countryiso3code"] != "" ]

    temp = temp.rename(
        columns={
            "countryiso3code": "country_code",
            "date": "year",
            "value": name
        }
    )

    temp["year"] = temp["year"].astype(int)

    all_data.append(temp)


# ---------------------------------------------------------
# MERGE INDICATORS
# ---------------------------------------------------------

wb = all_data[0]

for temp in all_data[1:]:

    wb = wb.merge(
        temp,
        on=["country_code", "year"],
        how="outer"
    )


# ---------------------------------------------------------
# CLEAN
# ---------------------------------------------------------

wb = wb.sort_values(
    ["country_code", "year"]
).reset_index(drop=True)


print("\nWORLD BANK DATA\n")

print(wb.head())

print("\n\nSHAPE")

print(wb.shape)

print("\n\nMISSING VALUES")

print(wb.isna().sum())


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_path = Path(
    "data/processed/world_bank_country_year.csv"
)

wb.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)