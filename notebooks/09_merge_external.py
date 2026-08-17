import pandas as pd


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

ucdp = pd.read_csv(
    "data/processed/country_month_panel_full.csv"
)

wb = pd.read_csv(
    "data/processed/world_bank_country_year.csv"
)

ucdp["month"] = pd.to_datetime(ucdp["month"])

ucdp["year"] = ucdp["month"].dt.year


# ---------------------------------------------------------
# COUNTRY NAME → ISO3 MAPPING
# ---------------------------------------------------------

country_mapping = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Bahrain": "BHR",
    "Bhutan": "BTN",
    "Cambodia (Kampuchea)": "KHM",
    "DR Congo (Zaire)": "COD",
    "Kingdom of eSwatini (Swaziland)": "SWZ",
    "Kuwait": "KWT",
    "Madagascar (Malagasy)": "MDG",
    "Malta": "MLT",
    "Myanmar (Burma)": "MMR",
    "Qatar": "QAT",
    "Russia (Soviet Union)": "RUS",
    "Serbia (Yugoslavia)": "SRB",
    "Solomon Islands": "SLB",
    "Trinidad and Tobago": "TTO",
    "United States of America": "USA",
    "Yemen (North Yemen)": "YEM",
    "Zimbabwe (Rhodesia)": "ZWE",
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Angola": "AGO",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Azerbaijan": "AZE",
    "Bangladesh": "BGD",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Belize": "BLZ",
    "Benin": "BEN",
    "Bolivia": "BOL",
    "Bosnia-Herzegovina": "BIH",
    "Botswana": "BWA",
    "Brazil": "BRA",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cambodia": "KHM",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Comoros": "COM",
    "Congo": "COG",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cuba": "CUB",
    "Cyprus": "CYP",
    "Democratic Republic of the Congo": "COD",
    "Djibouti": "DJI",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Eritrea": "ERI",
    "Estonia": "EST",
    "Ethiopia": "ETH",
    "France": "FRA",
    "Gabon": "GAB",
    "Gambia": "GMB",
    "Georgia": "GEO",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guinea-Bissau": "GNB",
    "Guyana": "GUY",
    "Haiti": "HTI",
    "Honduras": "HND",
    "Hungary": "HUN",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Ivory Coast": "CIV",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kyrgyzstan": "KGZ",
    "Laos": "LAO",
    "Latvia": "LVA",
    "Lebanon": "LBN",
    "Lesotho": "LSO",
    "Liberia": "LBR",
    "Libya": "LBY",
    "Lithuania": "LTU",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Mali": "MLI",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Mongolia": "MNG",
    "Montenegro": "MNE",
    "Morocco": "MAR",
    "Mozambique": "MOZ",
    "Myanmar": "MMR",
    "Namibia": "NAM",
    "Nepal": "NPL",
    "Netherlands": "NLD",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "North Korea": "PRK",
    "North Macedonia": "MKD",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Papua New Guinea": "PNG",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "Russia": "RUS",
    "Rwanda": "RWA",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Sierra Leone": "SLE",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Somalia": "SOM",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "South Sudan": "SSD",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sudan": "SDN",
    "Swaziland": "SWZ",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tajikistan": "TJK",
    "Tanzania": "TZA",
    "Thailand": "THA",
    "Togo": "TGO",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Turkmenistan": "TKM",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Yemen": "YEM",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE"
}


# ---------------------------------------------------------
# APPLY MAPPING
# ---------------------------------------------------------

ucdp["country_code"] = ucdp["country"].map(
    country_mapping
)


# ---------------------------------------------------------
# CHECK UNMATCHED COUNTRIES
# ---------------------------------------------------------

unmatched = sorted(
    ucdp.loc[
        ucdp["country_code"].isna(),
        "country"
    ].unique()
)

print("\nUNMATCHED UCDP COUNTRIES\n")

print(unmatched)

print(
    f"\nUnmatched: {len(unmatched)}"
)


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

merged = ucdp.merge(
    wb,
    on=["country_code", "year"],
    how="left",
    validate="many_to_one"
)


# ---------------------------------------------------------
# MERGE QUALITY
# ---------------------------------------------------------

external_cols = [
    "population",
    "gdp",
    "gdp_per_capita",
    "gdp_growth"
]

print("\n\nMERGED DATA")

print(
    merged[
        [
            "country",
            "year",
            "population",
            "gdp",
            "gdp_per_capita",
            "gdp_growth"
        ]
    ].head(20)
)

print("\n\nEXTERNAL DATA COVERAGE\n")

print(
    merged[external_cols]
    .notna()
    .mean()
)

print("\n\nMERGED SHAPE")

print(merged.shape)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_path = (
    "data/processed/"
    "country_month_panel_external.csv"
)

merged.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)

# ---------------------------------------------------------
# EXTERNAL DATA COVERAGE BY YEAR
# ---------------------------------------------------------

print("\n\nEXTERNAL DATA COVERAGE BY YEAR\n")

coverage_by_year = (
    merged
    .groupby("year")[external_cols]
    .apply(lambda x: x.notna().mean())
)

print(
    coverage_by_year.to_string()
)


# ---------------------------------------------------------
# COUNTRIES WITH MISSING EXTERNAL DATA
# ---------------------------------------------------------

print("\n\nCOUNTRIES WITH MISSING POPULATION\n")

missing_population = (
    merged.loc[
        merged["population"].isna(),
        "country"
    ]
    .drop_duplicates()
    .sort_values()
)

print(
    missing_population.to_list()
)


print("\n\nCOUNTRIES WITH MISSING GDP\n")

missing_gdp = (
    merged.loc[
        merged["gdp"].isna(),
        "country"
    ]
    .drop_duplicates()
    .sort_values()
)

print(
    missing_gdp.to_list()
)