import pandas as pd
import numpy as np

# ========= USER PATHS =========
COMMUTERS_FILE = "/Users/PycharmProjects/EUPC/.venv/commuters_with_car_shares _fleet and car pool adjusted.xlsx"
COMMUTERS_SHEET = "Car commuters"
ROADMAP_FILE   = "/Users/PycharmProjects/EUPC/results_summary(in).xlsx"
OUTPUT_CSV     = "evcharge_vkt_ratio.csv"

# EU fallback share of car-km that are commuting (distance share), e.g. 0.3315
s_EU = 0.3315

# ---------- helpers ----------
def to_num(x):
    if isinstance(x, str):
        x = x.replace("\u00A0","").strip()
        # treat commas smartly: if only commas and no dot -> thousands sep; drop
        if x.count(",") > 0 and x.count(".") == 0:
            x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

# ---------- load commuters (NUTS3) ----------
df = pd.read_excel(COMMUTERS_FILE, sheet_name=COMMUTERS_SHEET)
# required columns
need_cols = {"NUTS_ID","Region","FinalShare_Car"}
missing = need_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Commuters file missing columns: {missing}")

df = df[["NUTS_ID","Region","FinalShare_Car"]].copy()
df["NUTS_ID"] = df["NUTS_ID"].astype(str).str.strip().str.upper()
df["ISO"]     = df["NUTS_ID"].str[:2]
# p = share of cars that are commuter cars in this NUTS
df["p"] = pd.to_numeric(df["FinalShare_Car"], errors="coerce")

# ---------- load roadmap (national PC VKT per vehicle) ----------
# Directly read the correct sheet and header row
rm = pd.read_excel(ROADMAP_FILE, sheet_name="Sheet 1 - results_summary(in)", header=1)

# accept ISO or CountryCode column
iso_col = None
for c in ["ISO","CountryCode","iso","countrycode"]:
    if c in rm.columns:
        iso_col = c
        break
if iso_col is None:
    raise RuntimeError("Roadmap file must contain an ISO column.")

if "VKTpVeh" not in rm.columns:
    raise RuntimeError("Roadmap file must contain a 'VKTpVeh' column (national PC km/vehicle).")

nrm = rm[[iso_col, "VKTpVeh"]].copy()
nrm["ISO_raw"] = nrm[iso_col].astype(str).str.strip().str.upper()
# Normalized key to match NUTS
iso_map = {
    "ALB": "AL", "AUT": "AT", "BEL": "BE", "BGR": "BG", "HRV": "HR", "CYP": "CY", "CZE": "CZ", "DNK": "DK",
    "EST": "EE", "FIN": "FI", "FRA": "FR", "DEU": "DE", "GRC": "EL", "HUN": "HU", "IRL": "IE", "ITA": "IT",
    "LVA": "LV", "LTU": "LT", "LUX": "LU", "MLT": "MT", "NLD": "NL", "POL": "PL", "PRT": "PT", "ROU": "RO",
    "SVK": "SK", "SVN": "SI", "ESP": "ES", "SWE": "SE", "NOR": "NO", "CHE": "CH", "ISL": "IS", "LIE": "LI",
    "GBR": "UK"
}
nrm["ISO_norm"] = nrm["ISO_raw"].replace(iso_map)
nrm["A"] = nrm["VKTpVeh"].apply(to_num)
# Group: keep mean A per ISO_norm and the first original ISO code for output
rm = nrm.groupby("ISO_norm", as_index=False).agg(A=("A","mean"), ISO_out=("ISO_raw","first"))

# ---------- attach national A and roadmap ISO to each NUTS ----------
df = df.merge(rm[["ISO_norm","A","ISO_out"]], left_on="ISO", right_on="ISO_norm", how="left")

# ---------- compute ratios ----------
# A = national average km/car (PC)
# s = EU-wide share of car-km that are commuting (distance share)
# p = NUTS3 share of cars that are commuter cars
s = s_EU

# guard rails to avoid zero-division; mark invalid as NaN
valid = (df["A"].notna() & df["p"].notna() & (df["p"]>0) & (s>0) & (s<1))

# commute km per commuter car = (s * A) / p
df.loc[valid, "commute_per_commuter"] = (s * df.loc[valid, "A"]) / df.loc[valid, "p"]

# non-commute km per car (same for commuter and non-commuter cars)
df.loc[valid, "noncommute_per_car"] = (1 - s) * df.loc[valid, "A"]

# total km per commuter car = commuting + non-commuting
df.loc[valid, "commuter_total"] = df.loc[valid, "commute_per_commuter"] + df.loc[valid, "noncommute_per_car"]

# ratio = total commuter-car km / non-commuter-car km (which equals non-commute per car)
df["Commuter_VKT_ratio"] = np.where(
    valid,
    df["commuter_total"] / df["noncommute_per_car"],
    np.nan
)

# ---------- build EV-Charge input ----------
out = pd.DataFrame({
    "ISO": df["ISO_out"].fillna(df["ISO"]),  # use roadmap ISO formatting
    "Subregion": df["NUTS_ID"],
    "Subregion_Name": df["Region"],
    "Vehicle": "PC",
    "Commuter_VKT_ratio": df["Commuter_VKT_ratio"].round(4)
})

# sanity clip for absurd values (e.g., p≈0 or 1)
out.loc[(out["Commuter_VKT_ratio"]<0) | (out["Commuter_VKT_ratio"]>10), "Commuter_VKT_ratio"] = np.nan

# ---- List ISOs (from NUTS) that have no roadmap A ----
miss_df = df[df["A"].isna()].copy()
if not miss_df.empty:
    by_iso = miss_df.groupby("ISO").size().sort_values(ascending=False)
    print("ISOs with no roadmap A (and their NUTS3 row counts):")
    for iso, cnt in by_iso.items():
        samples = ", ".join(miss_df.loc[miss_df["ISO"]==iso, "NUTS_ID"].head(3))
        print(f"  {iso}: {cnt} rows (e.g., {samples})")

# ---- Diagnostics for missing ratios ----
miss_A = df["A"].isna().sum()
miss_p = df["p"].isna().sum()
nonpos_p = (df["p"]<=0).sum()
unmatched_iso = df.loc[df["A"].isna(), "ISO"].nunique()
print(f"A missing (no roadmap match): {miss_A}")
print(f"p missing (FinalShare_Car NaN): {miss_p}")
print(f"p <= 0: {nonpos_p}")
print(f"Distinct NUTS ISO with no roadmap A: {unmatched_iso}")

out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Wrote {OUTPUT_CSV} with {out.shape[0]} rows.")
print("Null ratios (check inputs):", out["Commuter_VKT_ratio"].isna().sum())
