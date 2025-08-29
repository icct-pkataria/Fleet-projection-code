"""
EU regional EV allocation
---------------------------------------

WHAT THIS DOES
- Reads NUTS3 2022–2023 stock (totals + EVs), and an EU-wide roadmap summary.
- Builds regional shares and allocates national vehicle + EV line stocks (BEV + PHEV by fuel)
  to every NUTS3 region for years 2024–2045.
- Writes a tidy output with Stock, VKTpVeh, MJpKM and stock_per_1000.

INPUT FILES
- "NUTS 3 vehicle stock 2022-2023.xlsx"         : NUTS3 totals (and EV totals) for 2023
- "results_summary(in).xlsx"                    : roadmap summary; sheet "Sheet 1 - results_summary(in)"
                                                  (headers may start on row 2 — the code already detects this)

OUTPUT
- "regional_projection_2024_2045.xlsx"          : regionalized projections 2024–2045

ASSUMPTIONS / NOTES
- Vehicle scope currently normalized to passenger cars only (PC).
- Roadmap must include columns like ISO, Vehicle, Powertrain, Fuel, CY/Year, Stock, VKTpVeh, MJpKM
  (the reader is tolerant to spacing/case and header row = 1 or 2).
- NUTS3 file must include NUTS code, name, 2023 total stock, and (optionally) 2023 EV stock.
- Locale quirks (commas, thin spaces) are sanitized where needed.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# === FILES ===
PATH_NUTS = Path("NUTS 3 vehicle stock 2022-2023.xlsx")
PATH_SUMM = Path("results_summary(in).xlsx")
OUT_PATH  = Path("regional_projection_2024_2045.xlsx")

# === SETTINGS ===
YEARS_OUT = list(range(2024, 2046))  # years 2024..2045
BASE_YEAR = 2023
EV_TARGET_YEAR = 2055

# 2-letter code -> ISO3
CC2_TO_ISO3 = {
    "AT":"AUT","BE":"BEL","BG":"BGR","HR":"HRV","CY":"CYP","CZ":"CZE","DK":"DNK","EE":"EST","FI":"FIN","FR":"FRA",
    "DE":"DEU","EL":"GRC","GR":"GRC","HU":"HUN","IE":"IRL","IS":"ISL","IT":"ITA","LV":"LVA","LT":"LTU","LU":"LUX",
    "MT":"MLT","NL":"NLD","NO":"NOR","PL":"POL","PT":"PRT","RO":"ROU","SK":"SVK","SI":"SVN","ES":"ESP","SE":"SWE",
    "UK":"GBR","GB":"GBR"
}

# --- helpers: simple names for columns ---
def norm(s: str) -> str:
    s = str(s).replace("\xa0", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", "_", s)

def read_roadmap(path: Path) -> pd.DataFrame:
    """read roadmap; header may be on row 1 or row 2; return standard columns"""
    sheet = "Sheet 1 - results_summary(in)"
    for hdr in (0, 1):
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=hdr)
        except Exception:
            continue
        df = df.rename(columns={c: norm(c) for c in df.columns})

        # map many names -> one name
        want = {
            "iso":           ["iso"],
            "vehcat":        ["vehcat","vehicle_category","veh_category","veh_cat"],
            "vehicle":       ["vehicle","veh","vehname"],
            "powertrain":    ["powertrain","pwrtrain","pt"],
            "fuel":          ["fuel","fuel_type"],
            "cy":            ["cy","year"],
            "stock":         ["stock","fleet_stock","vehicles"],
            "vktpveh":       ["vktpveh","vkt_per_vehicle","vktveh","vkt_pveh"],
            "mjpkm":         ["mjpkm","mj_per_km","energy_intensity","mj_km"],
        }

        found = {}
        cols = set(df.columns)
        for canon, aliases in want.items():
            hit = next((a for a in aliases if a in cols), None)
            if not hit:
                break
            found[canon] = hit
        else:
            keep = [found["iso"], found["vehcat"], found["vehicle"], found["powertrain"], found["fuel"],
                    found["cy"], found["stock"], found["vktpveh"], found["mjpkm"]]
            out = df[keep].rename(columns={
                found["iso"]: "ISO",
                found["vehcat"]: "VehCat",
                found["vehicle"]: "Vehicle",
                found["powertrain"]: "Powertrain",
                found["fuel"]: "Fuel",
                found["cy"]: "CY",
                found["stock"]: "Stock",
                found["vktpveh"]: "VKTpVeh",
                found["mjpkm"]: "MJpKM",
            })
            return out

    raise ValueError(
        "Roadmap headers not found. Sheet must be 'Sheet 1 - results_summary(in)' with ISO, VehCat, "
        "Vehicle, Powertrain, Fuel, CY/Year, Stock, VKTpVeh, MJpKM."
    )

def lerp(y0, y1, x0, x1, x):
    if x <= x0: return y0
    if x >= x1: return y1
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

# === load NUTS3 data ===
nuts_raw = pd.read_excel(PATH_NUTS, sheet_name="Sheet1")
nuts_raw = nuts_raw.rename(columns={c: c.replace("\xa0", " ").strip() for c in nuts_raw.columns})

# simple column map (be tolerant to spaces)
map_nuts = {
    "GEO (Codes)": "Subregion",            # NUTS3 code
    "GEO (Labels)": "SubregionName",       # NUTS3 name
    "2023 total vehicle stock ": "stock_2023_total",
    "2023 EV  stock": "stock_2023_ev",
}
for k in list(map_nuts.keys()):
    if k not in nuts_raw.columns:
        alt = k.strip()
        if alt in nuts_raw.columns:
            map_nuts[alt] = map_nuts.pop(k)
nuts = nuts_raw.rename(columns=map_nuts)

# clean rows (need code + name)
drop_regions = {"Ain","Unknown","Inconnu",None,""}
nuts["Subregion"] = nuts["Subregion"].astype(str).str.strip()
nuts["SubregionName"] = nuts["SubregionName"].astype(str).str.strip()
nuts = nuts[(nuts["Subregion"]!="") & (nuts["SubregionName"]!="")]
nuts = nuts[~nuts["SubregionName"].isin(drop_regions)].copy()

need = ["Subregion","SubregionName","stock_2023_total"]
miss = [c for c in need if c not in nuts.columns]
if miss:
    raise ValueError(f"NUTS file missing columns: {miss}")

# add ISO from NUTS3 code
nuts["ISO"] = nuts["Subregion"].astype(str).str[:2].map(CC2_TO_ISO3)
nuts = nuts.dropna(subset=["ISO","stock_2023_total"]).copy()

# country totals (2023)
totals_2023 = nuts.groupby("ISO", as_index=False).agg(
    country_total_2023=("stock_2023_total","sum"),
    country_ev_2023=("stock_2023_ev","sum"),
)
nuts = nuts.merge(totals_2023, on="ISO", how="left")

# share of each region in 2023
nuts["total_share_2023"] = nuts["stock_2023_total"] / nuts["country_total_2023"]
nuts["ev_share_2023"] = np.where(
    (nuts["country_ev_2023"] > 0) & nuts["stock_2023_ev"].notna(),
    nuts["stock_2023_ev"] / nuts["country_ev_2023"],
    0.0,
)

# === load roadmap (auto header) ===
summary = read_roadmap(PATH_SUMM)
summary = summary[summary["CY"].isin(YEARS_OUT)].copy()

# keep passenger cars
summary = summary[summary["Vehicle"].str.upper().isin(["PC","PASSENGER CARS","CARS"])].copy()
summary["Vehicle"] = "PC"

# === national totals (by ISO, year, vehicle) ===
nat_totals_vehicle = (summary.groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum()
                      .rename(columns={"Stock":"TOTAL_nat_vehicle"}))

# EV rows (BEV + PHEV) at nation level
ptu = summary["Powertrain"].astype(str).str.upper()
is_ev = ptu.str.contains("BEV") | ptu.str.contains("PHEV")
ev_totals_vehicle = (summary[is_ev].groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum()
                     .rename(columns={"Stock":"EV_nat_vehicle"}))

# BEV total nation
bev_nat = (summary[ptu.eq("BEV")].groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum()
           .rename(columns={"Stock":"BEV_nat_vehicle"}))

# PHEV total nation
phev_nat = (summary[ptu.str.contains("PHEV", na=False)].groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum()
            .rename(columns={"Stock":"PHEV_nat_vehicle"}))

# join nation-level numbers
nat_vehicle = (nat_totals_vehicle
               .merge(ev_totals_vehicle, on=["ISO","CY","Vehicle"], how="left")
               .merge(bev_nat, on=["ISO","CY","Vehicle"], how="left")
               .merge(phev_nat, on=["ISO","CY","Vehicle"], how="left"))
nat_vehicle[["EV_nat_vehicle","BEV_nat_vehicle","PHEV_nat_vehicle"]] = \
    nat_vehicle[["EV_nat_vehicle","BEV_nat_vehicle","PHEV_nat_vehicle"]].fillna(0.0)

# === EV rows share of EV total (simple, clear) ===
ev_lines = summary[ ptu.eq("BEV") | ptu.str.contains("PHEV", na=False) ].copy()
ev_nat_tot = (ev_lines.groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum()
              .rename(columns={"Stock":"EV_nat_total"}))
ev_lines = ev_lines.merge(ev_nat_tot, on=["ISO","CY","Vehicle"], how="left")
ev_lines["EV_row_share"] = np.where(ev_lines["EV_nat_total"]>0,
                                    ev_lines["Stock"] / ev_lines["EV_nat_total"], 0.0)

# keep small set for lookup
ev_row_shares = ev_lines[["ISO","CY","Vehicle","Powertrain","Fuel","EV_row_share","VKTpVeh","MJpKM"]].copy()

# === make regional shares for all years ===
rows = []
for _, r in nuts.iterrows():
    for y in YEARS_OUT:
        rows.append({
            "ISO": r["ISO"],
            "Subregion": r["Subregion"],
            "SubregionName": r["SubregionName"],
            "CY": int(y),
            "total_share": r["total_share_2023"],                 # share of all cars (fixed from 2023)
            "ev_share": lerp(r["ev_share_2023"], r["total_share_2023"], BASE_YEAR, EV_TARGET_YEAR, y),  # EV share moves
        })
alloc_shares = pd.DataFrame(rows)

# === allocate to regions ===
out_rows = []
for (iso, cy), g_share in alloc_shares.groupby(["ISO","CY"]):
    nv = nat_vehicle[(nat_vehicle["ISO"]==iso) & (nat_vehicle["CY"]==cy)]
    if nv.empty:
        continue

    # rows for EV in this country-year
    ev_xy = ev_row_shares[(ev_row_shares["ISO"]==iso) & (ev_row_shares["CY"]==cy)]

    for _, nv_row in nv.iterrows():
        veh       = nv_row["Vehicle"]
        total_nat = float(nv_row["TOTAL_nat_vehicle"])
        ev_nat    = float(nv_row["EV_nat_vehicle"])

        # filter to this vehicle
        ev_v = ev_xy[ev_xy["Vehicle"]==veh]

        if ev_nat <= 0 or ev_v.empty:
            # only totals, no EV
            for _, sr in g_share.iterrows():
                sub  = sr["Subregion"]
                name = sr["SubregionName"]
                t_sh = float(sr["total_share"])
                total_reg_veh = total_nat * t_sh
                out_rows.append([iso, sub, name, int(cy), veh, "total", "", total_reg_veh, "", ""])
            continue

        for _, sr in g_share.iterrows():
            sub  = sr["Subregion"]
            name = sr["SubregionName"]
            t_sh = float(sr["total_share"])
            e_sh = float(sr["ev_share"])

            # total cars for region
            total_reg_veh = total_nat * t_sh
            out_rows.append([iso, sub, name, int(cy), veh, "total", "", total_reg_veh, "", ""])

            # EV stock for region
            ev_reg_veh = ev_nat * e_sh

            # scale EV rows
            for _, ln in ev_v.iterrows():
                pt   = ln["Powertrain"]
                fuel = ln["Fuel"]
                sh   = float(ln["EV_row_share"])
                vkt  = ln["VKTpVeh"]
                mj   = ln["MJpKM"]

                reg_stock = ev_reg_veh * sh
                out_rows.append([iso, sub, name, int(cy), veh, pt, fuel, reg_stock, vkt, mj])

# === build output table ===
out = pd.DataFrame(
    out_rows,
    columns=["ISO","Subregion","SubregionName","CY","Vehicle","Powertrain","Fuel","Stock","VKTpVeh","MJpKM"]
).sort_values(["ISO","Subregion","SubregionName","CY","Vehicle","Powertrain","Fuel"]).reset_index(drop=True)

# === add EV per 1000 cars ===
# total cars per region
totals = (out[out["Powertrain"]=="total"][["ISO","Subregion","SubregionName","CY","Vehicle","Stock"]]
          .rename(columns={"Stock":"TotalStock"}))

# join totals
out = out.merge(totals, on=["ISO","Subregion","SubregionName","CY","Vehicle"], how="left")

# EV per 1000 cars
out["stock_per_1000"] = np.where(
    out["Powertrain"]!="total",
    (out["Stock"] / out["TotalStock"]) * 1000,
    np.nan
)

# === save file ===
out.to_excel(OUT_PATH, index=False)
print(f"Done. {len(out):,} rows → {OUT_PATH}")