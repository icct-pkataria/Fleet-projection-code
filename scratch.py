import pandas as pd
import numpy as np
import re
from pathlib import Path

# ===== FILES (your paths) =====
PATH_NUTS = Path("/Users/PycharmProjects/EUPC/NUTS 3 vehicle stock 2022-2023.xlsx")
PATH_SUMM = Path("/Users/PycharmProjects/EUPC/results_summary(in).xlsx")
OUT_PATH  = Path("/Users/PycharmProjects/EUPC/regional_projection_2024_2045.xlsx")

# ===== SETTINGS =====
YEARS_OUT = list(range(2024, 2046))   # 2024..2045 inclusive
BASE_YEAR = 2023
EV_TARGET_YEAR = 2055

# NUTS2 prefix -> ISO3
CC2_TO_ISO3 = {
    "AT":"AUT","BE":"BEL","BG":"BGR","HR":"HRV","CY":"CYP","CZ":"CZE","DK":"DNK","EE":"EST","FI":"FIN","FR":"FRA",
    "DE":"DEU","EL":"GRC","GR":"GRC","HU":"HUN","IE":"IRL","IS":"ISL","IT":"ITA","LV":"LVA","LT":"LTU","LU":"LUX",
    "MT":"MLT","NL":"NLD","NO":"NOR","PL":"POL","PT":"PRT","RO":"ROU","SK":"SVK","SI":"SVN","ES":"ESP","SE":"SWE",
    "UK":"GBR","GB":"GBR"
}

# ---------- helpers ----------
def norm(s: str) -> str:
    """normalize column name: lower, strip, collapse spaces, remove non-alnum except underscore"""
    s = str(s).replace("\xa0", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", "_", s)

def read_roadmap(path: Path) -> pd.DataFrame:
    """Read roadmap sheet; try header on row 1 then row 2; normalize and map needed columns."""
    sheet = "Sheet 1 - results_summary(in)"
    for hdr in (0, 1):
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=hdr)
        except Exception:
            continue
        cols_norm = {c: norm(c) for c in df.columns}
        df = df.rename(columns=cols_norm)

        # alias map -> canonical
        want = {
            "iso":           ["iso"],
            "vehcat":        ["vehcat", "vehicle_category", "veh_category", "veh_cat"],
            "vehicle":       ["vehicle", "veh", "vehname"],
            "powertrain":    ["powertrain", "pwrtrain", "pt"],
            "fuel":          ["fuel", "fuel_type"],
            "cy":            ["cy", "year"],
            "stock":         ["stock", "fleet_stock", "vehicles"],
            "vktpveh":       ["vktpveh", "vkt_per_vehicle", "vktveh", "vkt_pveh"],
            "mjpkm":         ["mjpkm", "mj_per_km", "energy_intensity", "mj_km"],
        }

        found = {}
        cols_set = set(df.columns)
        for canon, aliases in want.items():
            hit = next((a for a in aliases if a in cols_set), None)
            if not hit:
                break
            found[canon] = hit
        else:
            # got them all
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

    raise ValueError("Could not detect roadmap headers. Open the file and ensure the sheet name is exactly "
                     "'Sheet 1 - results_summary(in)' and columns include ISO, VehCat, Vehicle, Powertrain, Fuel, "
                     "CY/Year, Stock, VKTpVeh, MJpKM (any spacing/case is fine).")

def lerp(y0, y1, x0, x1, x):
    if x <= x0: return y0
    if x >= x1: return y1
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

# ===== NUTS3 2023 shares =====
nuts_raw = pd.read_excel(PATH_NUTS, sheet_name="Sheet1")
nuts_raw = nuts_raw.rename(columns={c: c.replace("\xa0", " ").strip() for c in nuts_raw.columns})
# be resilient to trailing spaces in provided file headers
map_nuts = {
    "GEO (Codes)": "nuts3",
    "GEO (Labels)": "Subregion",
    "2023 total vehicle stock ": "stock_2023_total",
    "2023 EV  stock": "stock_2023_ev",
}
for k in list(map_nuts.keys()):
    if k not in nuts_raw.columns:
        # try trimmed key
        alt = k.strip()
        if alt in nuts_raw.columns:
            map_nuts[alt] = map_nuts.pop(k)
nuts = nuts_raw.rename(columns=map_nuts)
required_nuts = ["nuts3", "Subregion", "stock_2023_total"]
missing_nuts = [c for c in required_nuts if c not in nuts.columns]
if missing_nuts:
    raise ValueError(f"NUTS file missing columns: {missing_nuts}")

nuts["ISO"] = nuts["nuts3"].astype(str).str[:2].map(CC2_TO_ISO3)
nuts = nuts.dropna(subset=["ISO", "stock_2023_total"]).copy()

# calculate totals & shares
totals_2023 = nuts.groupby("ISO", as_index=False).agg(
    country_total_2023=("stock_2023_total", "sum"),
    country_ev_2023=("stock_2023_ev", "sum"),
)
nuts = nuts.merge(totals_2023, on="ISO", how="left")
nuts["total_share_2023"] = nuts["stock_2023_total"] / nuts["country_total_2023"]
nuts["ev_share_2023"] = np.where(
    (nuts["country_ev_2023"] > 0) & nuts["stock_2023_ev"].notna(),
    nuts["stock_2023_ev"] / nuts["country_ev_2023"],
    0.0,
)

# ===== Roadmap (auto-detected headers) =====
summary = read_roadmap(PATH_SUMM)
summary = summary[summary["CY"].isin(YEARS_OUT)].copy()

# national totals/splits per ISO, CY, Vehicle
nat_totals_vehicle = summary.groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum() \
                            .rename(columns={"Stock":"TOTAL_nat_vehicle"})
is_ev_mask = summary["Powertrain"].str.upper().str.contains("BEV") | summary["Powertrain"].str.upper().str.contains("PHEV")
ev_totals_vehicle = summary[is_ev_mask].groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum() \
                                       .rename(columns={"Stock":"EV_nat_vehicle"})
bev_nat = summary[summary["Powertrain"].str.upper().eq("BEV")].groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum() \
         .rename(columns={"Stock":"BEV_nat_vehicle"})
phev_nat = summary[summary["Powertrain"].str.upper().str.contains("PHEV", na=False)].groupby(["ISO","CY","Vehicle"], as_index=False)["Stock"].sum() \
          .rename(columns={"Stock":"PHEV_nat_vehicle"})

nat_vehicle = nat_totals_vehicle.merge(ev_totals_vehicle, on=["ISO","CY","Vehicle"], how="left") \
                                .merge(bev_nat, on=["ISO","CY","Vehicle"], how="left") \
                                .merge(phev_nat, on=["ISO","CY","Vehicle"], how="left")
nat_vehicle[["EV_nat_vehicle","BEV_nat_vehicle","PHEV_nat_vehicle"]] = \
    nat_vehicle[["EV_nat_vehicle","BEV_nat_vehicle","PHEV_nat_vehicle"]].fillna(0.0)

# row-level splits within BEV/PHEV (per Vehicle)
bev_rows  = summary[summary["Powertrain"].str.upper().eq("BEV")].copy()
phev_rows = summary[summary["Powertrain"].str.upper().str.contains("PHEV", na=False)].copy()

def row_shares(df):
    if df.empty:
        return df.assign(row_share=pd.Series(dtype=float))
    grp_sum = df.groupby(["ISO","CY","Vehicle","Powertrain"])["Stock"].transform("sum")
    df["row_share"] = np.where(grp_sum > 0, df["Stock"] / grp_sum, 0.0)
    return df[["ISO","CY","Vehicle","Powertrain","Fuel","row_share","VKTpVeh","MJpKM"]]

bev_row_sh  = row_shares(bev_rows)
phev_row_sh = row_shares(phev_rows)


# ===== build regional shares for 2024–2045 =====
rows = []
for _, r in nuts.iterrows():
    for y in YEARS_OUT:
        rows.append({
            "ISO": r["ISO"],
            "Subregion": r["Subregion"],
            "CY": int(y),
            "total_share": r["total_share_2023"],  # fixed from 2023
            "ev_share": lerp(r["ev_share_2023"], r["total_share_2023"], BASE_YEAR, EV_TARGET_YEAR, y),
        })
alloc_shares = pd.DataFrame(rows)

# ===== allocate per Vehicle to regions =====
out_rows = []
for (iso, cy), g_share in alloc_shares.groupby(["ISO","CY"]):
    nat_v = nat_vehicle[(nat_vehicle["ISO"] == iso) & (nat_vehicle["CY"] == cy)]
    if nat_v.empty:
        continue

    for _, nv in nat_v.iterrows():
        veh = nv["Vehicle"]
        total_nat = float(nv["TOTAL_nat_vehicle"])
        ev_nat    = float(nv["EV_nat_vehicle"])
        bev_nat   = float(nv["BEV_nat_vehicle"])
        phev_nat  = float(nv["PHEV_nat_vehicle"])

        bev_share_nat  = (bev_nat / ev_nat)  if ev_nat > 0 else 0.0
        phev_share_nat = (phev_nat / ev_nat) if ev_nat > 0 else 0.0

        bev_rows_v  = bev_row_sh[(bev_row_sh["ISO"]==iso) & (bev_row_sh["CY"]==cy) & (bev_row_sh["Vehicle"]==veh)]
        phev_rows_v = phev_row_sh[(phev_row_sh["ISO"]==iso) & (phev_row_sh["CY"]==cy) & (phev_row_sh["Vehicle"]==veh)]
        bev_int     = bev_intens[(bev_intens["ISO"]==iso) & (bev_intens["CY"]==cy) & (bev_intens["Vehicle"]==veh)]
        bev_vkt_val = float(bev_int["VKTpVeh_BEV"].iloc[0]) if not bev_int.empty else np.nan
        bev_mj_val  = float(bev_int["MJpKM_BEV"].iloc[0])   if not bev_int.empty else np.nan

        for _, sr in g_share.iterrows():
            sub = sr["Subregion"]
            t_share = float(sr["total_share"])
            e_share = float(sr["ev_share"])

            # total row
            total_reg_veh = total_nat * t_share
            out_rows.append([iso, sub, int(cy), veh, "total", "", total_reg_veh, "", ""])

            # EV split
            ev_reg_veh   = ev_nat * e_share
            bev_reg_veh  = ev_reg_veh * bev_share_nat
            phev_reg_veh = ev_reg_veh * phev_share_nat

            # BEV (collapsed)
            if bev_reg_veh > 0:
                out_rows.append([iso, sub, int(cy), veh, "BEV", "Electric", bev_reg_veh, bev_vkt_val, bev_mj_val])

            # PHEV (detailed per roadmap row shares)
            if (phev_reg_veh > 0) and (not phev_rows_v.empty):
                shares = phev_rows_v["row_share"].to_numpy()
                if (np.nansum(shares) == 0) or np.isnan(shares).all():
                    shares = np.ones(len(phev_rows_v)) / len(phev_rows_v)
                for i, rr in phev_rows_v.reset_index(drop=True).iterrows():
                    share = rr["row_share"] if rr["row_share"] > 0 else shares[i]
                    out_rows.append([
                        iso, sub, int(cy), veh, rr["Powertrain"], rr["Fuel"],
                        phev_reg_veh * float(share), rr["VKTpVeh"], rr["MJpKM"]
                    ])

# ===== save =====
out = pd.DataFrame(out_rows, columns=["ISO","Subregion","CY","Vehicle","Powertrain","Fuel","Stock","VKTpVeh","MJpKM"])
out = out.sort_values(["ISO","Subregion","CY","Vehicle","Powertrain","Fuel"]).reset_index(drop=True)
out.to_excel(OUT_PATH, index=False)
print(f"Done. {len(out):,} rows → {OUT_PATH}")
