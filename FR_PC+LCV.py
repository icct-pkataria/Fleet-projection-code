"""
France NUTS3 EV allocation (PC + LCV)

Inputs:
- FR_NUTS3_PC_LCV_stock_EV+total.xlsx   (NUTS3 totals + EVs for 2024)
- results_summary(in).xlsx              (roadmap summary, sheet: 'Sheet 1 - results_summary(in)')

Outputs:
- fr_regional_projection_with_totals.xlsx   (keeps the 'total' rows)
- fr_regional_projection_no_totals.xlsx     (drops 'total' rows + adds stock_per_1000)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# === FILES ===
PATH_FR_BASE = Path("FR_NUTS3_PC_LCV_stock_EV+total.xlsx")   # NUTS3 2024 totals
PATH_SUMM    = Path("results_summary(in).xlsx")               # Roadmap national
OUT_WITH     = Path("fr_regional_projection_with_totals.xlsx")
OUT_NO       = Path("fr_regional_projection_no_totals.xlsx")

# === SETTINGS ===
YEARS_OUT = list(range(2024, 2046))  # 2024..2045
BASE_YEAR = 2024
EV_TARGET_YEAR = 2055
ISO = "FRA"
FORCE_2024_MATCH_BASE = True  # make 2024 national totals = sum of base file

# --- helpers ---
def norm(s: str) -> str:
    """simple column normalizer"""
    s = str(s).replace("\xa0"," ").strip().lower()
    s = re.sub(r"\s+"," ", s).replace("/", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+","", s)
    return re.sub(r"\s+","_", s)

def read_roadmap(path: Path) -> pd.DataFrame:
    """read roadmap; header may be on row 1 or 2; return standard columns"""
    sheet = "Sheet 1 - results_summary(in)"
    for hdr in (0, 1):
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=hdr)
        except Exception:
            continue
        df = df.rename(columns={c: norm(c) for c in df.columns})

        need = {
            "iso":       ["iso"],
            "vehicle":   ["vehicle","veh"],
            "powertrain":["powertrain","pt"],
            "fuel":      ["fuel","fuel_type"],
            "cy":        ["cy","year"],
            "stock":     ["stock","fleet_stock","vehicles"],
            "vktpveh":   ["vktpveh","vkt_per_vehicle","vktveh"],
            "mjpkm":     ["mjpkm","mj_per_km","energy_intensity"],
        }
        found, cols = {}, set(df.columns)
        ok = True
        for k, als in need.items():
            hit = next((a for a in als if a in cols), None)
            if not hit:
                ok = False
                break
            found[k] = hit
        if not ok:
            continue

        out = df[[found["iso"],found["vehicle"],found["powertrain"],found["fuel"],
                  found["cy"],found["stock"],found["vktpveh"],found["mjpkm"]]].rename(columns={
            found["iso"]:"ISO", found["vehicle"]:"Vehicle", found["powertrain"]:"Powertrain",
            found["fuel"]:"Fuel", found["cy"]:"CY", found["stock"]:"Stock",
            found["vktpveh"]:"VKTpVeh", found["mjpkm"]:"MJpKM"
        })
        return out
    raise ValueError("Roadmap headers not detected (check sheet name and columns).")

def lerp(y0, y1, x0, x1, x):
    """linear move from y0 to y1 across years"""
    if x <= x0: return y0
    if x >= x1: return y1
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

def map_group_to_vehicle(g):
    """base file groups -> PC / LCV"""
    if g == "VP": return "PC"
    if g in ("VUL","VU"): return "LCV"
    return g

def map_roadmap_vehicle_to_group(v):
    """roadmap names -> PC / LCV"""
    s = str(v).upper()
    s_clean = re.sub(r"[^A-Z]", "", s)
    if s_clean == "PC":
        return "PC"
    if any(x in s_clean for x in ["LCV","VAN","VANS","LGV","LIGHTCOMMERCIAL"]):
        return "LCV"
    return "OTHER"

# === 1) read France base (NUTS3), 2024 ===
fr_base = pd.read_excel(PATH_FR_BASE)

# need these columns
need_cols = ["NUTS3_name","VehicleGroup","Total_2024","EV_2024"]
missing = [c for c in need_cols if c not in fr_base.columns]
if missing:
    raise ValueError(f"FR base file missing columns: {missing}")

# optional code column (many names)
code_col = None
for cand in ["NUTS3","NUTS3_code","NUTS3 Code","NUTS3_code","NUTS-3 code","nuts3","nuts3_code"]:
    if cand in fr_base.columns:
        code_col = cand
        break

# standardize code column
if code_col is not None:
    fr_base = fr_base.rename(columns={code_col: "NUTS3_code"})
else:
    fr_base["NUTS3_code"] = pd.NA  # will drop empties

# remove unwanted regions
drop_regions = {"Ain","Unknown","Inconnu",None,""}
fr_base = fr_base[~fr_base["NUTS3_name"].isin(drop_regions)].copy()

# drop empty codes
fr_base["NUTS3_code"] = fr_base["NUTS3_code"].astype(str).str.strip()
fr_base = fr_base[fr_base["NUTS3_code"].ne("") & fr_base["NUTS3_code"].notna()].copy()

# keep rows with totals
fr_base = fr_base[fr_base["Total_2024"].notna()].copy()

# map VP/VUL -> PC/LCV; keep only PC/LCV
fr_base["Vehicle"] = fr_base["VehicleGroup"].apply(map_group_to_vehicle)
fr_base = fr_base[fr_base["Vehicle"].isin(["PC","LCV"])].copy()

# national sum (from base file)
fr_nat_2024 = fr_base.groupby("Vehicle", as_index=False).agg(
    nat_total_2024=("Total_2024","sum"),
    nat_ev_2024=("EV_2024","sum")
)

# shares in 2024
fr_base = fr_base.merge(fr_nat_2024, on="Vehicle", how="left")
fr_base["total_share_base"] = np.where(fr_base["nat_total_2024"]>0,
                                       fr_base["Total_2024"]/fr_base["nat_total_2024"], 0.0)
fr_base["ev_share_base"]    = np.where(fr_base["nat_ev_2024"]>0,
                                       fr_base["EV_2024"]/fr_base["nat_ev_2024"], 0.0)

# expand shares to years; EV share moves toward total share by 2055
alloc_rows = []
for _, r in fr_base.iterrows():
    for y in YEARS_OUT:
        alloc_rows.append({
            "ISO": ISO,
            "Subregion": r["NUTS3_code"],
            "SubregionName": r["NUTS3_name"],
            "Vehicle": r["Vehicle"],   # PC / LCV
            "CY": int(y),
            "total_share": float(r["total_share_base"]),
            "ev_share": float(lerp(r["ev_share_base"], r["total_share_base"], BASE_YEAR, EV_TARGET_YEAR, y)),
        })
alloc = pd.DataFrame(alloc_rows)

# === 2) read roadmap (FRA only) ===
summary = read_roadmap(PATH_SUMM)
summary = summary[(summary["ISO"]==ISO) & (summary["CY"].isin(YEARS_OUT))].copy()

# roadmap vehicles -> PC/LCV
summary["VehicleGroup"] = summary["Vehicle"].apply(map_roadmap_vehicle_to_group)
summary = summary[summary["VehicleGroup"].isin(["PC","LCV"])].copy()
summary["Vehicle"] = summary["VehicleGroup"]

# national totals per year/vehicle
nat_totals = (summary.groupby(["CY","Vehicle"], as_index=False)["Stock"].sum()
                      .rename(columns={"Stock":"TOTAL_nat"}))

# national EV totals (BEV + PHEV)
ptu = summary["Powertrain"].astype(str).str.upper()
is_ev = (ptu=="BEV") | (ptu.str.contains("PHEV", na=False))
ev_totals = (summary[is_ev].groupby(["CY","Vehicle"], as_index=False)["Stock"].sum()
                        .rename(columns={"Stock":"EV_nat"}))

# join national totals
nat = nat_totals.merge(ev_totals, on=["CY","Vehicle"], how="left").fillna({"EV_nat":0.0})


# EV rows share of EV total (per year/vehicle)
ev_lines = summary[is_ev].copy()
ev_nat_per = (ev_lines.groupby(["CY","Vehicle"], as_index=False)["Stock"].sum()
              .rename(columns={"Stock":"EV_nat_total"}))
ev_lines = ev_lines.merge(ev_nat_per, on=["CY","Vehicle"], how="left")
ev_lines["EV_row_share"] = np.where(ev_lines["EV_nat_total"]>0,
                                    ev_lines["Stock"]/ev_lines["EV_nat_total"], 0.0)
ev_row_shares = ev_lines[["CY","Vehicle","Powertrain","Fuel","EV_row_share","VKTpVeh","MJpKM"]].copy()

# === 3) allocate to NUTS3 ===
rows_out = []
for (cy, veh), shard in alloc.groupby(["CY","Vehicle"]):
    nat_xy = nat[(nat["CY"]==cy) & (nat["Vehicle"]==veh)]
    if nat_xy.empty:
        continue
    total_nat = float(nat_xy["TOTAL_nat"].iloc[0])
    ev_nat    = float(nat_xy["EV_nat"].iloc[0])

    # EV rows (this year/vehicle)
    ev_sh_xy = ev_row_shares[(ev_row_shares["CY"]==cy) & (ev_row_shares["Vehicle"]==veh)]

    for _, s in shard.iterrows():
        sub_code = s["Subregion"]
        sub_name = s["SubregionName"]
        t_share  = float(s["total_share"])
        e_share  = float(s["ev_share"])

        # total cars for region
        total_reg = total_nat * t_share
        rows_out.append([ISO, sub_code, sub_name, int(cy), veh, "total", "", total_reg, "", ""])

        # EV cars for region
        if ev_nat > 0 and not ev_sh_xy.empty:
            ev_reg = ev_nat * e_share
            # scale EV rows
            for _, ln in ev_sh_xy.iterrows():
                pt   = ln["Powertrain"]
                fuel = ln["Fuel"]
                sh   = float(ln["EV_row_share"])
                vkt  = ln["VKTpVeh"]
                mj   = ln["MJpKM"]
                rows_out.append([ISO, sub_code, sub_name, int(cy), veh, pt, fuel, ev_reg*sh, vkt, mj])

# table
cols = ["ISO","Subregion","SubregionName","CY","Vehicle","Powertrain","Fuel","Stock","VKTpVeh","MJpKM"]
out = (pd.DataFrame(rows_out, columns=cols)
         .sort_values(["Subregion","SubregionName","CY","Vehicle","Powertrain","Fuel"])
         .reset_index(drop=True))

# === write two files ===
# 1) with totals
out_with_totals = out.copy()
out_with_totals.to_excel(OUT_WITH, index=False)

# 2) no totals + per-1000
totals = (out[out["Powertrain"]=="total"][["Subregion","SubregionName","CY","Vehicle","Stock"]]
          .rename(columns={"Stock":"TotalStock"}))
no_totals = (out[out["Powertrain"]!="total"]
             .merge(totals, on=["Subregion","SubregionName","CY","Vehicle"], how="left"))
no_totals["stock_per_1000"] = (no_totals["Stock"] / no_totals["TotalStock"]) * 1000
no_totals = no_totals[["ISO","Subregion","SubregionName","CY","Vehicle","Powertrain","Fuel","Stock","VKTpVeh","MJpKM","stock_per_1000"]]
no_totals.to_excel(OUT_NO, index=False)

print(f"Saved:\n - With totals: {OUT_WITH}\n - No totals (+stock_per_1000): {OUT_NO}")