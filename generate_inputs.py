"""Generate the six Excel input files for the IIMC Supply-Chain ML Lab."""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
IN = Path(__file__).resolve().parent / "inputs"
IN.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. demand
def gen_demand():
    weeks = 156  # 3 years
    t = np.arange(1, weeks + 1)
    dates = pd.date_range("2023-07-03", periods=weeks, freq="W-MON")
    trend = 400 + 1.8 * t
    season = 60 * np.sin(2 * np.pi * (t % 52) / 52) + 25 * np.sin(2 * np.pi * (t % 13) / 13)
    festival = np.zeros(weeks)
    for yr in range(3):                       # Puja/Diwali spike ~ week 14-17 of Oct block
        festival[yr * 52 + 40: yr * 52 + 44] = 1
    promo = (rng.random(weeks) < 0.12).astype(int)
    noise = rng.normal(0, 28, weeks)
    demand = trend + season + 140 * festival + 70 * promo + noise
    demand = np.maximum(demand, 60).round(0)
    df = pd.DataFrame({"week_start": dates.date, "week_no": t, "demand_cartons": demand.astype(int),
                       "promo_flag": promo, "festival_flag": festival.astype(int)})
    df.to_excel(IN / "input_1_demand_forecasting.xlsx", sheet_name="Demand", index=False)

# ---------------------------------------------------------------- 2. inventory
def gen_inventory():
    n = 40
    cats = rng.choice(["Bearings", "Filters", "Belts", "Electricals", "Fasteners"], n)
    unit_cost = np.round(rng.lognormal(5.4, 0.9, n), 0)             # ~ Rs 80 - 3000
    annual_demand = np.round(rng.lognormal(7.6, 0.8, n), 0)         # ~ 600 - 15000
    df = pd.DataFrame({
        "sku_id": [f"SKU-{i:03d}" for i in range(1, n + 1)],
        "category": cats,
        "annual_demand_units": annual_demand.astype(int),
        "unit_cost_inr": unit_cost.astype(int),
        "ordering_cost_inr": rng.choice([300, 400, 500, 600], n),
        "holding_rate_pct": rng.choice([18, 20, 22, 25], n),
        "lead_time_days": rng.integers(4, 22, n),
        "daily_demand_std": np.round(annual_demand / 360 * rng.uniform(0.25, 0.6, n), 1),
        "service_level_target": rng.choice([0.90, 0.95, 0.99], n, p=[0.3, 0.5, 0.2]),
    })
    df.to_excel(IN / "input_2_inventory_optimization.xlsx", sheet_name="SKUs", index=False)

# ---------------------------------------------------------------- 3. delays
def gen_delays():
    n = 600
    distance = rng.integers(80, 1600, n)
    congestion = np.round(rng.uniform(1, 10, n), 1)
    weather = rng.choice(["Clear", "Rain", "Fog"], n, p=[0.62, 0.28, 0.10])
    truck_age = rng.integers(1, 15, n)
    halts = rng.integers(0, 9, n)
    load = np.round(rng.uniform(4, 24, n), 1)
    carrier = rng.choice(["CarrierA", "CarrierB", "CarrierC"], n, p=[0.45, 0.35, 0.20])
    festival = (rng.random(n) < 0.15).astype(int)
    logit = (-4.4 + 0.0016 * distance + 0.16 * congestion
             + 0.85 * (weather == "Rain") + 1.5 * (weather == "Fog")
             + 0.09 * truck_age + 0.16 * halts + 0.02 * load
             + 0.55 * (carrier == "CarrierC") + 0.7 * festival)
    p = 1 / (1 + np.exp(-logit))
    delayed = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"shipment_id": [f"SH-{i:04d}" for i in range(1, n + 1)],
                       "distance_km": distance, "congestion_index": congestion, "weather": weather,
                       "truck_age_yrs": truck_age, "border_toll_halts": halts, "load_tonnes": load,
                       "carrier": carrier, "festival_window": festival, "delayed": delayed})
    df.to_excel(IN / "input_3_shipment_delays.xlsx", sheet_name="Shipments", index=False)

# ---------------------------------------------------------------- 4. supplier quality
def gen_supplier():
    vendors = [f"Vendor-{c}" for c in "ABCDEFGH"]
    base_p = dict(zip(vendors, [0.045, 0.05, 0.055, 0.06, 0.07, 0.09, 0.11, 0.13]))
    rows = []
    for v in vendors:
        drift = 0.0009 if v in ("Vendor-G", "Vendor-H") else 0.0
        for w in range(1, 53):
            nsz = int(rng.integers(80, 160))
            p = min(base_p[v] + drift * w + rng.normal(0, 0.004), 0.35)
            rows.append([w, v, nsz, rng.binomial(nsz, max(p, 0.005))])
    insp = pd.DataFrame(rows, columns=["week", "vendor", "sample_size", "defects"])
    lt_rows = []
    mean_lt = dict(zip(vendors, [4.6, 5.0, 5.2, 5.4, 5.8, 6.3, 6.8, 7.4]))
    for v in vendors:
        for s in range(60):
            lt_rows.append([v, f"{v[-1]}{s:03d}", max(round(rng.normal(mean_lt[v], 1.5), 1), 1.0)])
    lts = pd.DataFrame(lt_rows, columns=["vendor", "shipment_ref", "lead_time_days"])
    with pd.ExcelWriter(IN / "input_4_supplier_quality.xlsx") as xw:
        insp.to_excel(xw, sheet_name="Inspections", index=False)
        lts.to_excel(xw, sheet_name="LeadTimes", index=False)

# ---------------------------------------------------------------- 5. freight cost
def gen_freight():
    n = 300
    distance = rng.integers(60, 2100, n)
    weight = np.round(rng.uniform(3, 25, n), 1)
    fuel = np.round(rng.uniform(92, 118, n), 1)          # diesel index
    tolls = rng.integers(0, 13, n)
    vehicle = rng.choice(["10T", "16T", "24T"], n, p=[0.35, 0.4, 0.25])
    region = rng.choice(["East", "North", "West", "South"], n)
    monsoon = (rng.random(n) < 0.25).astype(int)
    veh_add = np.select([vehicle == "10T", vehicle == "16T"], [0, 2500], 5500)
    cost = (9000 + 26 * distance + 240 * weight + 55 * (fuel - 100) * distance / 1000 * 10
            + 320 * tolls + veh_add + 1400 * monsoon + rng.normal(0, 1800, n))
    df = pd.DataFrame({"shipment_id": [f"FR-{i:04d}" for i in range(1, n + 1)],
                       "distance_km": distance, "weight_tonnes": weight, "fuel_index": fuel,
                       "toll_plazas": tolls, "vehicle_type": vehicle, "region": region,
                       "monsoon_flag": monsoon, "total_cost_inr": np.round(cost, 0).astype(int)})
    df.to_excel(IN / "input_5_freight_costs.xlsx", sheet_name="Shipments", index=False)

# ---------------------------------------------------------------- 6. sku + basket
def gen_sku_basket():
    n = 60
    seg = rng.choice([0, 1, 2], n, p=[0.45, 0.35, 0.20])   # latent: slow/medium/fast
    picks = np.where(seg == 0, rng.uniform(2, 25, n), np.where(seg == 1, rng.uniform(30, 90, n), rng.uniform(100, 240, n)))
    value = np.where(seg == 0, rng.uniform(150, 900, n), np.where(seg == 1, rng.uniform(400, 2500, n), rng.uniform(900, 6000, n)))
    cv = np.round(np.where(seg == 2, rng.uniform(0.15, 0.5, n), rng.uniform(0.3, 1.1, n)), 2)
    lead = rng.integers(3, 30, n)
    prof = pd.DataFrame({"sku_id": [f"P-{i:03d}" for i in range(1, n + 1)],
                         "weekly_picks": np.round(picks, 0).astype(int),
                         "unit_value_inr": np.round(value, 0).astype(int),
                         "demand_cv": cv, "lead_time_days": lead})
    # transactions over 8 popular service items with embedded affinities
    items = ["OilFilter", "AirFilter", "FuelFilter", "BrakePad", "BrakeFluid", "WiperBlade", "Coolant", "SparkPlug"]
    rows, tid = [], 0
    for _ in range(500):
        tid += 1
        basket = set()
        if rng.random() < 0.55:
            basket.add("OilFilter")
            if rng.random() < 0.70: basket.add("AirFilter")
            if rng.random() < 0.35: basket.add("FuelFilter")
        if rng.random() < 0.35:
            basket.add("BrakePad")
            if rng.random() < 0.60: basket.add("BrakeFluid")
        for it in ["WiperBlade", "Coolant", "SparkPlug"]:
            if rng.random() < 0.18: basket.add(it)
        if not basket:
            basket.add(rng.choice(items))
        for it in sorted(basket):
            rows.append([f"T-{tid:04d}", it])
    txn = pd.DataFrame(rows, columns=["txn_id", "item"])
    with pd.ExcelWriter(IN / "input_6_sku_basket.xlsx") as xw:
        prof.to_excel(xw, sheet_name="SKU_Profile", index=False)
        txn.to_excel(xw, sheet_name="Transactions", index=False)

if __name__ == "__main__":
    gen_demand(); gen_inventory(); gen_delays(); gen_supplier(); gen_freight(); gen_sku_basket()
    for f in sorted(IN.glob("*.xlsx")):
        print(f.name)
