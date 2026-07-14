"""Model 2 — Multi-SKU Inventory Optimization (Hooghly Electronics, industrial version).

For a 40-SKU portfolio: EOQ, safety stock and reorder point per SKU under its
own service target; ABC classification on annual value; and a Monte-Carlo
simulation that verifies the achieved cycle-service level of the recommended
policy for the top-value SKU.
Run:  python model_2_inventory_optimization.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)
DAYS = 360


def load_input():
    return pd.read_excel(BASE / "inputs" / "input_2_inventory_optimization.xlsx", sheet_name="SKUs")


def run():
    df = load_input().copy()
    df["H_inr"] = df["unit_cost_inr"] * df["holding_rate_pct"] / 100
    df["EOQ"] = np.sqrt(2 * df["annual_demand_units"] * df["ordering_cost_inr"] / df["H_inr"]).round(0)
    df["orders_per_year"] = (df["annual_demand_units"] / df["EOQ"]).round(1)
    df["daily_demand"] = df["annual_demand_units"] / DAYS
    df["sigma_LT"] = df["daily_demand_std"] * np.sqrt(df["lead_time_days"])
    df["z"] = norm.ppf(df["service_level_target"])
    df["safety_stock"] = np.ceil(df["z"] * df["sigma_LT"])
    df["reorder_point"] = np.ceil(df["daily_demand"] * df["lead_time_days"] + df["safety_stock"])
    df["annual_ordering_cost"] = (df["annual_demand_units"] / df["EOQ"] * df["ordering_cost_inr"]).round(0)
    df["annual_holding_cost"] = ((df["EOQ"] / 2 + df["safety_stock"]) * df["H_inr"]).round(0)
    df["annual_value_inr"] = df["annual_demand_units"] * df["unit_cost_inr"]

    # ABC on annual value
    d = df.sort_values("annual_value_inr", ascending=False).reset_index(drop=True)
    d["cum_value_share"] = d["annual_value_inr"].cumsum() / d["annual_value_inr"].sum()
    d["abc_class"] = np.where(d["cum_value_share"] <= 0.70, "A",
                       np.where(d["cum_value_share"] <= 0.90, "B", "C"))
    plan = d[["sku_id", "category", "abc_class", "annual_demand_units", "unit_cost_inr",
              "annual_value_inr", "EOQ", "orders_per_year", "safety_stock", "reorder_point",
              "annual_ordering_cost", "annual_holding_cost", "service_level_target"]]

    abc_summary = (plan.groupby("abc_class")
                   .agg(skus=("sku_id", "count"),
                        value_share_pct=("annual_value_inr", lambda s: round(100 * s.sum() / plan["annual_value_inr"].sum(), 1)),
                        total_holding_inr=("annual_holding_cost", "sum"))
                   .reset_index())

    # Monte-Carlo check of the top-value SKU's policy (10,000 replenishment cycles)
    top = plan.iloc[0]
    src = df.set_index("sku_id").loc[top["sku_id"]]
    rng = np.random.default_rng(7)
    n_sim = 10_000
    lt_demand = rng.normal(src["daily_demand"] * src["lead_time_days"],
                           src["sigma_LT"], n_sim)
    achieved = float((lt_demand <= top["reorder_point"]).mean())
    sim = pd.DataFrame({"metric": ["SKU simulated", "Target cycle-service level",
                                    "Achieved (10,000 simulated cycles)",
                                    "Reorder point (units)", "Safety stock (units)"],
                        "value": [top["sku_id"], f'{src["service_level_target"]:.0%}',
                                  f"{achieved:.1%}", int(top["reorder_point"]), int(top["safety_stock"])]})

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    pv = plan.sort_values("annual_value_inr", ascending=False).reset_index(drop=True)
    colors = pv["abc_class"].map({"A": "#1f4e9c", "B": "#0e9f8a", "C": "#9aa7b5"})
    axes[0].bar(range(len(pv)), pv["annual_value_inr"] / 1e5, color=colors)
    axes[0].set_title("Annual value by SKU (ABC)"); axes[0].set_ylabel("Rs lakh"); axes[0].set_xlabel("SKUs (ranked)")
    axes[1].hist(lt_demand, bins=40, color="#0e9f8a", alpha=0.8)
    axes[1].axvline(top["reorder_point"], color="#d97706", lw=2, label="Reorder point")
    axes[1].set_title(f'Lead-time demand vs ROP — {top["sku_id"]}'); axes[1].legend(fontsize=8)
    fig.tight_layout()

    return {"input": df, "plan": plan, "abc_summary": abc_summary, "simulation": sim, "figure": fig}


def save_outputs(res):
    with pd.ExcelWriter(OUT / "output_2_inventory_plan.xlsx") as xw:
        res["plan"].to_excel(xw, sheet_name="Replenishment_Plan", index=False)
        res["abc_summary"].to_excel(xw, sheet_name="ABC_Summary", index=False)
        res["simulation"].to_excel(xw, sheet_name="MonteCarlo_Check", index=False)
    res["figure"].savefig(OUT / "output_2_inventory_charts.png", dpi=150)
    print("Model 2 done ->", OUT / "output_2_inventory_plan.xlsx")
    print(res["abc_summary"].to_string(index=False))


if __name__ == "__main__":
    save_outputs(run())
