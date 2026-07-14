"""Model 5 — Freight-Cost Drivers & Quote Engine (Sundarban Shippers, industrial version).

Multiple regression (statsmodels OLS) for interpretability — every coefficient
is a costing lever — benchmarked against a Random Forest for pure accuracy,
plus an instant-quote function for new lanes.
Run:  python model_5_freight_costs.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)


def load_input():
    return pd.read_excel(BASE / "inputs" / "input_5_freight_costs.xlsx", sheet_name="Shipments")


def design(df):
    X = pd.get_dummies(df[["distance_km", "weight_tonnes", "fuel_index", "toll_plazas",
                           "vehicle_type", "region", "monsoon_flag"]],
                       columns=["vehicle_type", "region"], drop_first=True).astype(float)
    return X


def run():
    df = load_input()
    X, y = design(df), df["total_cost_inr"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)

    ols = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
    ols_pred = ols.predict(sm.add_constant(Xte))
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, random_state=7).fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)

    def scores(a, f):
        a, f = np.asarray(a, float), np.asarray(f, float)
        ss = 1 - np.sum((a - f) ** 2) / np.sum((a - a.mean()) ** 2)
        return round(float(ss), 3), round(float(np.sqrt(np.mean((a - f) ** 2))), 0), \
            round(float(np.mean(np.abs(a - f) / a) * 100), 2)

    m1, m2 = scores(yte, ols_pred), scores(yte, rf_pred)
    metrics = pd.DataFrame({"model": ["OLS multiple regression", "Random Forest"],
                            "R2_test": [m1[0], m2[0]], "RMSE_inr": [m1[1], m2[1]],
                            "MAPE_pct": [m1[2], m2[2]]})

    coefs = pd.DataFrame({"driver": ols.params.index, "coefficient_inr": ols.params.round(1).values,
                          "p_value": ols.pvalues.round(4).values})
    coefs["significant_5pct"] = np.where(coefs["p_value"] < 0.05, "Yes", "No")

    # quote engine on three illustrative new lanes
    lanes = pd.DataFrame({
        "lane": ["Kolkata-Bhubaneswar", "Kolkata-Guwahati", "Kolkata-Nagpur"],
        "distance_km": [440, 990, 1130], "weight_tonnes": [14.0, 18.5, 21.0],
        "fuel_index": [104.0, 104.0, 104.0], "toll_plazas": [4, 7, 9],
        "vehicle_type": ["16T", "24T", "24T"], "region": ["East", "East", "West"],
        "monsoon_flag": [0, 1, 0]})
    Xq = design(lanes).reindex(columns=X.columns, fill_value=0.0)
    lanes["OLS_quote_inr"] = ols.predict(sm.add_constant(Xq, has_constant="add")).round(0).astype(int)
    lanes["RF_quote_inr"] = rf.predict(Xq).round(0).astype(int)
    quotes = lanes[["lane", "distance_km", "weight_tonnes", "vehicle_type",
                    "monsoon_flag", "OLS_quote_inr", "RF_quote_inr"]]

    resid = yte.values - np.asarray(ols_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(ols_pred, yte, s=12, color="#1f4e9c", alpha=0.6)
    lims = [min(yte.min(), ols_pred.min()), max(yte.max(), ols_pred.max())]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_title(f"OLS: predicted vs actual (R2 = {m1[0]})")
    axes[0].set_xlabel("Predicted cost (Rs)"); axes[0].set_ylabel("Actual cost (Rs)")
    axes[1].scatter(ols_pred, resid, s=12, color="#0e9f8a", alpha=0.6)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("Residuals vs fitted — pattern check"); axes[1].set_xlabel("Fitted (Rs)")
    fig.tight_layout()

    return {"input": df, "metrics": metrics, "coefficients": coefs, "quotes": quotes,
            "figure": fig, "ols_summary": str(ols.summary())}


def save_outputs(res):
    with pd.ExcelWriter(OUT / "output_5_freight_model.xlsx") as xw:
        res["metrics"].to_excel(xw, sheet_name="Model_Comparison", index=False)
        res["coefficients"].to_excel(xw, sheet_name="OLS_Cost_Drivers", index=False)
        res["quotes"].to_excel(xw, sheet_name="Lane_Quotes", index=False)
    (OUT / "output_5_ols_summary.txt").write_text(res["ols_summary"])
    res["figure"].savefig(OUT / "output_5_freight_charts.png", dpi=150)
    print("Model 5 done ->", OUT / "output_5_freight_model.xlsx")
    print(res["metrics"].to_string(index=False))


if __name__ == "__main__":
    save_outputs(run())
