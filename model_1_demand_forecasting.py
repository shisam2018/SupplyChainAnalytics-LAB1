"""Model 1 — Demand Forecasting at Scale (Bengal Basket, industrial version).

Compares a classical Holt-Winters seasonal model against a machine-learning
Gradient Boosting model built on lag/calendar features, backtested on the
final 26 weeks, then produces a 12-week forward forecast.
Run:  python model_1_demand_forecasting.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)
HOLDOUT, HORIZON, SEASON = 26, 12, 52


def load_input():
    return pd.read_excel(BASE / "inputs" / "input_1_demand_forecasting.xlsx", sheet_name="Demand")


def make_features(df):
    f = df.copy()
    for lag in (1, 2, 4, 13, 52):
        f[f"lag_{lag}"] = f["demand_cartons"].shift(lag)
    f["roll_mean_4"] = f["demand_cartons"].shift(1).rolling(4).mean()
    f["roll_mean_13"] = f["demand_cartons"].shift(1).rolling(13).mean()
    f["week_of_year"] = f["week_no"] % SEASON
    f["sin_w"] = np.sin(2 * np.pi * f["week_of_year"] / SEASON)
    f["cos_w"] = np.cos(2 * np.pi * f["week_of_year"] / SEASON)
    return f.dropna().reset_index(drop=True)


def mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.mean(np.abs(a - f) / a) * 100)


def run():
    df = load_input()
    y = df["demand_cartons"].astype(float)
    train_y, test_y = y[:-HOLDOUT], y[-HOLDOUT:]

    # -- classical benchmark: Holt-Winters additive trend + seasonality
    hw = ExponentialSmoothing(train_y, trend="add", seasonal="add",
                              seasonal_periods=SEASON).fit()
    hw_test = np.asarray(hw.forecast(HOLDOUT))

    # -- ML challenger: gradient boosting on lag / calendar / promo features
    feats = make_features(df)
    Xcols = [c for c in feats.columns if c.startswith(("lag_", "roll_", "sin", "cos"))] + \
            ["promo_flag", "festival_flag"]
    cut = len(feats) - HOLDOUT
    gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                   max_depth=3, subsample=0.9, random_state=7)
    gb.fit(feats.loc[:cut - 1, Xcols], feats.loc[:cut - 1, "demand_cartons"])
    gb_test = gb.predict(feats.loc[cut:, Xcols])

    metrics = pd.DataFrame({
        "model": ["Holt-Winters (classical)", "Gradient Boosting (ML)"],
        "MAPE_holdout_pct": [round(mape(test_y, hw_test), 2), round(mape(test_y, gb_test), 2)],
        "RMSE_holdout": [round(float(np.sqrt(np.mean((test_y - hw_test) ** 2))), 1),
                          round(float(np.sqrt(np.mean((test_y - gb_test) ** 2))), 1)],
    })

    # -- refit on full history and forecast forward (HW for the forward path;
    #    recursive ML forecasting is left as a class discussion point)
    hw_full = ExponentialSmoothing(y, trend="add", seasonal="add",
                                   seasonal_periods=SEASON).fit()
    fwd = np.asarray(hw_full.forecast(HORIZON)).round(0)
    resid_sd = float(np.std(y - hw_full.fittedvalues))
    fut = pd.DataFrame({
        "week_no": np.arange(df["week_no"].iloc[-1] + 1, df["week_no"].iloc[-1] + 1 + HORIZON),
        "forecast_cartons": fwd.astype(int),
        "lo_95": (fwd - 1.96 * resid_sd).round(0).astype(int),
        "hi_95": (fwd + 1.96 * resid_sd).round(0).astype(int),
    })

    backtest = pd.DataFrame({"week_no": df["week_no"].iloc[-HOLDOUT:].values,
                             "actual": test_y.values,
                             "holt_winters": hw_test.round(1),
                             "gradient_boosting": gb_test.round(1)})
    fi = pd.DataFrame({"feature": Xcols, "importance": gb.feature_importances_.round(4)}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["week_no"], y, lw=1, color="#7a8aa0", label="History")
    ax.plot(backtest["week_no"], backtest["holt_winters"], lw=1.6, color="#d97706", label="HW backtest")
    ax.plot(backtest["week_no"], backtest["gradient_boosting"], lw=1.6, color="#0e9f8a", label="GB backtest")
    ax.plot(fut["week_no"], fut["forecast_cartons"], lw=2, color="#1f4e9c", label="12-wk forecast")
    ax.fill_between(fut["week_no"], fut["lo_95"], fut["hi_95"], color="#1f4e9c", alpha=0.15)
    ax.set_xlabel("Week"); ax.set_ylabel("Cartons"); ax.legend(fontsize=8); ax.set_title("Demand: history, backtest and forecast")
    fig.tight_layout()

    return {"input": df, "metrics": metrics, "backtest": backtest,
            "forecast": fut, "feature_importance": fi, "figure": fig}


def save_outputs(res):
    with pd.ExcelWriter(OUT / "output_1_demand_forecasting.xlsx") as xw:
        res["metrics"].to_excel(xw, sheet_name="Model_Comparison", index=False)
        res["backtest"].to_excel(xw, sheet_name="Backtest_26wk", index=False)
        res["forecast"].to_excel(xw, sheet_name="Forecast_12wk", index=False)
        res["feature_importance"].to_excel(xw, sheet_name="GB_Feature_Importance", index=False)
    res["figure"].savefig(OUT / "output_1_forecast_chart.png", dpi=150)
    print("Model 1 done ->", OUT / "output_1_demand_forecasting.xlsx")
    print(res["metrics"].to_string(index=False))


if __name__ == "__main__":
    save_outputs(run())
