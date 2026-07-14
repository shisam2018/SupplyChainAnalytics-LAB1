"""Model 4 — Supplier Quality & Risk Analytics (Titagarh Auto, industrial version).

For 8 vendors x 52 weeks of incoming inspection: per-vendor p-charts with
out-of-control detection, pairwise two-proportion z-tests vs the best vendor,
one-sample lead-time tests against the contractual 5-day promise, and a
composite vendor scorecard.
Run:  python model_4_supplier_quality.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)
LT_PROMISE = 5.0


def load_input():
    xls = BASE / "inputs" / "input_4_supplier_quality.xlsx"
    return pd.read_excel(xls, sheet_name="Inspections"), pd.read_excel(xls, sheet_name="LeadTimes")


def run():
    insp, lts = load_input()
    insp = insp.copy(); insp["p_hat"] = insp["defects"] / insp["sample_size"]

    # ---- per-vendor SPC: p-chart limits with average sample size
    spc_rows = []
    for v, g in insp.groupby("vendor"):
        pbar = g["defects"].sum() / g["sample_size"].sum()
        nbar = g["sample_size"].mean()
        sig = np.sqrt(pbar * (1 - pbar) / nbar)
        ucl, lcl = pbar + 3 * sig, max(pbar - 3 * sig, 0)
        viol = int(((g["p_hat"] > ucl) | (g["p_hat"] < lcl)).sum())
        # simple trend rule: 8 consecutive points above pbar
        above = (g.sort_values("week")["p_hat"] > pbar).astype(int)
        run8 = int((above.rolling(8).sum() == 8).any())
        spc_rows.append([v, round(pbar, 4), round(ucl, 4), round(lcl, 4), viol, "Yes" if run8 else "No"])
    spc = pd.DataFrame(spc_rows, columns=["vendor", "p_bar", "UCL", "LCL",
                                          "points_beyond_limits", "8_point_upward_run"])

    # ---- two-proportion z-test of each vendor vs the best (lowest p_bar)
    agg = insp.groupby("vendor").agg(n=("sample_size", "sum"), d=("defects", "sum"))
    agg["p_hat"] = agg["d"] / agg["n"]
    best = agg["p_hat"].idxmin()
    zrows = []
    for v in agg.index:
        if v == best:
            zrows.append([v, round(agg.loc[v, "p_hat"], 4), "-", "-", "benchmark"])
            continue
        n1, d1 = agg.loc[v, "n"], agg.loc[v, "d"]
        n2, d2 = agg.loc[best, "n"], agg.loc[best, "d"]
        pp = (d1 + d2) / (n1 + n2)
        se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        z = (agg.loc[v, "p_hat"] - agg.loc[best, "p_hat"]) / se
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        zrows.append([v, round(agg.loc[v, "p_hat"], 4), round(float(z), 2),
                      round(float(pval), 4), "worse than best" if pval < 0.05 else "not distinguishable"])
    ztests = pd.DataFrame(zrows, columns=["vendor", "defect_rate", "z_vs_best",
                                          "p_value", "verdict_at_5pct"])

    # ---- lead-time: one-sided t-test vs the 5-day contractual promise
    lrows = []
    for v, g in lts.groupby("vendor"):
        t, p = stats.ttest_1samp(g["lead_time_days"], LT_PROMISE)
        p_one = p / 2 if t > 0 else 1 - p / 2
        lrows.append([v, round(g["lead_time_days"].mean(), 2), round(g["lead_time_days"].std(), 2),
                      round(float(t), 2), round(float(p_one), 4),
                      "breaches promise" if p_one < 0.05 else "consistent with promise"])
    ltt = pd.DataFrame(lrows, columns=["vendor", "mean_lead_time", "std_dev",
                                       "t_stat", "one_sided_p", "verdict_at_5pct"])

    # ---- composite scorecard (0-100): quality 60%, delivery 40%
    sc = agg[["p_hat"]].join(ltt.set_index("vendor")["mean_lead_time"])
    sc["quality_score"] = (100 * (1 - (sc["p_hat"] - sc["p_hat"].min())
                                  / (sc["p_hat"].max() - sc["p_hat"].min()))).round(1)
    sc["delivery_score"] = (100 * (1 - (sc["mean_lead_time"] - sc["mean_lead_time"].min())
                                   / (sc["mean_lead_time"].max() - sc["mean_lead_time"].min()))).round(1)
    sc["composite"] = (0.6 * sc["quality_score"] + 0.4 * sc["delivery_score"]).round(1)
    sc["tier"] = pd.cut(sc["composite"], [-1, 40, 70, 101], labels=["Exit/PIP", "Develop", "Preferred"])
    scorecard = sc.sort_values("composite", ascending=False).reset_index().rename(
        columns={"p_hat": "defect_rate"})

    # ---- chart: p-chart for the worst vendor + scorecard bars
    worst = agg["p_hat"].idxmax()
    g = insp[insp["vendor"] == worst].sort_values("week")
    row = spc.set_index("vendor").loc[worst]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(g["week"], g["p_hat"], marker="o", ms=3, lw=1, color="#1f4e9c")
    for yv, c, lbl in [(row["p_bar"], "#0e9f8a", "p-bar"), (row["UCL"], "#d97706", "UCL"),
                       (row["LCL"], "#d97706", "LCL")]:
        axes[0].axhline(yv, color=c, lw=1.4, ls="--" if lbl != "p-bar" else "-", label=lbl)
    axes[0].set_title(f"p-chart — {worst}"); axes[0].set_xlabel("Week"); axes[0].legend(fontsize=8)
    axes[1].barh(scorecard["vendor"], scorecard["composite"],
                 color=["#0e9f8a" if t == "Preferred" else "#d97706" if t == "Develop" else "#c2410c"
                        for t in scorecard["tier"]])
    axes[1].set_title("Composite vendor scorecard (0-100)")
    fig.tight_layout()

    return {"input": insp, "spc": spc, "ztests": ztests, "leadtime_tests": ltt,
            "scorecard": scorecard, "figure": fig}


def save_outputs(res):
    with pd.ExcelWriter(OUT / "output_4_vendor_scorecard.xlsx") as xw:
        res["scorecard"].to_excel(xw, sheet_name="Scorecard", index=False)
        res["spc"].to_excel(xw, sheet_name="SPC_pcharts", index=False)
        res["ztests"].to_excel(xw, sheet_name="Proportion_Tests", index=False)
        res["leadtime_tests"].to_excel(xw, sheet_name="LeadTime_Tests", index=False)
    res["figure"].savefig(OUT / "output_4_quality_charts.png", dpi=150)
    print("Model 4 done ->", OUT / "output_4_vendor_scorecard.xlsx")
    print(res["scorecard"].to_string(index=False))


if __name__ == "__main__":
    save_outputs(run())
