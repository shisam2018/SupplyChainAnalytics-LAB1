"""Model 6 — SKU Segmentation & Market-Basket Mining (Maidan Marts, industrial version).

k-means on 60 SKUs across four standardized features, with elbow and
silhouette diagnostics to choose k, plus association-rule mining (support /
confidence / lift, computed directly with pandas) on 500 POS transactions.
Run:  python model_6_sku_clustering_basket.py
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)
FEATS = ["weekly_picks", "unit_value_inr", "demand_cv", "lead_time_days"]


def load_input():
    xls = BASE / "inputs" / "input_6_sku_basket.xlsx"
    return pd.read_excel(xls, sheet_name="SKU_Profile"), pd.read_excel(xls, sheet_name="Transactions")


def run():
    prof, txn = load_input()
    Xs = StandardScaler().fit_transform(prof[FEATS])

    ks = range(2, 8)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    diag = pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 1),
                         "silhouette": np.round(sil, 3)})
    best_k = int(diag.loc[diag["silhouette"].idxmax(), "k"])

    km = KMeans(n_clusters=best_k, n_init=10, random_state=7).fit(Xs)
    prof = prof.copy(); prof["cluster"] = km.labels_
    centers = prof.groupby("cluster")[FEATS].mean().round(1)
    # name clusters by picks level for managerial readability
    order = centers["weekly_picks"].sort_values().index.tolist()
    names = {order[0]: "C-class: slow & steady"}
    if len(order) >= 2: names[order[-1]] = "A-class: fast & valuable"
    for c in order[1:-1]:
        names[c] = "B-class: mid movers"
    prof["segment"] = prof["cluster"].map(names)
    centers = centers.reset_index()
    centers["segment"] = centers["cluster"].map(names)
    seg_sizes = prof.groupby("segment")["sku_id"].count().rename("skus").reset_index()

    # ---- association rules on item pairs (pure pandas — no extra dependency)
    baskets = txn.groupby("txn_id")["item"].apply(set)
    n_tx = len(baskets)
    items = sorted(txn["item"].unique())
    supp1 = {i: sum(i in b for b in baskets) / n_tx for i in items}
    rows = []
    for a, b in combinations(items, 2):
        s_ab = sum((a in bx) and (b in bx) for bx in baskets) / n_tx
        if s_ab < 0.05:
            continue
        for x, yv in [(a, b), (b, a)]:
            conf = s_ab / supp1[x]
            lift = s_ab / (supp1[x] * supp1[yv])
            rows.append([f"{x} -> {yv}", round(supp1[x], 3), round(supp1[yv], 3),
                         round(s_ab, 3), round(conf, 3), round(lift, 3)])
    rules = pd.DataFrame(rows, columns=["rule", "supp_antecedent", "supp_consequent",
                                        "supp_pair", "confidence", "lift"])
    rules = rules[rules["lift"] > 1.05].sort_values(["lift", "confidence"],
                                                    ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(diag["k"], diag["inertia"], marker="o", color="#1f4e9c")
    axes[0].set_title("Elbow (inertia)"); axes[0].set_xlabel("k")
    axes[1].plot(diag["k"], diag["silhouette"], marker="o", color="#0e9f8a")
    axes[1].axvline(best_k, color="#d97706", ls="--", lw=1.4)
    axes[1].set_title(f"Silhouette (best k = {best_k})"); axes[1].set_xlabel("k")
    palette = {s: c for s, c in zip(sorted(prof["segment"].unique()),
                                    ["#1f4e9c", "#0e9f8a", "#d97706", "#c2410c", "#7c3aed", "#64748b"])}
    for s, g in prof.groupby("segment"):
        axes[2].scatter(g["weekly_picks"], g["unit_value_inr"], s=26, label=s, color=palette[s], alpha=0.85)
    axes[2].set_xlabel("Weekly picks"); axes[2].set_ylabel("Unit value (Rs)")
    axes[2].set_title("SKU segments"); axes[2].legend(fontsize=7)
    fig.tight_layout()

    return {"profile": prof, "diagnostics": diag, "centers": centers,
            "segment_sizes": seg_sizes, "rules": rules.head(12), "figure": fig, "best_k": best_k}


def save_outputs(res):
    with pd.ExcelWriter(OUT / "output_6_segments_rules.xlsx") as xw:
        res["profile"].to_excel(xw, sheet_name="SKU_Segments", index=False)
        res["centers"].to_excel(xw, sheet_name="Cluster_Centers", index=False)
        res["diagnostics"].to_excel(xw, sheet_name="k_Diagnostics", index=False)
        res["rules"].to_excel(xw, sheet_name="Association_Rules", index=False)
    res["figure"].savefig(OUT / "output_6_cluster_charts.png", dpi=150)
    print(f"Model 6 done (best k = {res['best_k']}) ->", OUT / "output_6_segments_rules.xlsx")
    print(res["rules"].head(6).to_string(index=False))


if __name__ == "__main__":
    save_outputs(run())
