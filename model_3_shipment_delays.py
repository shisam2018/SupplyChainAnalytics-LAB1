"""Model 3 — Predicting Shipment Delays (GangaFreight, industrial version).

Binary classification on 600 shipments: interpretable Logistic Regression
benchmark vs a Random Forest challenger; confusion matrix, precision/recall,
ROC-AUC, feature importance, and a scored file of at-risk live shipments.
Run:  python model_3_shipment_delays.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, roc_curve)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)
NUM = ["distance_km", "congestion_index", "truck_age_yrs", "border_toll_halts", "load_tonnes", "festival_window"]
CAT = ["weather", "carrier"]


def load_input():
    return pd.read_excel(BASE / "inputs" / "input_3_shipment_delays.xlsx", sheet_name="Shipments")


def run():
    df = load_input()
    X, y = df[NUM + CAT], df["delayed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)

    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first"), CAT)])
    logit = Pipeline([("prep", prep),
                      ("clf", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
    rf = Pipeline([("prep", ColumnTransformer([("num", "passthrough", NUM),
                                               ("cat", OneHotEncoder(drop="first"), CAT)])),
                   ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                  random_state=7))]).fit(Xtr, ytr)

    rows, roc_data, cms = [], {}, {}
    for name, mdl in [("Logistic Regression", logit), ("Random Forest", rf)]:
        proba = mdl.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cms[name] = confusion_matrix(yte, pred)
        fpr, tpr, _ = roc_curve(yte, proba)
        roc_data[name] = (fpr, tpr)
        rows.append([name, round(accuracy_score(yte, pred), 3), round(precision_score(yte, pred), 3),
                     round(recall_score(yte, pred), 3), round(f1_score(yte, pred), 3),
                     round(roc_auc_score(yte, proba), 3)])
    metrics = pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "F1", "ROC_AUC"])

    tn, fp, fn, tp = cms["Random Forest"].ravel()
    cm_df = pd.DataFrame({"": ["Actual: On-time", "Actual: Delayed"],
                          "Pred: On-time": [tn, fn], "Pred: Delayed": [fp, tp]})

    # odds ratios from the logistic model — the interpretability layer
    feat_names = (NUM + list(logit.named_steps["prep"].named_transformers_["cat"]
                             .get_feature_names_out(CAT)))
    odds = pd.DataFrame({"feature": feat_names,
                         "odds_ratio": np.exp(logit.named_steps["clf"].coef_[0]).round(3)}) \
        .sort_values("odds_ratio", ascending=False).reset_index(drop=True)

    rf_names = (NUM + list(rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT)))
    imp = pd.DataFrame({"feature": rf_names,
                        "importance": rf.named_steps["clf"].feature_importances_.round(4)}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    # score the full book and surface the riskiest 15 shipments
    df_sc = df.copy()
    df_sc["delay_risk"] = rf.predict_proba(X)[:, 1].round(3)
    watchlist = df_sc.sort_values("delay_risk", ascending=False).head(15)[
        ["shipment_id", "distance_km", "weather", "carrier", "border_toll_halts",
         "festival_window", "delay_risk"]].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, (fpr, tpr) in roc_data.items():
        axes[0].plot(fpr, tpr, lw=2, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC curves (test set)"); axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate"); axes[0].legend(fontsize=8)
    top = imp.head(8).iloc[::-1]
    axes[1].barh(top["feature"], top["importance"], color="#1f4e9c")
    axes[1].set_title("Random-Forest feature importance")
    fig.tight_layout()

    return {"input": df, "metrics": metrics, "confusion_rf": cm_df, "odds_ratios": odds,
            "importance": imp, "watchlist": watchlist, "figure": fig}


def save_outputs(res):
    with pd.ExcelWriter(OUT / "output_3_delay_model.xlsx") as xw:
        res["metrics"].to_excel(xw, sheet_name="Model_Comparison", index=False)
        res["confusion_rf"].to_excel(xw, sheet_name="RF_Confusion_Matrix", index=False)
        res["odds_ratios"].to_excel(xw, sheet_name="Logit_Odds_Ratios", index=False)
        res["importance"].to_excel(xw, sheet_name="RF_Importance", index=False)
        res["watchlist"].to_excel(xw, sheet_name="Risk_Watchlist", index=False)
    res["figure"].savefig(OUT / "output_3_delay_charts.png", dpi=150)
    print("Model 3 done ->", OUT / "output_3_delay_model.xlsx")
    print(res["metrics"].to_string(index=False))


if __name__ == "__main__":
    save_outputs(run())
