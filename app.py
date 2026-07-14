"""IIMC Supply-Chain ML Lab — executive Streamlit demo.

Run:  streamlit run app.py
Each use case is presented as Problem -> Method -> Outcome, with the input
data, model results and charts, plus one-click download of the output file.
"""
import io
import pandas as pd
import streamlit as st

import model_1_demand_forecasting as m1
import model_2_inventory_optimization as m2
import model_3_shipment_delays as m3
import model_4_supplier_quality as m4
import model_5_freight_costs as m5
import model_6_sku_clustering_basket as m6

st.set_page_config(page_title="IIMC Supply-Chain ML Lab", page_icon=":truck:", layout="wide")

PAGES = {
    "Overview": None,
    "1 · Demand Forecasting": m1,
    "2 · Inventory Optimization": m2,
    "3 · Shipment Delay Prediction": m3,
    "4 · Supplier Quality Analytics": m4,
    "5 · Freight Cost Drivers": m5,
    "6 · SKU Segments & Market Basket": m6,
}

META = {
    "1 · Demand Forecasting": dict(
        problem="Bengal Basket must commit vendor volumes weeks ahead while weekly demand "
                "carries trend, 52-week seasonality, festival spikes and promotions. Bad "
                "forecasts become stockouts or markdowns.",
        method="Backtest on the last 26 weeks: a classical Holt-Winters model (trend + "
               "seasonality) vs a Gradient-Boosting model on lag, rolling-mean and calendar "
               "features. Winner refit on full history for a 12-week forecast with a 95% band.",
        outcome="A model-comparison scorecard (MAPE/RMSE), a 12-week forward plan with "
                "uncertainty bands, and the ML model's feature-importance ranking."),
    "2 · Inventory Optimization": dict(
        problem="Hooghly Electronics stocks 40 SKUs with very different value, demand and "
                "lead-time profiles. One-size-fits-all ordering ties up cash and still lets "
                "critical parts stock out.",
        method="Per-SKU EOQ, safety stock and reorder point from each SKU's own service "
               "target; ABC classification on annual value; 10,000-cycle Monte-Carlo "
               "simulation to verify the top SKU's achieved service level.",
        outcome="A complete replenishment plan (EOQ / SS / ROP per SKU), the ABC value "
                "summary, and simulation evidence that the policy hits its service promise."),
    "3 · Shipment Delay Prediction": dict(
        problem="GangaFreight pays SLA penalties on late truckloads. Planners need a "
                "dispatch-time early-warning score, not a post-mortem.",
        method="600 historical shipments; interpretable Logistic Regression (odds ratios) "
               "vs Random Forest; stratified train/test split; confusion matrix, "
               "precision/recall trade-off and ROC-AUC; full book re-scored.",
        outcome="A risk watchlist of the 15 most at-risk live shipments, model scorecard, "
                "and the drivers of delay (weather, carrier, congestion, halts)."),
    "4 · Supplier Quality Analytics": dict(
        problem="Titagarh Auto buys castings from 8 vendors. Which are statistically worse, "
                "which are drifting out of control, and who breaches the 5-day lead-time promise?",
        method="Per-vendor p-charts with 3-sigma limits and run rules; two-proportion "
               "z-tests vs the best vendor; one-sided t-tests of lead time vs the 5-day "
               "contract; weighted composite scorecard (quality 60 / delivery 40).",
        outcome="A tiered vendor scorecard (Preferred / Develop / Exit-PIP), SPC violation "
                "log, and statistically defensible evidence for sourcing decisions."),
    "5 · Freight Cost Drivers": dict(
        problem="Sundarban Shippers quotes lanes on gut feel. Which cost drivers actually "
                "matter, what does each toll plaza or tonne really cost, and what should a "
                "new lane be quoted at?",
        method="OLS multiple regression (every coefficient is a costing lever, with "
               "p-values) benchmarked against a Random Forest; residual diagnostics; an "
               "instant-quote engine applied to three new lanes.",
        outcome="A cost-driver table in rupees, model accuracy comparison (R2 / RMSE / "
                "MAPE), and side-by-side OLS vs RF quotes for new business."),
    "6 · SKU Segments & Market Basket": dict(
        problem="Maidan Marts slots 60 spare-part SKUs and wants co-purchased items placed "
                "together. Which SKUs behave alike, and which items travel together at the till?",
        method="k-means on four standardized features with elbow + silhouette diagnostics "
               "to choose k; association rules (support, confidence, lift) computed over "
               "500 POS transactions.",
        outcome="Named SKU segments with differentiated slotting / cycle-count policy, and "
                "a ranked rule list (e.g., OilFilter -> AirFilter) driving bin co-location."),
}


@st.cache_resource(show_spinner="Training models ...")
def get_result(page):
    return PAGES[page].run()


def header(title, page):
    st.title(title)
    meta = META[page]
    c1, c2, c3 = st.columns(3)
    c1.subheader("Problem"); c1.write(meta["problem"])
    c2.subheader("Method"); c2.write(meta["method"])
    c3.subheader("Outcome"); c3.write(meta["outcome"])
    st.divider()


def download_xlsx(label, sheets, fname):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    st.download_button(f"Download {label}", buf.getvalue(), file_name=fname,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


page = st.sidebar.radio("Use case", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("IIM Calcutta · Guest Lecture Lab\nDecision Models for Supply Chain Management")

if page == "Overview":
    st.title("Supply-Chain ML Lab - EPSCM — Six Use Cases, One Storyline")
    st.write("Each pen-and-paper assignment from the course is scaled up here into an "
             "industrial machine-learning problem. Pick a use case from the sidebar; every "
             "page shows the **Problem**, the **Method** and the **Outcome**, with the "
             "input data, model results and downloadable outputs.")
    rows = [[k, META[k]["problem"][:90] + "..."] for k in list(PAGES)[1:]]
    st.table(pd.DataFrame(rows, columns=["Use case", "Business problem (abridged)"]))
    st.info("Everything runs live from the six Excel files in ./inputs — swap in your own "
            "data with the same column names and the models retrain instantly.")

elif page == "1 · Demand Forecasting":
    header("Demand Forecasting at Scale", page)
    r = get_result(page)
    a, b, c = st.columns(3)
    hwm = r["metrics"].iloc[0]; gbm = r["metrics"].iloc[1]
    a.metric("Holt-Winters MAPE (holdout)", f'{hwm["MAPE_holdout_pct"]}%')
    b.metric("Gradient Boosting MAPE", f'{gbm["MAPE_holdout_pct"]}%')
    c.metric("Forecast horizon", "12 weeks")
    st.pyplot(r["figure"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input demand history (tail)"); st.dataframe(r["input"].tail(10), width='stretch')
        st.subheader("Model comparison"); st.dataframe(r["metrics"], width='stretch')
    with c2:
        st.subheader("12-week forecast"); st.dataframe(r["forecast"], width='stretch')
        st.subheader("Top ML features"); st.dataframe(r["feature_importance"].head(8), width='stretch')
    download_xlsx("forecast workbook", {"Model_Comparison": r["metrics"], "Backtest": r["backtest"],
                                        "Forecast_12wk": r["forecast"]}, "output_1_demand_forecasting.xlsx")

elif page == "2 · Inventory Optimization":
    header("Multi-SKU Inventory Optimization", page)
    r = get_result(page)
    a, b, c = st.columns(3)
    a.metric("SKUs optimized", len(r["plan"]))
    a_share = r["abc_summary"].set_index("abc_class").loc["A", "value_share_pct"]
    b.metric("A-class value share", f"{a_share}%")
    c.metric("Simulated service (top SKU)", r["simulation"].iloc[2]["value"])
    st.pyplot(r["figure"])
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Replenishment plan"); st.dataframe(r["plan"], width='stretch', height=320)
    with c2:
        st.subheader("ABC summary"); st.dataframe(r["abc_summary"], width='stretch')
        st.subheader("Monte-Carlo check"); st.dataframe(r["simulation"], width='stretch')
    download_xlsx("replenishment plan", {"Plan": r["plan"], "ABC": r["abc_summary"],
                                         "Simulation": r["simulation"]}, "output_2_inventory_plan.xlsx")

elif page == "3 · Shipment Delay Prediction":
    header("Predicting Shipment Delays", page)
    r = get_result(page)
    rfm = r["metrics"].iloc[1]
    a, b, c = st.columns(3)
    a.metric("Recall (Delayed class)", rfm["recall"])
    b.metric("ROC-AUC", rfm["ROC_AUC"])
    c.metric("Shipments scored", len(r["input"]))
    st.pyplot(r["figure"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model scorecard"); st.dataframe(r["metrics"], width='stretch')
        st.subheader("RF confusion matrix (test)"); st.dataframe(r["confusion_rf"], width='stretch')
    with c2:
        st.subheader("Delay drivers — logit odds ratios"); st.dataframe(r["odds_ratios"], width='stretch', height=260)
    st.subheader("Risk watchlist — 15 most at-risk shipments")
    st.dataframe(r["watchlist"], width='stretch')
    download_xlsx("delay model workbook", {"Metrics": r["metrics"], "Watchlist": r["watchlist"],
                                           "Odds_Ratios": r["odds_ratios"]}, "output_3_delay_model.xlsx")

elif page == "4 · Supplier Quality Analytics":
    header("Supplier Quality & Risk Analytics", page)
    r = get_result(page)
    n_pref = int((r["scorecard"]["tier"] == "Preferred").sum())
    n_exit = int((r["scorecard"]["tier"] == "Exit/PIP").sum())
    a, b, c = st.columns(3)
    a.metric("Vendors analysed", len(r["scorecard"]))
    b.metric("Preferred tier", n_pref)
    c.metric("Exit / PIP tier", n_exit)
    st.pyplot(r["figure"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vendor scorecard"); st.dataframe(r["scorecard"], width='stretch')
        st.subheader("SPC p-chart summary"); st.dataframe(r["spc"], width='stretch')
    with c2:
        st.subheader("Two-proportion tests vs best vendor"); st.dataframe(r["ztests"], width='stretch')
        st.subheader("Lead-time promise tests (5 days)"); st.dataframe(r["leadtime_tests"], width='stretch')
    download_xlsx("vendor scorecard", {"Scorecard": r["scorecard"], "SPC": r["spc"],
                                       "Ztests": r["ztests"], "LeadTime": r["leadtime_tests"]},
                  "output_4_vendor_scorecard.xlsx")

elif page == "5 · Freight Cost Drivers":
    header("Freight-Cost Drivers & Quote Engine", page)
    r = get_result(page)
    ols = r["metrics"].iloc[0]
    a, b, c = st.columns(3)
    a.metric("OLS R2 (test)", ols["R2_test"])
    b.metric("OLS MAPE", f'{ols["MAPE_pct"]}%')
    c.metric("Lanes quoted", len(r["quotes"]))
    st.pyplot(r["figure"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cost drivers (Rs per unit of driver)"); st.dataframe(r["coefficients"], width='stretch', height=300)
    with c2:
        st.subheader("Model comparison"); st.dataframe(r["metrics"], width='stretch')
        st.subheader("Instant quotes — new lanes"); st.dataframe(r["quotes"], width='stretch')
    download_xlsx("freight model workbook", {"Metrics": r["metrics"], "Drivers": r["coefficients"],
                                             "Quotes": r["quotes"]}, "output_5_freight_model.xlsx")

elif page == "6 · SKU Segments & Market Basket":
    header("SKU Segmentation & Market-Basket Mining", page)
    r = get_result(page)
    a, b, c = st.columns(3)
    a.metric("Best k (silhouette)", r["best_k"])
    b.metric("SKUs segmented", len(r["profile"]))
    c.metric("Rules above lift 1.05", len(r["rules"]))
    st.pyplot(r["figure"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cluster centres"); st.dataframe(r["centers"], width='stretch')
        st.subheader("Segment sizes"); st.dataframe(r["segment_sizes"], width='stretch')
    with c2:
        st.subheader("Top association rules"); st.dataframe(r["rules"], width='stretch', height=320)
    st.subheader("Segmented SKU master (sample)")
    st.dataframe(r["profile"].head(15), width='stretch')
    download_xlsx("segments & rules", {"Segments": r["profile"], "Centers": r["centers"],
                                       "Rules": r["rules"]}, "output_6_segments_rules.xlsx")
