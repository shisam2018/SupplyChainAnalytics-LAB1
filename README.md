# IIM Calcutta Supply-Chain ML Lab
Guest-lecture companion code: six pen-and-paper course assignments scaled up
into industrial machine-learning problems.
URL: https://shisam2018-supplychainanalytics-lab1-app-ebspgs.streamlit.app/

## Contents
- `inputs/`  - six Excel input files (one per model). Swap in your own data
  with the same column names and everything retrains.
- `model_1_demand_forecasting.py`    Holt-Winters vs Gradient Boosting, 26-wk backtest, 12-wk forecast
- `model_2_inventory_optimization.py` 40-SKU EOQ / safety stock / ROP, ABC, Monte-Carlo service check
- `model_3_shipment_delays.py`        Delay classification: Logistic Regression vs Random Forest, risk watchlist
- `model_4_supplier_quality.py`       p-charts, proportion & lead-time tests, composite vendor scorecard
- `model_5_freight_costs.py`          OLS cost drivers vs Random Forest, instant lane quotes
- `model_6_sku_clustering_basket.py`  k-means with elbow/silhouette + association rules
- `app.py`                            executive Streamlit demo (Problem -> Method -> Outcome per use case)
- `generate_inputs.py`                regenerates the six input files (seeded, reproducible)
- `outputs/`                          Excel workbooks + PNG charts written by each model

## Quick start
```
pip install -r requirements.txt
python model_1_demand_forecasting.py        # or any other model; outputs land in ./outputs
streamlit run app.py                        # executive UI at http://localhost:8501
```
Prepared for the guest lecture "Data to Decisions: Statistics, ML & AI in the
Modern Supply Chain", Decision Models for Supply Chain Management (PGP Core).
