# Retail Worker Demand Forecaster

This project predicts how many workers each retail store might need on a given day, based on past sales, promotions, pricing, weather, and more. I trained a machine learning model to make these forecasts and built an interactive Streamlit dashboard to explore the results.

---

## Dataset

I used the [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) from Kaggle. It contains daily data for multiple products across different stores, including information like units sold, pricing, promotions, inventory levels, regional info, weather, and seasonality.

---

## How I prepared the dataset

Before building the model, I cleaned and transformed the raw data with the following steps:

- Aggregated product-level data into **daily totals per store**
- Created new features like:
  - `Day of the week`
  - `Is weekend`
  - `Estimated staff needed` (calculated based on units sold)
- Filtered out unused columns to streamline the model
- Applied one-hot encoding to categorical features like region, weather, and seasonality

These transformations made the data easier to work with and helped the model learn more effectively.

---

## Project Structure

| File | Description |
|------|-------------|
| `retail_store_inventory.csv` | Preprocessed dataset with store-level data |
| `retail_forecaster.py` | Script for data cleaning, feature engineering, model training, and prediction |
| `retail_staff_model.joblib` | Trained XGBoost model |
| `staff_predictions.csv` | Model output with daily staff demand predictions |
| `dashboard.py` | Streamlit dashboard to visualize model results |
| `requirements.txt` | List of Python dependencies |
| `README.md` | This documentation |

---

## How to run the project

1. **Set up the environment**

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Generate the predictions**

```bash
python retail_forecaster.py
```

3. **Launch the dashboard**

```bash
streamlit run dashboard.py
```

---

## Dashboard features

- Line chart comparing actual vs predicted staff levels per store
- Heatmap showing over/understaffing by week and store
- Summary metrics: RMSE, MAE, R², average staff gap
- Filters for date range and store selection
- Downloadable CSV of forecasted staff data

---

## Model details

- **Model used:** XGBoost Regressor
- **Key features:** day of week, weekend flag, weather, seasonality, region, price, discount, units sold
- **Performance:**
  - RMSE: 1.3
  - MAE: 1.0
  - R²: 0.99

---

## Why this matters

Retail teams often overstaff or understaff based on rough estimates. This tool gives store and regional managers a data-backed way to plan ahead, reduce labor costs, and ensure smoother operations—especially during seasonal or promotional spikes.

---

## Future enhancements

- Add holiday and local event indicators
- Try out multiple model types and compare results
- Add confidence intervals for predictions
- Include cost analysis for staffing decisions 