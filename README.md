# Retail Worker Demand Forecaster

A machine learning-based tool that predicts daily staffing needs for retail stores using historical sales, promotions, and weather data. The project includes a fully interactive dashboard for business users.

---

## Project Structure

| File | Description |
|------|-------------|
| `retail_store_inventory.csv` | Raw dataset with inventory, sales, region, and weather information |
| `retail_forecaster.py` | Data preprocessing and model training script |
| `retail_staff_model.joblib` | Trained XGBoost model |
| `staff_predictions.csv` | Predicted daily staff levels |
| `dashboard.py` | Streamlit app for visualizing predictions |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Setup Instructions

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## Running the Project

**1. Train the model and generate staff predictions:**

```bash
python retail_forecaster.py
```

**2. Launch the interactive dashboard:**

```bash
streamlit run dashboard.py
```

---

## Dashboard Features

* Line chart comparing actual vs predicted staff levels
* Weekly heatmap showing staff prediction errors
* Summary statistics and model performance
* Filters for date range and store selection
* Downloadable forecast data for further analysis

---

## Model Overview

* Model: XGBoost Regressor
* Features used: day of week, weekend flag, region, seasonality, weather, price, discount
* Evaluation metrics:
  * RMSE: 1.3 staff
  * R²: 0.99
  * MAE: 1.0 staff

---

## Business Impact

This tool enables retail managers to:

* Optimize daily staffing schedules
* Reduce labor cost from overstaffing
* Avoid service disruptions caused by understaffing
* Make informed, data-driven operational decisions

---

## Future Enhancements

* Add holiday and local event data
* Incorporate multiple model comparisons
* Include prediction confidence intervals
* Add cost analysis for staffing errors 