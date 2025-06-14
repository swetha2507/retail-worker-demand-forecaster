# Retail Worker Demand Forecaster

This project predicts staffing needs for retail stores based on historical sales data and various features like weather conditions, seasonality, and regional factors.

## Project Structure

- `retail_store_inventory.csv`: Raw dataset containing store inventory and sales data
- `retail_forecaster.py`: Main script for data processing and model training
- `dashboard.py`: Streamlit dashboard for visualizing predictions
- `requirements.txt`: Project dependencies
- `staff_predictions.csv`: Generated predictions (created after running the model)
- `retail_staff_model.joblib`: Trained model file (created after running the model)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model and generate predictions:
```bash
python retail_forecaster.py
```

2. Launch the dashboard:
```bash
streamlit run dashboard.py
```

## Features

### Data Processing
- Date parsing and cleaning
- Feature engineering (day of week, weekend flags)
- Staff demand estimation based on sales volume
- Data aggregation by store and date

### Model
- XGBoost regression model
- Feature engineering including one-hot encoding
- Model evaluation using RMSE and R² metrics

### Dashboard
- Interactive line charts comparing actual vs predicted staff levels
- Heatmap visualization of over/understaffed days
- Filtering by store and date range
- Summary statistics and metrics

## Business Impact

This tool helps retail managers:
- Optimize staffing levels based on predicted demand
- Reduce overstaffing costs
- Prevent understaffing during peak periods
- Make data-driven staffing decisions

## Future Improvements

- Add more features (holidays, special events)
- Implement multiple model comparison
- Add confidence intervals to predictions
- Include cost analysis for staffing decisions 