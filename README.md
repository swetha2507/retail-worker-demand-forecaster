# Retail Worker Demand Forecaster

This is a project I built to help retail store managers figure out how many staff members they might need each day. The idea is simple: use past sales data, weather, promotions, and other details to predict future staffing needs. I trained a machine learning model to do the forecasting and created an interactive dashboard to explore the predictions.

## About the dataset

I used the Retail Store Inventory Forecasting dataset from Kaggle. It includes daily sales and inventory data for different products across multiple stores. It also has info like price, discounts, promotions, weather conditions, and which region the store is in. All of this gave me enough context to estimate staff demand.

## How I prepared the data

The raw data was at the product level, so the first thing I did was group everything by date and store. I created a few new features like:

* Day of the week
* A flag to check if it was a weekend
* An estimated number of staff needed (based on units sold)

Then I cleaned up the dataset by removing columns I didn't need and one-hot encoded the categorical values like region, weather, and season.

## What the project includes

Here's what's inside the repo:

* The cleaned dataset
* A Python script that processes the data, trains the model, and saves predictions
* The trained model itself (XGBoost)
* A Streamlit dashboard to explore the predictions
* A requirements file with the Python libraries I used

## How to run it

1. Create a virtual environment
2. Install the dependencies from the requirements.txt file
3. Run the script to generate predictions
4. Launch the Streamlit app

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python retail_forecaster.py
streamlit run dashboard.py
```

## What the dashboard shows

The dashboard lets you compare predicted vs actual staff needs over time. There's a line chart, a heatmap that highlights over- or understaffed days, filters for selecting stores and date ranges, and a section for key metrics like RMSE, MAE, and R². You can also download the forecasted data as a CSV.

## How the model works

I used XGBoost for the regression task. The model uses features like day of the week, weekend flag, region, seasonality, weather, pricing, discounts, and sales volume. After training and testing, here's how it performed:

* RMSE: 1.3
* MAE: 1.0
* R²: 0.99

## Why I built this

In many retail environments, staffing decisions are made based on gut feeling or fixed schedules. I wanted to show that even basic historical data could be used to forecast demand more accurately. This could help reduce costs from overstaffing or avoid missed sales and poor customer service due to being understaffed.

## What I'd like to add next

* Holidays and local event info to improve the predictions
* A way to compare multiple models and see which works best
* Confidence intervals for the forecasts
* A simple cost impact analysis for over- and understaffing 