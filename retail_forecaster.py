import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

def load_and_preprocess_data(file_path):
    """Load and preprocess the retail store inventory data."""
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Check for duplicates and remove them
    df = df.drop_duplicates()
    
    # Check for nulls and fill them
    df = df.fillna(method='ffill')
    
    return df

def engineer_features(df):
    """Engineer features from the raw data."""
    # Create date-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Calculate estimated staff needed
    df['Estimated_Staff_Needed'] = np.ceil(df['Units Sold'] / 50).clip(lower=1)
    
    return df

def aggregate_data(df):
    """Aggregate data by Store and Date."""
    # Define aggregation functions
    agg_dict = {
        'Units Sold': 'sum',
        'Price': 'mean',
        'Region': lambda x: x.mode()[0],
        'Weather Condition': lambda x: x.mode()[0],
        'Seasonality': lambda x: x.mode()[0],
        'Estimated_Staff_Needed': 'sum'
    }
    
    # Group by Store and Date
    aggregated_df = df.groupby(['Store ID', 'Date']).agg(agg_dict).reset_index()
    
    return aggregated_df

def prepare_features(df):
    """Prepare features for modeling."""
    # Select categorical columns for one-hot encoding
    categorical_cols = ['Region', 'Weather Condition', 'Seasonality']
    
    # Create one-hot encoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Fit and transform categorical columns
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, 
                            columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combine with numerical features
    numerical_cols = ['DayOfWeek', 'IsWeekend', 'Units Sold', 'Price']
    final_df = pd.concat([df[numerical_cols], encoded_df], axis=1)
    
    return final_df, df['Estimated_Staff_Needed']

def train_model(X, y):
    """Train the model and return predictions."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train XGBoost model
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Return the test set indices
    test_indices = X_test.index
    return model, X_test, y_test, y_pred, test_indices

def save_predictions(df, y_test, y_pred, test_indices):
    """Save predictions to a CSV file."""
    predictions_df = pd.DataFrame({
        'Date': df.loc[test_indices, 'Date'],
        'Store ID': df.loc[test_indices, 'Store ID'],
        'Predicted_Staff': y_pred,
        'Actual_Staff': y_test
    })
    predictions_df.to_csv('staff_predictions.csv', index=False)
    return predictions_df

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('retail_store_inventory.csv')
    
    # Engineer features
    df = engineer_features(df)
    
    # Aggregate data
    df = aggregate_data(df)
    
    # Re-create date-based features after aggregation
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model and get predictions
    model, X_test, y_test, y_pred, test_indices = train_model(X, y)
    
    # Save predictions
    predictions_df = save_predictions(df, y_test, y_pred, test_indices)
    
    # Save the model
    joblib.dump(model, 'retail_staff_model.joblib')
    
    print("Model training and prediction complete!")

if __name__ == "__main__":
    main() 