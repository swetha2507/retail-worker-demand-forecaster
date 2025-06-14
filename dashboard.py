import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time

def print_debug(message):
    """Print debug message with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

@st.cache_data
def load_data():
    """Load and prepare the data."""
    print_debug("Starting data load...")
    start_time = time.time()
    
    # Load the predictions
    df = pd.read_csv('staff_predictions.csv')
    print_debug(f"CSV loaded. Shape: {df.shape}")
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    print_debug(f"Date conversion complete. Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    end_time = time.time()
    print_debug(f"Data load completed in {end_time - start_time:.2f} seconds")
    return df

@st.cache_data
def create_line_chart(df):
    """Create a line chart showing actual vs predicted staff levels."""
    print_debug("Creating line chart...")
    start_time = time.time()
    
    # Resample data to daily averages for smoother visualization
    numeric_cols = ['Actual_Staff', 'Predicted_Staff']
    df_numeric = df[numeric_cols].copy()
    df_numeric.index = df['Date']
    daily_avg = df_numeric.resample('D').mean()
    print_debug(f"Data resampled to daily. Shape: {daily_avg.shape}")
    
    fig = go.Figure()
    
    # Add actual staff line
    fig.add_trace(go.Scatter(
        x=daily_avg.index,
        y=daily_avg['Actual_Staff'],
        name='Actual Staff',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add predicted staff line
    fig.add_trace(go.Scatter(
        x=daily_avg.index,
        y=daily_avg['Predicted_Staff'],
        name='Predicted Staff',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Daily Staff Levels',
        xaxis_title='Date',
        yaxis_title='Number of Staff',
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    end_time = time.time()
    print_debug(f"Line chart created in {end_time - start_time:.2f} seconds")
    return fig

@st.cache_data
def create_heatmap(df):
    """Create a heatmap showing over/understaffed days."""
    print_debug("Creating heatmap...")
    start_time = time.time()
    
    # Calculate staff difference
    df['Staff_Difference'] = df['Predicted_Staff'] - df['Actual_Staff']
    print_debug("Staff difference calculated")
    
    # Resample data to weekly averages to reduce data points
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Week_Start'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='d')
    print_debug("Weekly aggregation prepared")
    
    # Create pivot table for heatmap with weekly aggregation
    # Convert Store ID to string to ensure proper handling
    df['Store ID'] = df['Store ID'].astype(str)
    pivot_df = df.pivot_table(
        values='Staff_Difference',
        index='Week_Start',
        columns='Store ID',
        aggfunc='mean'
    )
    print_debug(f"Pivot table created. Shape: {pivot_df.shape}")
    
    # Format week labels
    pivot_df.index = pivot_df.index.strftime('Week of %b %d')
    
    # Format store labels
    pivot_df.columns = [f'Store {col[1:]}' for col in pivot_df.columns]
    
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Store", y="Week", color="Staff Difference"),
        title="Weekly Staff Difference Heatmap (Predicted - Actual)",
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        xaxis_title="Store",
        yaxis_title="Week",
        height=600
    )
    
    end_time = time.time()
    print_debug(f"Heatmap created in {end_time - start_time:.2f} seconds")
    return fig

def calculate_model_metrics(df):
    """Calculate model performance metrics."""
    rmse = np.sqrt(np.mean((df['Predicted_Staff'] - df['Actual_Staff'])**2))
    r2 = 1 - np.sum((df['Actual_Staff'] - df['Predicted_Staff'])**2) / np.sum((df['Actual_Staff'] - df['Actual_Staff'].mean())**2)
    mae = np.mean(np.abs(df['Predicted_Staff'] - df['Actual_Staff']))
    return rmse, r2, mae

def main():
    st.set_page_config(page_title="Retail Worker Demand Forecaster", layout="wide")
    st.title("Retail Worker Demand Forecaster")
    
    # About section in sidebar
    with st.sidebar:
        st.header("About this Dashboard")
        st.markdown("""
        **Purpose:** Forecast daily staff demand per store using historical retail trends.
        
        **Audience:** Regional/store managers.
        
        **Technology:** Python, Streamlit, XGBoost.
        
        **Goal:** Improve scheduling efficiency and reduce under/overstaffing.
        """)
        st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Store selection
    stores = ['All'] + sorted(df['Store ID'].unique().tolist())
    selected_store = st.sidebar.selectbox("Select Store", stores)
    
    # Date range selection
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    default_start = max_date - pd.Timedelta(days=30)
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_store != 'All':
        filtered_df = filtered_df[filtered_df['Store ID'] == selected_store]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= date_range[0]) &
            (filtered_df['Date'].dt.date <= date_range[1])
        ]
    
    # Calculate staff difference
    filtered_df['Staff_Difference'] = filtered_df['Predicted_Staff'] - filtered_df['Actual_Staff']
    
    # Calculate model metrics
    rmse, r2, mae = calculate_model_metrics(filtered_df)
    
    # Display model metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    st.sidebar.markdown(f"""
    📉 RMSE: {rmse:.1f} staff
    📈 R²: {r2:.2f}
    📊 MAE: {mae:.1f} staff
    """)
    
    # Display summary statistics
    st.header("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Staff Difference",
            f"{filtered_df['Staff_Difference'].mean():.1f}",
            f"{filtered_df['Staff_Difference'].std():.1f} std",
            help="Staff Difference = Predicted - Actual. Positive values indicate overprediction."
        )
    
    with col2:
        st.metric(
            "Average Actual Staff",
            f"{filtered_df['Actual_Staff'].mean():.1f}",
            f"{filtered_df['Actual_Staff'].std():.1f} std",
            help="Average number of staff actually present."
        )
    
    with col3:
        st.metric(
            "Average Predicted Staff",
            f"{filtered_df['Predicted_Staff'].mean():.1f}",
            f"{filtered_df['Predicted_Staff'].std():.1f} std",
            help="Average number of staff predicted by the model."
        )
    
    # Display staffing alerts
    st.subheader("Staffing Alerts")
    col1, col2 = st.columns(2)
    
    with col1:
        overstaffed = (filtered_df['Staff_Difference'] > 0).sum()
        understaffed = (filtered_df['Staff_Difference'] < 0).sum()
        st.markdown(f"""
        🔺 Overstaffed Days: {overstaffed}  
        🔻 Understaffed Days: {understaffed}
        """)
    
    with col2:
        severe_understaff = (filtered_df['Staff_Difference'] < -2).sum()
        within_prediction = (abs(filtered_df['Staff_Difference']) <= 1).sum()
        accuracy = (within_prediction / len(filtered_df)) * 100
        st.markdown(f"""
        🚨 Severely Understaffed Days: {severe_understaff}  
        ✅ {accuracy:.0f}% of days within ±1 staff of prediction
        """)
        st.caption("Within ±1 staff = close prediction zone used for performance scoring")
    
    # Add context summary
    st.markdown(f"""
    This forecast helps regional managers allocate staff efficiently across {len(stores)-1} stores. 
    You're currently viewing staff levels from {date_range[0].strftime('%b %Y')} to {date_range[1].strftime('%b %Y')}.
    """)
    
    # Create and display charts
    st.header("Staffing Analysis")
    
    # Line chart
    st.subheader("Daily Staff Levels")
    line_chart = create_line_chart(filtered_df)
    st.plotly_chart(line_chart, use_container_width=True)
    
    # Heatmap (only show when viewing all stores)
    if selected_store == 'All':
        st.subheader("Staff Difference Heatmap")
        heatmap = create_heatmap(filtered_df)
        st.plotly_chart(heatmap, use_container_width=True)
    
    # Add download button
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "Download Forecast Data",
        filtered_df.to_csv(index=False),
        "staff_forecast.csv",
        "text/csv"
    )

if __name__ == "__main__":
    print_debug("Starting dashboard...")
    main() 