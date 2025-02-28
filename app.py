from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS to allow communication between frontend and backend
# Load and preprocess data
data = pd.read_csv('newdata.csv.xlsx')

# Function to calculate demand trends
def calculate_trends():
    # Step 1: Parse 'Order date' and set as index
    data['Order date'] = pd.to_datetime(data['Order date'].str.strip(), dayfirst=True, errors='coerce')
    data_sorted = data.sort_values('Order date')
    data_sorted.set_index('Order date', inplace=True)

    # Step 2: Aggregate daily demand
    daily_demand = data_sorted['Quantity'].resample('D').sum().fillna(0)

    # Step 3: Create moving averages
    df = pd.DataFrame(daily_demand)
    df['7_day_moving_avg'] = df['Quantity'].rolling(window=7).mean().fillna(0)
    df['30_day_moving_avg'] = df['Quantity'].rolling(window=30).mean().fillna(0)

    # Step 4: Forecast based on moving averages
    forecast = []
    for i in range(len(df)):
        daily_forecast = 0.6 * df['7_day_moving_avg'].iloc[i] + 0.4 * df['30_day_moving_avg'].iloc[i]
        forecast.append(daily_forecast)
    df['Forecast'] = forecast

    # Prepare response data
    response = {
        "dates": list(df.index.strftime('%Y-%m-%d')),
        "actual": list(df['Quantity']),
        "forecast": list(df['Forecast']),
        "categories": {
            "Office Supplies": [70, 60, 80, 50],  # Mock data; replace with real calculations
            "Furniture": [55, 75, 65],
            "Technology": [85, 60, 70]
        }
    }
    return response

@app.route('/get-demand-trends', methods=['GET'])
def get_demand_trends():
    """API Endpoint to send demand trends"""
    trends = calculate_trends()
    return jsonify(trends)

if __name__ == '__main__':
    app.run(debug=True)
