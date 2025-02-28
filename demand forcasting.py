import pandas as pd

# Load the dataset
#data = pd.read_csv("C:\Users\yamin\Desktop\Major Pproject data1.csv")
data = pd.read_csv("C:\\Users\\yamin\\Desktop\\Major Pproject data1.csv")
# Extract unique categories and their count
unique_categories = data['Category'].unique()
category_count = len(unique_categories)

# Display the count and names of categories
print(f"Number of unique categories: {category_count}")
print("Category names:")
for category in unique_categories:
    print(f"- {category}")

data.info()

# Ensure necessary libraries are imported
import pandas as pd

# Load the dataset
data['Order date'] = pd.to_datetime(data['Order date'], dayfirst=True, errors='coerce')

# Filter only the relevant columns
data = data[['Category', 'Sub Category', 'Quantity']]
data.dropna(inplace=True)

# Group by Category and Sub Category to calculate total demand
demand_summary = data.groupby(['Category', 'Sub Category']).sum().reset_index()

# Sort by Quantity in descending order (most demanded first)
demand_summary_sorted = demand_summary.sort_values(by='Quantity', ascending=False)

# Split into high-demand and low-demand groups
high_demand = demand_summary_sorted[demand_summary_sorted['Quantity'] > 0]
low_or_no_demand = demand_summary_sorted[demand_summary_sorted['Quantity'] == 0]

# Display results
print("Most Demanded Categories and Sub-Categories:")
print(high_demand)

print("\nLeast or No Demanded Categories and Sub-Categories:")
print(low_or_no_demand)

##################################### ALGORITHUM ####################
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
#data = pd.read_csv('major project data.csv')
data = pd.read_csv("C:\\Users\\yamin\\Desktop\\Major Pproject data1.csv")
# Step 1: Parse 'Order date' and set as index
data['Order date'] = pd.to_datetime(data['Order date'].str.strip(), dayfirst=True, errors='coerce')
data = data.sort_values('Order date')
data.set_index('Order date', inplace=True)

# Step 2: Aggregate daily demand
daily_demand = data['Quantity'].resample('D').sum().fillna(0)

# Step 3: Feature Engineering - Create moving averages and trend analysis
df = pd.DataFrame(daily_demand)
df['7_day_moving_avg'] = df['Quantity'].rolling(window=7).mean()   # Weekly trend
df['30_day_moving_avg'] = df['Quantity'].rolling(window=30).mean() # Monthly trend

# Fill NaN values resulting from rolling calculations
df = df.fillna(0)

# Step 4: Simple forecasting using moving average and trend analysis
# We will forecast the next day's demand based on a weighted average of recent trends
forecast = []
for i in range(len(df)):
    # Simple formula: 0.6 * 7-day average + 0.4 * 30-day average
    daily_forecast = 0.6 * df['7_day_moving_avg'].iloc[i] + 0.4 * df['30_day_moving_avg'].iloc[i]
    forecast.append(daily_forecast)

# Append forecast to DataFrame for comparison with actual demand
df['Forecast'] = forecast

# Step 5: Evaluation - Calculate Mean Squared Error and Mean Absolute Error
mse = np.mean((df['Quantity'] - df['Forecast']) ** 2)
mae = np.mean(np.abs(df['Quantity'] - df['Forecast']))
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Step 6: Plotting actual vs forecasted values
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Quantity'], label='Actual Demand', color='blue')
plt.plot(df.index, df['Forecast'], label='Forecasted Demand', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Actual vs Forecasted Demand Using Moving Averages')
plt.legend()
plt.show()

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
#data = pd.read_csv('major project data.csv')
data = pd.read_csv("C:\\Users\\yamin\\Desktop\\Major Pproject data1.csv")

# Step 1: Parse 'Order date' and set as index
data['Order date'] = pd.to_datetime(data['Order date'].str.strip(), dayfirst=True, errors='coerce')
data = data.sort_values('Order date')
data.set_index('Order date', inplace=True)

# Step 2: Aggregate daily demand
daily_demand = data['Quantity'].resample('D').sum().fillna(0)

# Step 3: Feature Engineering - Create moving averages
df = pd.DataFrame(daily_demand)
df['7_day_moving_avg'] = df['Quantity'].rolling(window=7).mean()   # Weekly trend
df['30_day_moving_avg'] = df['Quantity'].rolling(window=30).mean() # Monthly trend

# Fill NaN values resulting from rolling calculations
df = df.fillna(0)

# Step 4: Simulate epochs by iteratively adjusting weights
epochs = 50  # Number of epochs for training
weight_7day = 0.5  # Initial weight for 7-day moving average
weight_30day = 0.5  # Initial weight for 30-day moving average

# Lists to store error values for each epoch
mse_list = []
mae_list = []

for epoch in range(epochs):
    # Calculate forecast based on current weights
    forecast = weight_7day * df['7_day_moving_avg'] + weight_30day * df['30_day_moving_avg']
    
    # Calculate errors
    mse = np.mean((df['Quantity'] - forecast) ** 2)
    mae = np.mean(np.abs(df['Quantity'] - forecast))
    
    # Store errors for each epoch
    mse_list.append(mse)
    mae_list.append(mae)
    
    # Update weights slightly to minimize error (simulating training adjustment)
    # For simplicity, reduce 7-day weight if error increased, otherwise increase it
    if epoch > 0 and mse_list[epoch] < mse_list[epoch - 1]:
        weight_7day += 0.01
        weight_30day -= 0.01
    else:
        weight_7day -= 0.01
        weight_30day += 0.01
    
    # Ensure weights remain between 0 and 1
    weight_7day = min(max(weight_7day, 0), 1)
    weight_30day = min(max(weight_30day, 0), 1)
    
    # Print weights and errors for each epoch
    print(f"Epoch {epoch + 1}/{epochs} - MSE: {mse:.4f}, MAE: {mae:.4f}, 7-day Weight: {weight_7day:.2f}, 30-day Weight: {weight_30day:.2f}")



# Step 6: Plotting final actual vs forecasted values using final weights
df['Forecast'] = weight_7day * df['7_day_moving_avg'] + weight_30day * df['30_day_moving_avg']
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Quantity'], label='Actual Demand', color='blue')
plt.plot(df.index, df['Forecast'], label='Forecasted Demand', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Actual vs Forecasted Demand Using Moving Averages')
plt.legend()
plt.show()

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the dataset
data = pd.read_csv('major project data.csv')
data['Order date'] = pd.to_datetime(data['Order date'].str.strip(), dayfirst=True, errors='coerce')
data = data.sort_values('Order date')
data.set_index('Order date', inplace=True)

# Aggregate daily demand
daily_demand = data['Quantity'].resample('D').sum().fillna(0)

# Split data into training and testing sets (e.g., 80% training, 20% testing)
split_ratio = 0.8
split_index = int(len(daily_demand) * split_ratio)
train_demand = daily_demand[:split_index]
test_demand = daily_demand[split_index:]

# Initialize DataFrames for training and testing sets with moving averages
train_df = pd.DataFrame(train_demand)
test_df = pd.DataFrame(test_demand)

for df in [train_df, test_df]:
    df['7_day_moving_avg'] = df['Quantity'].rolling(window=7).mean()
    df['30_day_moving_avg'] = df['Quantity'].rolling(window=30).mean()
    df.fillna(0, inplace=True)

# Initialize weights and variables for tracking errors
epochs = 50
weight_7day, weight_30day = 0.5, 0.5
train_mse_list, train_mae_list = [], []
test_mse_list, test_mae_list = [], []

# Training loop with weight adjustments
for epoch in range(epochs):
    # Calculate forecast for training and testing sets
    train_forecast = weight_7day * train_df['7_day_moving_avg'] + weight_30day * train_df['30_day_moving_avg']
    test_forecast = weight_7day * test_df['7_day_moving_avg'] + weight_30day * test_df['30_day_moving_avg']
    
    # Calculate MSE and MAE for training and testing sets
    train_mse = np.mean((train_df['Quantity'] - train_forecast) ** 2)
    train_mae = np.mean(np.abs(train_df['Quantity'] - train_forecast))
    test_mse = np.mean((test_df['Quantity'] - test_forecast) ** 2)
    test_mae = np.mean(np.abs(test_df['Quantity'] - test_forecast))
    
    # Store error metrics for plotting
    train_mse_list.append(train_mse)
    train_mae_list.append(train_mae)
    test_mse_list.append(test_mse)
    test_mae_list.append(test_mae)
    
    # Adjust weights to reduce training error
    if epoch > 0 and train_mse < train_mse_list[epoch - 1]:  # If error improved, increase 7-day weight
        weight_7day += 0.01
        weight_30day -= 0.01
    else:
        weight_7day -= 0.01
        weight_30day += 0.01
    
    # Ensure weights stay between 0 and 1
    weight_7day = min(max(weight_7day, 0), 1)
    weight_30day = min(max(weight_30day, 0), 1)

# Plot MSE and MAE for both training and testing sets over epochs
plt.figure(figsize=(14, 7))
plt.plot(range(1, epochs + 1), train_mse_list, label='Train MSE', color='blue')
plt.plot(range(1, epochs + 1), test_mse_list, label='Test MSE', color='blue', linestyle='--')
plt.plot(range(1, epochs + 1), train_mae_list, label='Train MAE', color='green')
plt.plot(range(1, epochs + 1), test_mae_list, label='Test MAE', color='green', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and Testing Error Over Epochs')
plt.legend()
plt.show()


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load and preprocess the dataset
#data = pd.read_csv('major project data.csv')
data = pd.read_csv("C:\\Users\\yamin\\Desktop\\Major Pproject data1.csv")
data['Order date'] = pd.to_datetime(data['Order date'], dayfirst=True, errors='coerce')

# Filter only the columns needed for prediction
data = data[['Order date', 'Category', 'Sub Category', 'Quantity']]
data.dropna(inplace=True)

# Set the index to Order date for time series manipulation
data.set_index('Order date', inplace=True)

# Aggregate demand per day for each Category and Sub Category
daily_demand = data.groupby(['Category', 'Sub Category']).resample('D').sum().fillna(0)

# Calculate rolling demand averages to create features
daily_demand['7_day_avg'] = daily_demand['Quantity'].rolling(window=7).mean().fillna(0)
daily_demand['30_day_avg'] = daily_demand['Quantity'].rolling(window=30).mean().fillna(0)

# Define high-demand threshold (e.g., demand above a certain quantity is "high demand")
threshold = daily_demand['Quantity'].quantile(0.75)  # Top 25% of demand is labeled high

# Create target variable: High demand (1) or low demand (0) based on threshold
daily_demand['High_demand'] = (daily_demand['Quantity'] > threshold).astype(int)

# Drop rows with missing data due to rolling averages
daily_demand.dropna(inplace=True)

# Prepare data for training
X = daily_demand[['7_day_avg', '30_day_avg']].values
y = daily_demand['High_demand'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Print classification report to evaluate the model
print(classification_report(y_test, y_pred))

# Predict future demand for each category and sub-category based on recent averages
# For simplicity, we will use the last 30 days average to make predictions
future_demand = daily_demand[['7_day_avg', '30_day_avg']].iloc[-30:]
future_predictions = clf.predict(future_demand.values)

# Show the predicted categories and sub-categories that will be in high demand
future_demand['Predicted_High_Demand'] = future_predictions
high_demand_future = future_demand[future_demand['Predicted_High_Demand'] == 1]

print("Categories and Sub-categories predicted to be in high demand:")
print(high_demand_future)
