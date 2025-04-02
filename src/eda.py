import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load parquet file into a Pandas DataFrame
data = pd.read_parquet("yellow_tripdata_2022-01.parquet")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# Display the first few rows of the dataset
print(data.head(5))

# Display the shape of the dataset
print("shape of the dataset : ", data.shape)


# Check for missing values
missingValues = data.isnull().sum()
print("Missing Values :", missingValues)

# Summary statistics of the dataset
print(data.describe())

# Drop rows with missing values.
data_cleaned = data.dropna()
data_cleaned.to_parquet("yellow_tripdata_2022-01_cleaned.parquet")

# Create a new column for trip duration in minutes
data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
data["trip_duration"] = (data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]).dt.total_seconds() / 60
#print(data.head(5))


# Create new columns for pickup hour and day of week

data["pickup_hour"] = data["tpep_pickup_datetime"].dt.hour  # Extracts the hour (0-23)
data["pickup_day_of_week"] = data["tpep_pickup_datetime"].dt.day_name()  # Extracts the day name (e.g., Monday)

# Display first few rows with new columns
#print(data[["tpep_pickup_datetime", "pickup_hour", "pickup_day_of_week"]].head(50))


# Create a lineplot displaying the number of trips by pickup hour

hourly_trips = data["pickup_hour"].value_counts().sort_index()

# Plot the line chart
plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_trips.index, y=hourly_trips.values, marker="o")

# Customize the plot
plt.title("Number of Trips by Pickup Hour")
plt.xlabel("Pickup Hour (24 Hours)")
plt.ylabel("Number of Trips")
plt.xticks(range(0, 24))
plt.grid(True)

# Show plot
#plt.show()


# Create a lineplot displaying the number of trips by pickup day
day_trips = data["pickup_day_of_week"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.lineplot(x=day_trips.index, y=day_trips.values, marker="o")

# Customize the plot
plt.title("Number of Trips by Pickup Day")
plt.xlabel("Pickup Day)")
plt.ylabel("Number of Trips")
plt.xticks(range(0, 7))
plt.grid(True)

# Show plot
#plt.show()


# Compute correlation matrix of numerical variables
columns = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'trip_duration']

# Compute the correlation matrix
correlation_matrix = data[columns].corr()
plt.figure(figsize=(8, 6))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Set the title
plt.title("Correlation Heatmap of Selected Variables")

# Show the plot
#plt.show()

# Create a scatter plot matrix of numerical variables. If memory issues try the df.sample method.

sampled_data = data[columns].sample(n=10_000, random_state=42)

# Create the pairplot
sns.pairplot(sampled_data, diag_kind="kde", corner=True)

# Show the plot
plt.title("Pair plots ::")
#plt.show()

# Create a Seaborn countplot for PULocationID and DOLocationID. Only plot the top 15 categories by value counts.
top_pu_locations = data['PULocationID'].value_counts().nlargest(15).index

top_do_locations = data['DOLocationID'].value_counts().nlargest(15).index

filtered_data = data[(data['PULocationID'].isin(top_pu_locations)) |
                     (data['DOLocationID'].isin(top_do_locations))]

plt.figure(figsize=(12, 6))

sns.countplot(x=filtered_data['PULocationID'], order=top_pu_locations, palette="viridis")
plt.title("Top 15 Pick-Up Locations")
plt.xlabel("Pick-Up Location ID")
plt.ylabel("Count")
plt.xticks(rotation=45)
#plt.show()


# Create a box plot of total amount by payment type. Do you see anything odd?
plt.figure(figsize=(10, 6))
sns.boxplot(x='payment_type', y='total_amount', data=data, palette='Set2')
plt.title("Distribution of Total Amount by Payment Type")
plt.xlabel("Payment Type")
plt.ylabel("Total Amount")
plt.ylim(-10, 200)
#plt.show()


# Explore data distributions for 'fare_amount', 'trip_distance' and 'extra' using Seaborn's histplot. Sample the data if you run into memory issues.
columns = ['fare_amount', 'trip_distance', 'extra']

sampled_data = data[columns].sample(n=10_000, random_state=42)

plt.figure(figsize=(15, 5))

for i, col in enumerate(columns, 1):
    plt.subplot(1, 3, i)
    sns.histplot(sampled_data[col], kde=True, bins=50, color='royalblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()