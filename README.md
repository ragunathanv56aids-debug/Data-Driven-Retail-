# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Sales_Data.csv")

# Data Cleaning
df.dropna(inplace=True)              # Handle missing values
df.drop_duplicates(inplace=True)     # Remove duplicates

# Feature selection for clustering
X = df[['Quantity', 'TotalAmount']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Quantity'], df['TotalAmount'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Quantity")
plt.ylabel("Total Amount")
plt.title("Customer Segmentation using KMeans")
plt.show()
📸 Output Screenshot Section (README.md)
https://github.com/ragunathanv56aids-debug/Data-Driven-Retail-/blob/main/WhatsApp%20Image%202026-06-28%20at%207.21.46%20PM.jpeg
