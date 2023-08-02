import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming you have a CSV file with customer data, load it into a DataFrame
# Replace 'path_to_csv_file.csv' with the actual path to your file
data = pd.read_csv('path_to_csv_file.csv')

# Select relevant features for segmentation (e.g., 'age', 'income', 'spending_score', etc.)
selected_features = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names

# Extract the selected features and drop any rows with missing values
X = data[selected_features].dropna()

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
max_clusters = 10
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph to find the optimal number of clusters
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Based on the Elbow Method, choose the appropriate number of clusters
num_clusters = 4  # Replace with the chosen number of clusters

# Perform K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the original DataFrame
data['cluster'] = kmeans.labels_

# Explore the characteristics of each customer segment
for cluster_num in range(num_clusters):
    segment_data = data[data['cluster'] == cluster_num]
    print(f"Segment {cluster_num} - Number of Customers: {len(segment_data)}")
    print(segment_data.describe())

# Visualize the clusters (assuming 2D data)
if len(selected_features) == 2:
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title('Customer Segmentation')
    plt.legend()
    plt.show()
