import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.cluster import KMeans

# 1. Load Dataset
dataset = pd.read_csv(r'C:\Users\ybali\Downloads\Mall_Customers.csv')

# 2. Select Features
X = dataset.iloc[:, [3, 4]].values

print("Dataset Loaded Successfully!")
print(dataset.head())

# 3. Elbow Method
wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# 4. Elbow Graph
plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.grid(True)
plt.show()

# 5. Train Final Model
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# 6. Add Cluster Labels
dataset["cluster"] = y_kmeans

# 7. Cluster Analysis
cluster_analysis = dataset.groupby("cluster").agg({
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean"
})

print("\nCluster Analysis:")
print(cluster_analysis)

# 8. Visualize Clusters
plt.figure(figsize=(10, 6))

for i in range(5):

    plt.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=100,
        label=f"Cluster {i+1}"
    )

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Mall Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend()
plt.grid(True)
plt.show()

# 9. Save Clustered Dataset
dataset.to_csv("clustered_customers.csv", index=False)

# 10. Save Model
with open("kmeans_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

print("\nModel Saved Successfully!")
print("Training Completed!")