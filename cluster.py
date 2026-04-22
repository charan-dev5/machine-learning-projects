from sklearn.cluster import KMeans
import numpy as np

# Customer data: [annual spending, visit frequency]
X = [
    [1000, 10],
    [950, 9],
    [200, 2],
    [180, 3],
    [500, 5],
    [480, 6],
    [900, 8],
    [210, 2],
    [520, 5],
    [990, 10]
]

# 3 groups: big spenders, occasional, window shoppers
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

print("Cluster labels:", model.labels_)
print("Cluster centers:", model.cluster_centers_)