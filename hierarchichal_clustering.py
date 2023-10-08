import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Generate sample data
data = np.array([[1, 2], [2, 3], [6, 8], [8, 8], [1, 6], [9, 10], [10, 11], [12, 13]])

# Perform hierarchical clustering using linkage
linkage_matrix = linkage(data, method='ward')  # You can choose different linkage methods

# Plot the dendrogram
dendrogram(linkage_matrix, labels=range(1, len(data) + 1))
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
