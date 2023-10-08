import numpy as np

def k_means_clustering(data, k, max_iterations=100):
    # Initialize k random centroids
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Update centroids as the mean of the points assigned to each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Example usage
if __name__ == "__main__":
    # Generate random data points
    np.random.seed(0)
    data = np.random.rand(100, 2)
    
    # Number of clusters (k)
    k = 3
    
    # Run K-Means clustering
    labels, centroids = k_means_clustering(data, k)
    
    # Print the labels and centroids
    print("Cluster Labels:")
    print(labels)
    print("\nCentroids:")
    print(centroids)
