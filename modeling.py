from sklearn.cluster import KMeans

def cluster_df_data(df, clustering_method='kmeans', n_clusters=3):
    if clustering_method=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df)

    # Add the cluster labels to the PCA DataFrame
    clustered_df = df.copy()
    clustered_df['Cluster'] = clusters

    return clustered_df, kmeans



from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, OPTICS
# from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler


def apply_clustering_methods(df, cluster_methods, labels_col=None, **kwargs):
    """

    ADDD T-SNE - check why it may be useful

    Apply multiple clustering methods to a time series dataframe and add their results as columns.

    Parameters:
    - df (pd.DataFrame): DataFrame where rows are samples (time series) and columns are time points.
    - cluster_methods (list): List of clustering methods to apply. Supported methods: "kmeans", "dbscan", "time_series_kmeans", "spectral", "optics".
    - labels_col (str): Column name of the true labels, if available (optional).
    - kwargs: Additional parameters for clustering methods (e.g., n_clusters, eps for DBSCAN).

    Returns:
    - pd.DataFrame: DataFrame with added columns containing clustering labels for each method.
    """
    X = df.drop(columns=[labels_col]) if labels_col else df  # Exclude labels column from features if provided
    X_scaled = StandardScaler().fit_transform(X)  # Scaling is often required for clustering

    results_df = df.copy()

    # Loop through the clustering methods
    for method in cluster_methods:
        if method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            results_df['kmeans_labels'] = kmeans.fit_predict(X_scaled)

        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            results_df['dbscan_labels'] = dbscan.fit_predict(X_scaled)

        # elif method == 'time_series_kmeans':
        #     n_clusters = kwargs.get('n_clusters', 3)
        #     ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
        #     results_df['ts_kmeans_labels'] = ts_kmeans.fit_predict(X.values)

        elif method == 'spectral':
            n_clusters = kwargs.get('n_clusters')
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=17)
            results_df['spectral_labels'] = spectral.fit_predict(X_scaled)

        elif method == 'optics':
            min_samples = kwargs.get('min_samples', 5)
            optics = OPTICS(min_samples=min_samples)
            results_df['optics_labels'] = optics.fit_predict(X_scaled)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

    return results_df


# # Example of how to use the function:
# df = pd.DataFrame(np.random.rand(80, 50))  # Example: 80 time series, each with 50 time points
# cluster_methods = ['kmeans', 'dbscan', 'spectral', 'optics']
# clustered_df = apply_clustering_methods(df, cluster_methods, n_clusters=4)
# print(clustered_df.head())
