import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd

def plot_mean_timeseries_per_cluster_subplots(data_df:pd.DataFrame, clustered_df:pd.DataFrame):
    """
    Plots the mean time series for each cluster in separate subplots.
    Parameters:
    data_df (pd.DataFrame): The original time series data (rows are samples, columns are time points).
    clustered_df (pd.DataFrame): The DataFrame containing cluster labels, must have a 'Cluster' column.
    """
    if 'Cluster' not in clustered_df.columns:
        raise ValueError("The clustered_df must contain a 'Cluster' column with cluster labels.")

    unique_clusters = clustered_df['Cluster'].unique()
    num_clusters = len(unique_clusters)

    # Create subplots, one for each cluster
    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 6 * num_clusters))

    # If there's only one cluster, wrap it in a list so we can iterate
    if num_clusters == 1:
        axes = [axes]

    # For each cluster, calculate the mean time series and plot in a separate subplot
    for i, cluster in enumerate(unique_clusters):
        # Get the rows in the original data that belong to the current cluster
        cluster_indices = clustered_df[clustered_df['Cluster'] == cluster].index
        cluster_timeseries = data_df.loc[cluster_indices]
        print(len(cluster_timeseries))
        sample_timeseries = cluster_timeseries.sample(n=1)
        for i_sample in range(len(sample_timeseries)):
            # sample_timeseries = cluster_timeseries.iloc[i_sample, :]
            axes[i].plot(sample_timeseries.iloc[i_sample, :], label=f'Sample1 {i_sample}, cluster {cluster}')

        # Compute the mean time series
        # mean_timeseries = cluster_timeseries.mean(axis=0)
        # sample_timeseries = cluster_timeseries.sample(n=2).iloc[0, :]
        # axes[i].plot(sample_timeseries, label=f'Sample1 {cluster}')
        # sample_timeseries = cluster_timeseries.sample(n=2).iloc[1, :]


        # Plot the mean time series on the respective subplot
        # axes[i].plot(mean_timeseries, label=f'Cluster {cluster}')

        # axes[i].plot(sample_timeseries, label=f'Sample2 {cluster}')
        axes[i].set_title(f'Mean Time Series for Cluster {cluster}')
        axes[i].set_xlabel('Time Points')
        axes[i].set_ylabel('Mean Value')
        axes[i].grid(True)
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()





def plot_cluster_timeseries(df, cluster_col, title):
    # Extract data columns (all except the cluster column)
    data_cols = df.columns.difference([cluster_col])

    # Get unique cluster labels
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    # Create subplots
    fig, axes = plt.subplots(n_clusters, 1, figsize=(8, 4 * n_clusters))

    # Make axes iterable in case there's only one cluster
    if n_clusters == 1:
        axes = [axes]
    import numpy as np
    # Plot one time series sample from each cluster
    rand_loc = int(np.random.randint(low=0, high=8, size=1))
    for i, cluster in enumerate(clusters):
        # Get one sample from the current cluster
        sample = df[df[cluster_col] == cluster].iloc[rand_loc][data_cols]

        # Plot the sample
        axes[i].plot(sample.values)
        axes[i].set_title(f'Cluster {cluster}'+title)
        axes[i].set_ylabel('Time Series Values')
        axes[i].set_xlabel('Time Step')

    plt.tight_layout()
    # plt.title(title)
    plt.show()



def plot_cluster_samples(clustered_df:pd.DataFrame):
    # re-write!!
    if 'Cluster' not in clustered_df.columns:
        raise ValueError("The clustered_df must contain a 'Cluster' column with cluster labels.")

    unique_clusters = clustered_df['Cluster'].unique()
    num_clusters = len(unique_clusters)

    # Create subplots, one for each cluster
    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 6 * num_clusters))

    # If there's only one cluster, wrap it in a list so we can iterate
    if num_clusters == 1:
        axes = [axes]

    # For each cluster, calculate the mean time series and plot in a separate subplot
    for i in range(num_clusters):
        # i_sample = clustered_df.loc[clustered_df['Cluster'] == i].values[0]
        # sample_timeseries = cluster_timeseries.iloc[i_sample, :]
        axes[i].plot(clustered_df.loc[clustered_df['Cluster'] == i].values[0,:], label=f'Cluster {i}')

        # Plot the mean time series on the respective subplot
        # axes[i].plot(mean_timeseries, label=f'Cluster {cluster}')

        # axes[i].plot(sample_timeseries, label=f'Sample2 {cluster}')
        axes[i].set_title(f'Mean Time Series for Cluster {i}')
        axes[i].set_xlabel('Time Points')
        axes[i].set_ylabel('Mean Value')
        axes[i].grid(True)
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Function to plot a single time series
def plot_time_series(series, title=None):
    plt.plot(series, marker='o')
    plt.title(title if title else "Time Series")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()




# Wrapper function to randomly select rows and plot time series
def plot_random_time_series(df, n_rows):
    # Randomly select n_rows rows from the dataframe
    selected_rows = df.sample(n=n_rows)

    # Plot each selected time series
    for idx, row in selected_rows.iterrows():
        plot_time_series(row.values, title=f"Time Series {idx}")

# Function to visualize the clustered data (only works well for 2 or 3 PCA components)
def plot_clusters(clustered_df, n_components=2):
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=clustered_df[0], y=clustered_df[1], hue=clustered_df['Cluster'], palette='viridis')
        plt.title("KMeans Clustering of PCA Data")
        plt.show()
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(clustered_df[0], clustered_df[1], clustered_df[2], c=clustered_df['Cluster'], cmap='viridis')
        plt.title("KMeans Clustering of PCA Data (3D)")
        plt.show()


def plot_mean_timeseries_per_cluster(data_df, clustered_df):
    """
    Plots the mean time series for each cluster.

    Parameters:
    data_df (pd.DataFrame): The original time series data (rows are samples, columns are time points).
    clustered_df (pd.DataFrame): The DataFrame containing cluster labels, must have a 'Cluster' column.
    """

    # Ensure the Cluster column exists
    if 'Cluster' not in clustered_df.columns:
        raise ValueError("The clustered_df must contain a 'Cluster' column with cluster labels.")

    # Get the unique clusters
    unique_clusters = clustered_df['Cluster'].unique()

    plt.figure(figsize=(10, 6))

    # For each cluster, calculate the mean time series
    for cluster in unique_clusters:
        # Get the rows in the original data that belong to the current cluster
        cluster_indices = clustered_df[clustered_df['Cluster'] == cluster].index
        cluster_timeseries = data_df.loc[cluster_indices]

        # Compute the mean time series
        mean_timeseries = cluster_timeseries.mean(axis=0)

        # Plot the mean time series
        plt.plot(mean_timeseries, label=f'Cluster {cluster}')

    plt.title('Mean Time Series for Each Cluster')
    plt.xlabel('Time Points')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)
    plt.show()
