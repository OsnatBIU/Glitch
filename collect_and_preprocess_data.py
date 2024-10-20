import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt

from sklearn.decomposition import PCA
from scipy.fftpack import fft

from preprocess_data import get_data, get_true_events

from sklearn.cluster import KMeans
import seaborn as sns

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


import matplotlib.pyplot as plt
import pandas as pd


def plot_cluster_timeseries(df, cluster_col):
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

    # Plot one time series sample from each cluster
    for i, cluster in enumerate(clusters):
        # Get one sample from the current cluster
        sample = df[df[cluster_col] == cluster].iloc[0][data_cols]

        # Plot the sample
        axes[i].plot(sample.values)
        axes[i].set_title(f'Cluster {cluster}')
        axes[i].set_ylabel('Time Series Values')
        axes[i].set_xlabel('Time Step')

    plt.tight_layout()
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



def convert_to_numeric(df):
    df_copy = df.copy()  # Work with a copy of the dataframe to avoid modifying the original
    for col in df.columns:
        if df_copy[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_copy[col]):
            df_copy[col] = pd.Categorical(df_copy[col]).codes
    return df_copy


def map_labels_to_values(df, porc_dict):
    mapping_dict = {'step': 1, 'glitch': 0, 'other': 0, 'drift': 0} #{'category': }
    df_copy = df.copy()
    for col in df.columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            if porc_dict['map_labels_to_values']=='convert_to_numeric':
                df_copy[col] = pd.Categorical(df_copy[col]).codes
            elif porc_dict['map_labels_to_values']=='convert_by_dict':
                df_copy[col] = df_copy[col].map(mapping_dict)
    return df_copy



def extract_event_series(data_df, indicators_df, ticks=25):
    # Initialize a list to store the extracted time series
    extracted_series = []

    # Iterate through each row in the indicators DataFrame
    for index, row in indicators_df.iterrows():
        # Get trial number, onset index, and the corresponding trial time series
        trial_num = int(row['trialNum']) - 1  # ASK YARDEN!! is trialNum 1-based??
        onset_idx = int(row['onsetIdx'])
        # Define the window: 25 time points before onset, onset itself, 24 after onset
        start_idx = onset_idx - ticks
        end_idx = onset_idx + ticks

        # Handle extreme case, op1: fix
        # if onset_idx<ticks:
        #     start_idx=0
        # if end_idx > data_df.shape[1]:
        #     end_idx = data_df.shape[1]
        # # op2: disregard
        # if onset_idx<ticks or end_idx > data_df.shape[1]:
        #     continue
        # Extract the relevant time series from the corresponding row in data_df
        series = data_df.iloc[trial_num, start_idx:end_idx].values

        # Append the extracted series to the list
        extracted_series.append(series)

    # Convert the list of series into a DataFrame
    extracted_df = pd.DataFrame(extracted_series)

    return extracted_df

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


def extract_fft_features(df):
    fft_features = []
    # Probably useless
    # Compute FFT for each row (time series) in the DataFrame
    for _, row in df.iterrows():
        # Apply FFT to the time series and take the absolute value of the frequencies
        fft_values = np.abs(fft(row.values))

        # Append the real part of the FFT values (first half, since FFT is symmetric)
        fft_features.append(fft_values[:len(fft_values) // 2])

    # Create a DataFrame for FFT features
    fft_df = pd.DataFrame(fft_features)

    return fft_df

# also try waveletts adn spectrogram
def extract_derivatives(df):
    derivatives = []

    for _, row in df.iterrows():  # Iterating over each row in the DataFrame
        derivative = np.abs(np.diff(row.values, prepend=row.values[0]))  # Calculating derivative (finite difference)
        derivatives.append(derivative)

    derivative_df = pd.DataFrame(derivatives, columns=df.columns, index=df.index)
    return derivative_df


def apply_pca(features_df, n_components=10):
    # Apply PCA on the input features DataFrame
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_df)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result)

    return pca_df

def do_smoothing(df, porc_dict, arg_dict):
    if porc_dict['do_smooth'] == 'rolling_avg':
        rolling_avg = df.apply(lambda row: row.rolling(window=arg_dict['window_size'], min_periods=1).mean(), axis=1)
        return rolling_avg
    if porc_dict['do_smooth'] == 'exp_avg':
        exp_avg = df.apply(lambda row: row.ewm(alpha=arg_dict['exp_avg_alpha']).mean(), axis=1)
        return exp_avg
# can you write a function calculate the mean using a sliding window  and
# the change of mean, input is a dataframe where each row is a time series sample,
# output should be a dataframe with differences in mean

def data_processing_pipeline(df, porc_dict, arg_dict):
    # assert df.isnull().sum().sum()==0, 'data_processing_pipeline: null found, check df'
    df.fillna(0, inplace=True)
    if porc_dict['do_smooth'] is not None:
        df = do_smoothing(df,porc_dict, arg_dict)
    if porc_dict['do_deriv'] ==True:
        df = extract_derivatives(df)
    if porc_dict['do_pca']==True:
        df = apply_pca(df, arg_dict['n_components'])
    return df



# Function to perform KMeans clustering on PCA DataFrame
def cluster_df_data(df, clustering_method='kmeans', n_clusters=3):
    if clustering_method=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df)

    # Add the cluster labels to the PCA DataFrame
    clustered_df = df.copy()
    clustered_df['Cluster'] = clusters

    return clustered_df, kmeans


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


def extract_true_labels():
    pass


if __name__=="__main__":
    porc_dict = {'do_deriv':False, 'do_pca':False, 'do_smooth': None, #'exp_avg',
                 'map_labels_to_values': 'convert_by_dict'}
    arg_dict = {'n_components':12, 'exp_avg_alpha':0.1,'n_clusters':2}

    x_df = get_data()
    indicators_df = get_true_events()
    indicators_df = map_labels_to_values(indicators_df,porc_dict)

    events_df = extract_event_series(x_df, indicators_df, ticks=25) # we want a window of 100msec, tick=4msec
    true_labels_x = indicators_df['Xclass'].values # same for y
    # print(true_events_df.head())
    # print(events_df.head())
    # plot_random_time_series(events_df, 8)  # Randomly plot 3 time series
    #
    # process data

    processed_df = data_processing_pipeline(events_df, porc_dict, arg_dict)
    #fit

    clustered_df, kmeans = cluster_df_data(processed_df,
                                           clustering_method='kmeans', n_clusters=arg_dict['n_clusters'])

    clustered_df.to_csv(os.path.join('Data', 'clustered_df.csv'), index=False)
    from sklearn.metrics import fowlkes_mallows_score
    from sklearn.metrics import precision_score, accuracy_score

    fmi = fowlkes_mallows_score(true_labels_x, clustered_df['Cluster'].values)
    print(clustered_df['Cluster'].values)
    print(true_labels_x)
    print(f"Fowlkes-Mallows Index: {fmi}")
    events_df['Cluster']=true_labels_x



    accuracy_ = accuracy_score(true_labels_x, clustered_df['Cluster'].values) #, precision_score(true_values, predictions)
    print(accuracy_)
    plot_cluster_timeseries(events_df,'Cluster')
    #
    # # Plot the results (works for PCA with 2 or 3 components)
    # plot_clusters(clustered_df, n_components=3)
    # print(len(events_df), len(clustered_df))
    # # plot_mean_timeseries_per_cluster(events_df, clustered_df)
    # plot_mean_timeseries_per_cluster_subplots(events_df, clustered_df)

    # cut events and put in DF with label
