import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt

from sklearn.decomposition import PCA
from scipy.fftpack import fft

from preprocess_data import get_data, get_true_events
from collect_and_preprocess_data import plot_mean_timeseries_per_cluster_subplots, extract_event_series

from sklearn.cluster import KMeans
import seaborn as sns

import pandas as pd
from sklearn.cluster import KMeans


def cluster_features_with_kmeans(index_df, num_clusters):
    """
    Extracts the 'amplitude (DVA)', 'direction (deg)', and 'max velocity' columns from the index DataFrame,
    and performs KMeans clustering on these features.

    Parameters:
    index_df (pd.DataFrame): Input DataFrame containing columns: 'trialNum', 'onsetIdx', 'offsetIdx', 'amplitude (DVA)',
                             'direction (deg)', 'max velocity'.
    num_clusters (int): The number of clusters to form.

    Returns:
    clustered_df (pd.DataFrame): The original index_df with an additional 'Cluster' column indicating the cluster assignment.
    kmeans (KMeans): The fitted KMeans model.
    """

    # Extract the relevant features: 'amplitude (DVA)', 'direction (deg)', and 'max velocity'
    feature_columns = ['amplitude (DVA)', 'direction (deg)', 'max velocity']
    features = index_df[feature_columns]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    index_df['Cluster'] = kmeans.fit_predict(features)

    return index_df, kmeans


if __name__=="__main__":
    x_df = get_data()
    indicators_df = get_true_events()
    events_df = extract_event_series(x_df, indicators_df)
    index_df, kmeans = cluster_features_with_kmeans(indicators_df,3)
    plot_mean_timeseries_per_cluster_subplots(events_df, index_df)
