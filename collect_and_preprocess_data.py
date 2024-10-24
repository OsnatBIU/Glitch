import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from modeling import cluster_df_data, apply_clustering_methods
from preprocess_data import get_data, get_true_events
from preprocessing import map_labels_to_values, do_smoothing
from visualization import plot_cluster_timeseries
from waveletts_functions import do_dwt

def extract_event_series(data_df, indicators_df, ticks=25):
    extracted_series = []
    for index, row in indicators_df.iterrows():
        # Get trial number, onset index, and the corresponding trial time series
        trial_num = int(row['trialNum']) - 1  # ASK YARDEN!! is trialNum 1-based??
        onset_idx = int(row['onsetIdx'])
        # Define the window: 25 time points before onset, onset itself, 24 after onset
        start_idx = onset_idx - ticks
        end_idx = onset_idx + ticks
        series = data_df.iloc[trial_num, start_idx:end_idx].values
        extracted_series.append(series)
    extracted_df = pd.DataFrame(extracted_series)
    return extracted_df

def extract_derivatives(df):
    derivatives = []
    for _, row in df.iterrows():  # Iterating over each row in the DataFrame
        derivative = np.abs(np.diff(row.values, prepend=row.values[0]))  # Calculating derivative (finite difference)
        derivatives.append(derivative)
    derivative_df = pd.DataFrame(derivatives, columns=df.columns, index=df.index)
    return derivative_df


def apply_pca(features_df, n_components=10):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_df)
    pca_df = pd.DataFrame(pca_result)
    return pca_df

# can you write a function calculate the mean using a sliding window  and
# the change of mean, input is a dataframe where each row is a time series sample,
# output should be a dataframe with differences in mean

def data_processing_pipeline(df, porc_dict, arg_dict):
    # assert df.isnull().sum().sum()==0, 'data_processing_pipeline: null found, check df'
    df.fillna(0, inplace=True)
    if porc_dict['do_standartization']==True:
        from sklearn.preprocessing import StandardScaler
        scaled_features = StandardScaler().fit_transform(df.values)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    if porc_dict['do_smooth'] is not None:
        df = do_smoothing(df,porc_dict, arg_dict)
    # if porc_dict['do_deriv'] ==True: -- we have in in feature extraction
    #     df = extract_derivatives(df)
    # if porc_dict['do_pca']==True:
    #     df = apply_pca(df, arg_dict['n_components'])
    return df

def feature_extraction_pipeline(df, porc_dict, arg_dict):
    if porc_dict['do_deriv'] ==True:
        df = extract_derivatives(df)
    if porc_dict['do_dwt']:
        df = do_dwt(df, wavelet=arg_dict['wavelet'], level=arg_dict['dwt_level'])
    if porc_dict['do_pca']==True:
        df = apply_pca(df, arg_dict['pca_n_components'])
    return df


def get_data_and_labels(porc_dict, arg_dict):
    x_df = get_data(axis='x')
    y_df = get_data(axis='y')
    indicators_df = get_true_events()
    indicators_df = map_labels_to_values(indicators_df, porc_dict, arg_dict)
    events_dfx = extract_event_series(x_df, indicators_df, ticks=arg_dict['n_ticks'])  # we want a window of 100msec, tick=4msec
    events_dfy = extract_event_series(y_df, indicators_df, ticks=arg_dict['n_ticks'])
    events_df = pd.concat([events_dfx,events_dfy])
    true_labels_x = indicators_df['Xclass'].values  # same for y
    true_labels_y = indicators_df['Xclass'].values
    true_labels = np.concatenate((true_labels_x,true_labels_y),axis=None)
    return events_df, true_labels


if __name__=="__main__":
    porc_dict = {'do_standartization': False, 'do_smooth': None, #'exp_avg',  # 'exp_avg',None 'do_deriv': False, 'do_pca': False,
                 'map_labels_to_values': 'convert_by_dict'} #convert_by_dict / convert_to_numeric
    arg_dict = {'n_ticks':20, 'exp_avg_alpha': 0.15, 'n_clusters': 3, #'n_components': 12,
                'mapping_dict': {'step': 1, 'glitch': 0, 'other': 0, 'drift': 0}}
    # --- Parms for Feature Extraction: ----
    fe_dict={'do_dwt':False, 'do_deriv':False, 'do_pca':False}
    fe_args_dict={'wavelet':'db1', 'dwt_level':3,'pca_n_components':12} #db1, db4
    #---------------------------------------------------------------
    # -- READ & ORDER DATA --- #
    events_df, true_labels = get_data_and_labels(porc_dict, arg_dict)
    events_df['labels'] = true_labels
    #
    events_df.dropna(axis=0, inplace=True)
    true_labels = events_df['labels']
    events_df.drop(columns=['labels'], inplace=True)
    print(events_df.shape)
    # events_df.to_csv(os.path.join('Data', 'events_trueLables.csv'), index=False)

    # -- PREPROCESS --- #
    # processed_df = data_processing_pipeline(events_df, porc_dict, arg_dict)

    # -- EXTRACT FEATURES --- #
    # processed_df = feature_extraction_pipeline(processed_df, fe_dict, fe_args_dict)

    processed_df = pd.read_csv('Data/scaled_dot_with_two_top_PCs.csv')
    # --- Fit ----#
    # cluster_methods = ['kmeans', 'dbscan', 'spectral', 'optics']
    cluster_methods = ['kmeans']
    clustered_df = apply_clustering_methods(processed_df, cluster_methods,
                                            n_clusters = arg_dict['n_clusters'])

    # aggregate cluster results:
    # Todo

    # #old option
    # clustered_df, kmeans = cluster_df_data(processed_df,
    #                                        clustering_method='kmeans', n_clusters=arg_dict['n_clusters'])
    # --- save --- #
    # clustered_df.to_csv(os.path.join('Data', 'clustered_df_pca.csv'), index=False)

    # --- Evaluate Clustering ----#
    from sklearn.metrics import fowlkes_mallows_score
    from sklearn.metrics import precision_score, accuracy_score

    cluster_labels_col = cluster_methods[0] + '_labels'

    fmi = fowlkes_mallows_score(true_labels, clustered_df[cluster_labels_col].values)
    print("Predicted: ",  clustered_df[cluster_labels_col].values)
    print('True: ', true_labels)
    print(f"Fowlkes-Mallows Index: {fmi}")

    accuracy_ = accuracy_score(true_labels, clustered_df[cluster_labels_col].values) #, precision_score(true_values, predictions)
    print('Accracy: %.3f' % accuracy_)
    from sklearn.metrics import recall_score, precision_score
    recall = recall_score(true_labels, clustered_df[cluster_labels_col].values, average='macro') #, average='binary')
    print('Recall: %.3f' % recall)
    # recall = recall_score(clustered_df[cluster_labels_col].values, true_labels)  # , average='binary')
    # print('Reverese Recall (how many of the predeicted are true: %.3f' % recall)


    # precision = precision_score(true_labels, clustered_df[cluster_labels_col].values)
    # print('Precision: %.3f' % precision)

    print('Fraction of ones: ', clustered_df[cluster_labels_col].values.sum()/len(clustered_df[cluster_labels_col].values))

    # plot predicted
    events_df[cluster_labels_col] = clustered_df[cluster_labels_col]
    plot_cluster_timeseries(events_df, cluster_labels_col, ' Predicted')

    events_df[cluster_labels_col] = clustered_df[cluster_labels_col]
    plot_cluster_timeseries(events_df, cluster_labels_col, ' Predicted')

    # predicted processed
    plot_cluster_timeseries(clustered_df, cluster_labels_col, ' ProcFeaturesPredicted')

    # Plot True (row data sample per cluster)
    events_df[cluster_labels_col] = true_labels
    plot_cluster_timeseries(events_df,cluster_labels_col, ' True')


    # Todo: plot samples where clustering is wrong
