import pandas as pd
from scipy.signal import savgol_filter

def convert_to_numeric(df):
    df_copy = df.copy()  # Work with a copy of the dataframe to avoid modifying the original
    for col in df.columns:
        if df_copy[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_copy[col]):
            df_copy[col] = pd.Categorical(df_copy[col]).codes
    return df_copy

def map_labels_to_values(df, porc_dict, arg_dict):
    df_copy = df.copy()
    for col in df.columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            if porc_dict['map_labels_to_values']=='convert_to_numeric':
                df_copy[col] = pd.Categorical(df_copy[col]).codes
            elif porc_dict['map_labels_to_values']=='convert_by_dict':
                df_copy[col] = df_copy[col].map(arg_dict['mapping_dict'])
    return df_copy

def do_smoothing(df, porc_dict, arg_dict):
    if porc_dict['do_smooth'] == 'rolling_avg':
        rolling_avg = df.apply(lambda row: row.rolling(window=arg_dict['window_size'], min_periods=1).mean(), axis=1)
        return rolling_avg
    if porc_dict['do_smooth'] == 'exp_avg':
        exp_avg = df.apply(lambda row: row.ewm(alpha=arg_dict['exp_avg_alpha']).mean(), axis=1)
        return exp_avg
    if porc_dict['do_smooth'] == 'savgol':
        return df.apply(lambda row: savgol_filter(arg_dict['savgol_window_length'], arg_dict['savgol_polyorder']), axis=1)


