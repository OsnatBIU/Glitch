import numpy as np
import pandas as pd
import pywt  # For wavelet transform


def do_dwt(df, wavelet='db4', level=3):
    """
    Normalize time series data and extract DWT features.

    Parameters:
    - df: DataFrame, rows are samples, columns are time points
    - wavelet: str, type of wavelet (default is Daubechies 4)
    - level: int, decomposition level for DWT

    Returns:
    - dwt_features: np.array, transformed DWT coefficients for each sample
    """
    # Standardize data

    # Initialize a list to hold the DWT coefficients
    dwt_features = []

    # Apply DWT on each row (time series sample)
    for index, row in df.iterrows():
    # for index in range(df.shape[0]):
        # Perform DWT
        coeffs = pywt.wavedec(row.values, wavelet=wavelet, level=level)
        # coeffs = pywt.wavedec(df[index, :], wavelet=wavelet, level=level)
        # Concatenate all coefficients to form the feature vector
        coeffs_concat = np.concatenate(coeffs)
        dwt_features.append(coeffs_concat)

    # Convert to a numpy array
    dwt_features = np.array(dwt_features)

    # return dwt_features
    return pd.DataFrame(dwt_features, index=df.index)

# Example of using the function
# df is your DataFrame with ~50 columns (time points)
# dwt_features = normalize_and_dwt(df)
