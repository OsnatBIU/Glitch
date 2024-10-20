import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the moving average function
from preprocess_data import get_data


def moving_avg(df, window_size):
    return df.rolling(window=window_size).mean()





def test_exponential_smoothing(df, alphas):
    """Test exponential smoothing with different alpha values on the same plot."""
    plt.figure(figsize=(10, 6))

    # Loop through each alpha, apply smoothing, and plot the results
    for alpha in alphas:
        smoothed_df = exponential_smoothing(df, alpha)

        # Plotting the smoothed values (mean over all rows, one line per alpha)
        plt.plot(smoothed_df.mean(axis=0), label=f'alpha={alpha}')

    plt.title("Exponential Smoothing with Different Alphas")
    plt.xlabel("Time")
    plt.ylabel("Smoothed Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def exponential_smoothing(df, alpha):
    """Apply exponential smoothing to each row of a DataFrame."""
    return df.apply(lambda row: row.ewm(alpha=alpha).mean(), axis=1)


def test_exponential_smoothing_single_row(df, alphas, row_idx=0):
    """Test exponential smoothing with different alpha values for a specific row."""
    plt.figure(figsize=(10, 6))

    # Select the row by its index
    original_row = df.iloc[row_idx]
    plt.plot(original_row, label='Original', linestyle='--', color='gray')

    # Loop through each alpha, apply smoothing, and plot the result for the chosen row
    for alpha in alphas:
        smoothed_row = exponential_smoothing(df, alpha).iloc[row_idx]

        # Plotting the smoothed values for the selected row
        plt.plot(smoothed_row, label=f'alpha={alpha}')

    plt.title(f"Exponential Smoothing on Row {row_idx} with Different Alphas")
    plt.xlabel("Time")
    plt.ylabel("Smoothed Values")
    plt.legend()
    plt.grid(True)
    plt.show()



# Example usage
if __name__ == "__main__":
    import os

    script_dir = (os.path.dirname(os.path.abspath(__file__)))

    x_df = get_data(os.path.dirname(script_dir))

    # Define list of alphas to test
    alphas = [0.01, 0.3, 0.5]

    # Call the test function with a window size of 5 for the moving average
    # test_exponential_smoothing(x_df, alphas)
    window_sizes = [2, 3, 5]
    # Define the list of alphas to test
    alphas = [0.1,  0.05]

    # Call the test function for a specific row (e.g., row 0)
    test_exponential_smoothing_single_row(x_df, alphas, row_idx=0)
