import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_random_time_series_per_label(df, label_col):
    labels = df[label_col].unique()  # Get the unique labels
    n_labels = len(labels)

    fig, axes = plt.subplots(n_labels, 1, figsize=(10, 5 * n_labels))  # Create subplots

    # Ensure axes is iterable even if there's only one label
    if n_labels == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        label_data = df[df[label_col] == label].drop(columns=[label_col])  # Data of the current label
        x = range(len(label_data.columns))
        for _ in range(1):
            random_samples = label_data.sample(1)  # Select 3 random rows (time series) for this label
            axes[i].plot(x,random_samples.values[0])
        # for sample in random_samples.values:
        #     axes[i].plot(sample.values)  # Plot each time series (row) as a separate line

        axes[i].set_title(f"Label: {label}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv(os.path.join('../Data','clustered_df.csv'),index_col=0)
    print(df.columns)
    plot_random_time_series_per_label(df, 'Cluster')