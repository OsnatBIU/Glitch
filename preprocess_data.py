import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt

def get_data(path=''):
    x_df = pd.read_csv(os.path.join(path,'Data/eyeXVec.csv'))
    x_df = x_df.transpose()
    x_df.reset_index(drop=True, inplace=True)
    return x_df
    # print(x_data.head())

def get_true_events():
    return pd.read_csv('Data/MSduringVecs.csv')

def get_min_max(df):
    # return min max values across all columns and rows
    min_val = df.min().min()
    max_val = df.max().max()
    return min_val, max_val

def detect_events_in_df(df, window_size=10, threshold=None, step_size=1):
    all_events = []
    for idx, row in df.iterrows():
        events = detect_events(row.values, window_size=window_size, threshold=threshold, step_size=step_size)
        if events:  # If events were found in this row
            all_events.append((idx, events))  # Append the row index and the list of events

    return all_events

def detect_events(time_series, window_size=10, threshold=1.0, step_size=1):
    events = []
    n = len(time_series)

    for start in range(0, n - window_size + 1, step_size):
        window = time_series[start:start + window_size]

        # Detect an event based on the max-min difference in the window
        if np.max(window) - np.min(window) > threshold:
            events.append((start, start + window_size - 1))

    return events



def plot_event_in_series(df, row_idx, event=None):
    # Extract the time series for the given row
    time_series = df.iloc[row_idx].values
    # start, end = event

    # Plot the time series
    plt.figure(figsize=(10, 4))
    plt.plot(time_series, label=f"Series {row_idx}")

    # # Add a colored rectangle for the event
    # ax = plt.gca()
    # rect = Rectangle((start, np.min(time_series)), end - start, np.max(time_series) - np.min(time_series),
    #                  color='red', alpha=0.3, label="Event")
    # ax.add_patch(rect)

    # Add labels and legend
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.title(f"Time Series {row_idx} with Event Highlighted")
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_all_events_in_series(data_df, row, events, true_events_df):
    # Extract the time series data for the specified row
    time_series = data_df.iloc[row]

    # Plot the time series data
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Time Series', color='blue')

    # Plot the detected events (assuming `events` is a list of (onset, offset) tuples)
    for event in events:
        onset, offset = event
        plt.axvspan(onset, offset, color='orange', alpha=0.3, label='Detected Event' if event == events[0] else "")

    # Plot the true events for this specific row (trial)
    trial_num = row + 1  # Assuming trialNum matches row index + 1
    true_events = true_events_df[true_events_df['trialNum'] == trial_num]

    for idx, true_event in true_events.iterrows():
        onset_true = true_event['onsetIdx']
        offset_true = true_event['offsetIdx']
        rect = patches.Rectangle((onset_true, min(time_series)),
                                 offset_true - onset_true,
                                 max(time_series) - min(time_series),
                                 linewidth=1, edgecolor='r', facecolor='green', alpha=0.2, label='True Event')
        plt.gca().add_patch(rect)

    # Add labels, legend, and show plot
    plt.xlabel('Time Steps')
    plt.ylabel('Signal')
    plt.legend(loc='upper right')
    plt.title(f'Time Series and Events for Row {row}')
    plt.show()


# Example usage:
# plot_all_events_in_series(data_df, row=3, events=detected_events, true_events_df=true_events)


def high_pass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def high_pass_filter_df(df, cutoff=0.1, order=5):
    filtered_data = df.apply(lambda row: high_pass_filter(row.values, cutoff=cutoff, order=order), axis=1)
    df_filtered = pd.DataFrame(filtered_data.tolist(), columns=df.columns, index=df.index)
    return df_filtered


def detect_events_from_data(df):
    x_dff = high_pass_filter_df(x_df, cutoff=0.05)
    print(len(x_dff.columns))
    global_min, global_max = get_min_max(x_dff)
    threshold = (global_max - global_min) / 2
    return detect_events_in_df(x_dff, window_size=10, threshold=threshold, step_size=5)

if __name__=="__main__":
    x_df = get_data()
    all_events = detect_events_from_data(x_df)
    print("Detected events:", all_events)
    true_events = get_true_events()
    # plot_event_in_series(df, row_idx, event)
    plot_all_events_in_series(x_df, 5, [(170, 179), (175, 184)], true_events)

    # # x_df, all_events = detect_events_from_data()
    # true_events = get_true_events()
    # print(len(true_events.columns))
    # print(len(true_events))

    # plot_event_in_series(x_df, row_idx=3)

    # plot_all_events_in_series(x_df, row_idx=0, events=[(0, 9), (20, 29), (25, 34), (30, 39)])
    # wd = os.getcwd()
    # print("Curr working dir: ",os. getcwd())

