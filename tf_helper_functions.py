"""
Helper Functions for Tensorflow

Note: Designed specifically for ease of use with Tensorflow modeling.
As such, functions are made to handle tensorflow specific data structures
such as the training history object or TF dataset.
"""

# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import tensorflow as tf

## General Training Functions ##

# Plot Training Curves with Plotly
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_curves(history, height=None, width=None):
    """
    Plots training curves (alongside validation if available) for 4 major machine learning metrics using Plotly.
    Will plot only those metrics which have been provided and will expand layout if necessary.
    
    Args:
        history (Dict): 
    Outputs:
        fig (Figure): Plotly figure composed of subplots containing the curves
    """
    
    metrics = ["loss","accuracy","precision","recall"]

    df = pd.DataFrame(history)
    metrics_present = [m for m in metrics if m in df.columns]

    # Determine rows/cols of subplots based on how many metrics are in data provided
    rows = math.ceil(len(metrics_present) / 2)
    cols = 1
    if len(metrics_present)>1:
        cols = 2

    # Set heigh/width if not provided (predicated on # of rows & cols)
    if not height:
        height = rows*300
    if not width: 
        width = cols*600

    fig = make_subplots(rows=rows, cols=cols,
                    subplot_titles=tuple(metrics_present),
                    vertical_spacing=0.1)

    showlegend = True
    for i, metric in enumerate(metrics_present):
        row = math.ceil((i+1)/2)
        col = i % 2 + 1
        
        fig.add_trace(go.Scatter(name=f"Training", x=df.index+1, y=df[metric],
                                mode='lines+markers', line=dict(color="#0000ff"),
                                hovertemplate=metric+': %{y:.2f}<extra></extra>',
                                legendgroup="group1",
                                showlegend=showlegend
                                ), row=row, col=col)
        if "val_"+metric in df.columns: # Validation line
        fig.add_trace(go.Scatter(name="Val", x=df.index+1, y=df[f"val_{metric}"],
                                mode='lines+markers', line=dict(color="orange"),
                                hovertemplate='Val '+ metric+': %{y:.2f}<extra></extra>',
                                legendgroup="group1",
                                showlegend=showlegend
                                ), row=row, col=col)
        showlegend = False
        
    fig.update_layout(hovermode="x unified",
                        height=height, width=width,
                        margin=go.layout.Margin(
                                    l=25, #left margin
                                    r=25, #right margin
                                    b=35, #bottom margin
                                    t=25, #top margin
                                ))
    
    return fig

def compare_histories(original, new, initial_epochs=None):
    """
    Combines training plots from TF history objects for original training and fine-tuning training epochs.
    Vertical line set at initial_epochs (or programmatically determined) will show when fine-tuning began to
    visualize impact during training.

    Args:
        original (TF history object): Training history object for model epochs prior to fine-tuning
        new (TF history object): Training history obejct for model epochs during fine-tuning
        initial_epochs (int, optional): Epoch number in which fine-tuning started. Defaults to None.

    Returns:
        Figure: Instance of plotly.graph_objects which contains all 4 major metrics present 
        (["loss","accuracy","precision","recall"]) with vertical line showing when fine-tuning started.
    """
    combined_hist = {}
    # Get the number of initial epochs from original history
    if not initial_epochs:
        initial_epochs = len(original.history[list(original.history.keys())[0]])

    # Combine original with new history
    for k in original.history.keys():
        combined_hist[k] = original.history[k] + new.history[k]
    
    # Build loss & accuracy curves for all epochs
    comp_fig = plot_curves(combined_hist)

    for i in range(len(set([d['xaxis'] for d in comp_fig.data]))):
        comp_fig.add_vline(x=initial_epochs, line_width=1, line_dash="dash", line_color="green",
                        row=math.ceil((i+1)/2),
                        col=i % 2 + 1
                        )
    return comp_fig

## Image Dataset Functions ##

def sample_data(dataset, ds_info, sample_size=3):
    """
    Shows a random sample_size of images with correpsoding labels from a Tensorflow image dataset
    Outputs images in rows of 3.

    Args:
        dataset (_type_): Tensorflow dataset of images, i.e. food101
        sample_size (int, optional): Number of images to sample. Defaults to 3.
    """
    rows = math.ceil(sample_size/3)
    
    plt.figure(figsize=(10,10))
    for i in range(sample_size):
        for images, labels in dataset.take(i+1):
            ax = plt.subplot(rows, 3, i + 1)
            plt.imshow(images.numpy())#.astype("uint8"))
            plt.title(ds_info.features["label"].names[labels.numpy()])
            plt.axis(False)

## Time-Series/Regression Functions ##

def make_sliding_windows(array: np.array, window: int, horizon: int) -> Tuple:   
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    Example: [1,2,3,4,5,6,7] -> ([1,2,3,4,5,6], [7]) window: 6, horizon: 1

    Args:
        array (np.array): 1D array of data
        window (int): Size of window
        horizon (int): Size of horizon

    Returns:
        Tuple: Sequential labelled windows in the form (windows, labels)
    """
    sub_windows = (
            0 +
            # expand_dims are used to convert a 1D array to 2D array.
            np.expand_dims(np.arange(window+horizon), 0) +
            np.expand_dims(np.arange(len(array) - window), 0).T
        )

    windowed_array = np.array(array)[sub_windows]

    windows, labels =  windowed_array[:, :-horizon], windowed_array[:, -horizon:]

    return windows, labels

def mean_absolute_scaled_error(y_true, y_pred, naive_offset=1) -> float:
  """
  Implement MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))

  # Find MAE of naive forecast (no seasonality)
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[naive_offset:] - y_true[:-naive_offset]))

  return mae / mae_naive_no_season

def evaluate_forecast_results(y_true, y_pred) -> Dict:
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}

