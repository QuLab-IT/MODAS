import numpy as np
from scipy import signal
from tqdm import tqdm

def DeMean_data(Data_2D, axis = 1):
    # Subtract the mean along the specified axis in-place
    return Data_2D - np.mean(Data_2D, axis=axis, keepdims=True)

def DeTrend_data(Data_2D, axis = 1):
    # Use scipy.detrend directly as it removes both trend and mean
    return signal.detrend(Data_2D, axis=axis)

def Normalize_MinMax(data, axis=None):
    # Calculate min and max along the specified axis
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    
    # Avoid division by zero by replacing constant regions with zeros
    range_ = data_max - data_min
    normalized_data = (data - data_min) / np.where(range_ == 0, 1, range_)
    return normalized_data

def Normalize_Zscore(data, axis=None):
    # Calculate mean and standard deviation along the specified axis
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    
    # Perform Z-score normalization
    normalized_data = (data - mean) / std
    return normalized_data

def temporal_decimation(data, original_rate, target_rate):
    if original_rate < target_rate:
        raise ValueError(f"Original rate must be higher than target rate for temporal decimation: {original_rate} < {target_rate}")
    
    # Calculate decimation factor and slice data without intermediate copy
    decimation_factor = int( original_rate // target_rate )
    return data[:, ::decimation_factor]

def spatial_stacking(data, median_window=5, mean_window=2):
    #Apply median and mean stacking to reduce data size
    num_traces, num_samples = data.shape
    truncated_data = data[:num_traces // (median_window * mean_window) * median_window * mean_window]
    
    # Median stacking: Reshape and apply median along the correct axis
    median_stacked = np.median(truncated_data.reshape(-1, median_window, num_samples), axis=1)

    # Mean stacking: Reshape and apply mean along the correct axis
    mean_stacked = np.mean(median_stacked.reshape(-1, mean_window, num_samples), axis=1)
    
    return mean_stacked

def cross_correlation(daily_data: np.ndarray):
    n_days   = daily_data.shape[0]
    corr_arr = np.zeros((n_days, n_days))
    
    # Precompute norms
    norms    = [np.linalg.norm(daily_data[i]) for i in range(n_days)]

    # Compute cross-correlation for each pair of days
    for i in tqdm(range(n_days)):
        for j in tqdm(range(i + 1, n_days)):  # j starts at i to avoid redundant calculations
            # Compute normalized cross-correlation
            corr     = signal.correlate2d(daily_data[i], daily_data[j], mode='same', boundary='wrap')
            max_corr = np.max(corr)
            # Normalize by product of individual norms to get correlation coefficient range [-1, 1]
            norm_product = norms[i] * norms[j]
            cross_corr   = max_corr / norm_product if norm_product != 0 else 0
            # Assign the result to both (i, j) and (j, i) in the symmetric matrix
            corr_arr[i, j] = cross_corr
            corr_arr[j, i] = cross_corr

    return corr_arr

