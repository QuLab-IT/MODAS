import numpy as np
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import normalize

def extract_spectrogram_features(Sxx, freqs, times, n_bands=20):
    features = {}

    # Normalize Sxx to avoid division by zero
    Sxx = np.nan_to_num(Sxx)
    Sxx_sum = np.sum(Sxx)
    if Sxx_sum > 0:
        Sxx /= Sxx_sum

    # Basic global stats
    features.update({
        'mean_power': np.mean(Sxx),
        'std_power': np.std(Sxx),
        'max_power': np.max(Sxx),
        'min_power': np.min(Sxx),
        'median_power': np.median(Sxx),
        'var_power': np.var(Sxx),
        'power_range': np.ptp(Sxx),  # peak-to-peak
    })

    # Temporal & Spectral stats
    time_stats = np.mean(Sxx, axis=0)
    freq_stats = np.mean(Sxx, axis=1)

    features.update({
        'temporal_variance': np.var(time_stats),
        'spectral_variance': np.var(freq_stats),
        'spectral_entropy': entropy(freq_stats / np.sum(freq_stats)),
        'temporal_entropy': entropy(time_stats / np.sum(time_stats)),
        'spectral_kurtosis': kurtosis(freq_stats),
        'spectral_skewness': skew(freq_stats),
        'temporal_kurtosis': kurtosis(time_stats),
        'temporal_skewness': skew(time_stats),
        'spectral_centroid': np.sum(freqs * freq_stats) / np.sum(freq_stats),
    })

    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        features[f'power_p{p}'] = np.percentile(Sxx, p)

    # Band energy distribution
    band_edges = np.linspace(0, len(freqs), n_bands+1, dtype=int)
    for i in range(n_bands):
        band = Sxx[band_edges[i]:band_edges[i+1], :]
        band_mean = np.mean(band)
        features[f'band_{i}_mean'] = band_mean
        features[f'band_{i}_std'] = np.std(band)
        features[f'band_{i}_entropy'] = entropy(np.mean(band, axis=1) + 1e-12)

    # Derivative features (spectral flux)
    flux = np.diff(Sxx, axis=1)
    features['spectral_flux_mean'] = np.mean(np.abs(flux))
    features['spectral_flux_std'] = np.std(flux)

    # Log transform for sparsity features
    Sxx_log = np.log1p(Sxx)
    features['log_power_mean'] = np.mean(Sxx_log)
    features['log_power_std'] = np.std(Sxx_log)

    # Row-wise (frequency-wise) stats
    row_mean = np.mean(Sxx, axis=1)
    row_std = np.std(Sxx, axis=1)
    features['row_mean_mean'] = np.mean(row_mean)
    features['row_std_mean'] = np.mean(row_std)
    features['row_mean_std'] = np.std(row_mean)
    features['row_std_std'] = np.std(row_std)

    # Column-wise (temporal) stats
    col_mean = np.mean(Sxx, axis=0)
    col_std = np.std(Sxx, axis=0)
    features['col_mean_mean'] = np.mean(col_mean)
    features['col_std_mean'] = np.mean(col_std)
    features['col_mean_std'] = np.std(col_mean)
    features['col_std_std'] = np.std(col_std)

    return features
