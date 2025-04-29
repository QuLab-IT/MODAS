import numpy as np
from scipy.stats import entropy, kurtosis, skew, linregress
from scipy.signal import find_peaks
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
    band_energies = []
    for i in range(n_bands):
        band = Sxx[band_edges[i]:band_edges[i+1], :]
        band_mean = np.mean(band)
        band_energy = np.sum(band)
        band_energies.append(band_energy)
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

    # 1. Spectral Flatness
    features['spectral_flatness'] = np.exp(np.mean(np.log(freq_stats + 1e-12))) / (np.mean(freq_stats) + 1e-12)

    # 2. Spectral Spread
    centroid = features['spectral_centroid']
    features['spectral_spread'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * freq_stats) / np.sum(freq_stats))

    # 3. Temporal Flatness
    features['temporal_flatness'] = np.exp(np.mean(np.log(time_stats + 1e-12))) / (np.mean(time_stats) + 1e-12)

    # 4. Max Band Energy Ratio
    features['max_band_energy_ratio'] = np.max(band_energies) / np.sum(Sxx)

    # 5. High Frequency Content (Top 25%)
    high_freq_band = Sxx[int(0.75 * len(freqs)):, :]
    features['high_freq_content'] = np.sum(high_freq_band) / np.sum(Sxx)

    # 6. Temporal Center of Mass
    features['temporal_centroid'] = np.sum(times * time_stats) / np.sum(time_stats)

    # 7. Spectral Roll-off (85%)
    cumulative = np.cumsum(freq_stats)
    roll_off_index = np.where(cumulative >= 0.85 * cumulative[-1])[0][0]
    features['spectral_rolloff_85'] = freqs[roll_off_index]

    # 8. Spectral Crest Factor
    features['spectral_crest'] = np.max(freq_stats) / (np.mean(freq_stats) + 1e-12)

    # 9. Temporal Crest Factor
    features['temporal_crest'] = np.max(time_stats) / (np.mean(time_stats) + 1e-12)

    # 10. Spectral Slope
    slope, _, _, _, _ = linregress(freqs, freq_stats)
    features['spectral_slope'] = slope

    # 11. Spectral Contrast (Peak vs Valley)
    peaks, _ = find_peaks(freq_stats)
    valleys, _ = find_peaks(-freq_stats)
    peak_mean = np.mean(freq_stats[peaks]) if len(peaks) > 0 else 0
    valley_mean = np.mean(freq_stats[valleys]) if len(valleys) > 0 else 1e-12
    features['spectral_contrast'] = peak_mean / valley_mean

    # 12. Zero-Crossing Rate (Temporal and Spectral)
    features['temporal_zero_crossing'] = np.mean(np.diff(np.sign(time_stats)) != 0)
    features['spectral_zero_crossing'] = np.mean(np.diff(np.sign(freq_stats)) != 0)

    # 13. Modulation Energy / Variation
    temporal_mod_energy = np.mean(np.diff(time_stats) ** 2)
    spectral_mod_energy = np.mean(np.diff(freq_stats) ** 2)
    features['temporal_mod_energy'] = temporal_mod_energy
    features['spectral_mod_energy'] = spectral_mod_energy

    # 14. Spectrogram Sharpness
    sharpness = np.mean(np.abs(np.gradient(Sxx)))
    features['spectrogram_sharpness'] = sharpness

    # 15. Texture-based Features (Row and Column Differences)
    diff_rows = np.diff(Sxx, axis=0)
    diff_cols = np.diff(Sxx, axis=1)
    features['texture_row_var'] = np.var(diff_rows)
    features['texture_col_var'] = np.var(diff_cols)

    return features

#returns 111 features right now 