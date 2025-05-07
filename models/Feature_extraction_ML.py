import numpy as np
from scipy.stats import entropy, kurtosis, skew, linregress
from scipy.signal import find_peaks, stft, hilbert
from sklearn.preprocessing import normalize
from scipy.fft import fft

def extract_spectrogram_features(Sxx, freqs, times, n_bands=20):
    features = {}

    # Normalize Sxx to avoid division by zero
    Sxx = np.nan_to_num(Sxx)
    Sxx_sum = np.sum(Sxx)
    if Sxx_sum > 0:
        Sxx /= Sxx_sum

    # Time Domain Features
    time_stats = np.mean(Sxx, axis=0)
    features.update({
        'mean_amplitude': np.mean(time_stats),
        'zero_crossing_rate': np.mean(np.diff(np.sign(time_stats)) != 0),
        'duration': np.sum(time_stats > 0),
        'rise_time': np.argmax(time_stats > 0),
        'decay_time': len(time_stats) - np.argmax(time_stats[::-1] > 0),
        'time_skewness': skew(time_stats),
        'time_kurtosis': kurtosis(time_stats),
    })

    # Frequency Domain Features
    freq_stats = np.mean(Sxx, axis=1)
    spectral_centroid = np.sum(freqs * freq_stats) / np.sum(freq_stats)

    features.update({
        'dominant_frequency': freqs[np.argmax(freq_stats)],
        'central_frequency': spectral_centroid,  # same as spectral centroid
        'average_frequency': np.mean(freqs),
        'peak_frequency': freqs[np.argmax(freq_stats)],
        'spectral_entropy': entropy(freq_stats / np.sum(freq_stats)),
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': np.sqrt(np.sum((freqs - spectral_centroid)**2 * freq_stats) / np.sum(freq_stats)),
        'spectral_flatness': np.exp(np.mean(np.log(freq_stats + 1e-12))) / (np.mean(freq_stats) + 1e-12),
        'spectral_rolloff_85': freqs[np.where(np.cumsum(freq_stats) >= 0.85 * np.sum(freq_stats))[0][0]],
        'spectral_skewness': skew(freq_stats),
    })


    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        features[f'power_p{p}'] = np.percentile(Sxx, p)

    # Band Energy Distribution
    band_edges = np.linspace(0, len(freqs), n_bands+1, dtype=int)
    band_energies = []
    for i in range(n_bands):
        band = Sxx[band_edges[i]:band_edges[i+1], :]
        band_energy = np.sum(band)
        band_energies.append(band_energy)
        features[f'band_{i}_mean'] = np.mean(band)
        features[f'band_{i}_std'] = np.std(band)
        features[f'band_{i}_entropy'] = entropy(np.mean(band, axis=1) + 1e-12)

    # Derivative features (spectral flux)
    flux = np.diff(Sxx, axis=1)
    features['spectral_flux_mean'] = np.mean(np.abs(flux))
    features['spectral_flux_std'] = np.std(flux)

    # Signal-to-Noise Ratio (SNR)
    noise = np.mean(Sxx[:10, :], axis=0)
    signal = np.mean(Sxx[10:, :], axis=0)
    features['snr'] = np.mean(signal) / np.std(noise)

    # Log Transform for sparsity features
    Sxx_log = np.log1p(Sxx)
    features['log_power_mean'] = np.mean(Sxx_log)
    features['log_power_std'] = np.std(Sxx_log)

    # Instantaneous Frequency (using Hilbert Transform)
    analytic_signal = hilbert(np.mean(Sxx, axis=1))
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * freqs[-1]
    features['instantaneous_frequency_mean'] = np.mean(instantaneous_frequency)
    features['instantaneous_frequency_std'] = np.std(instantaneous_frequency)

    # 1. Spectral Flatness
    features['spectral_flatness'] = np.exp(np.mean(np.log(freq_stats + 1e-12))) / (np.mean(freq_stats) + 1e-12)

    # Row-wise (frequency-wise) stats
    row_mean = np.mean(Sxx, axis=1)
    row_std = np.std(Sxx, axis=1)
    features['row_mean_mean'] = np.mean(row_mean)
    features['row_std_mean'] = np.mean(row_std)

    # 2. Spectral Slope (using Linear Regression)
    slope, _, _, _, _ = linregress(freqs, freq_stats)
    features['spectral_slope'] = slope

    # Modulation Energy
    temporal_mod_energy = np.mean(np.diff(time_stats) ** 2)
    spectral_mod_energy = np.mean(np.diff(freq_stats) ** 2)
    features['temporal_mod_energy'] = temporal_mod_energy
    features['spectral_mod_energy'] = spectral_mod_energy

    # Spectrogram Sharpness
    sharpness = np.mean(np.abs(np.gradient(Sxx)))
    features['spectrogram_sharpness'] = sharpness

    return features