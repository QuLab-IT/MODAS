import numpy as np
from scipy.stats import entropy, kurtosis, skew, linregress
from scipy.signal import find_peaks, stft, hilbert
from sklearn.preprocessing import normalize
from scipy.fft import fft
from scipy.signal import csd, welch

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

def extract_csd_features(multichannel_data, fs, nperseg=256, noverlap=128, max_freq_bins=10):
    n_channels, n_samples = multichannel_data.shape
    f, _ = welch(multichannel_data[0], fs=fs, nperseg=nperseg)
    
    # Limit number of frequency bins
    freq_indices = np.linspace(0, len(f) - 1, max_freq_bins, dtype=int)

    csd_matrix = np.zeros((n_channels, n_channels, len(f)), dtype=np.complex128)

    for i in range(n_channels):
        for j in range(i, n_channels):
            _, Pxy = csd(multichannel_data[i], multichannel_data[j], fs=fs,
                         nperseg=nperseg, noverlap=noverlap)
            csd_matrix[i, j, :] = Pxy
            if i != j:
                csd_matrix[j, i, :] = np.conj(Pxy)

    features = {}
    for k in freq_indices:
        C = csd_matrix[:, :, k]

        # Total power
        features[f"trace_f{k}"] = np.trace(C).real

        # Off-diagonal energy
        off_diag = C - np.diag(np.diag(C))
        features[f"offdiag_frobenius_f{k}"] = np.linalg.norm(off_diag, ord='fro').real

        # Spectral flatness of eigenvalues
        eigvals = np.abs(np.linalg.eigvalsh(C))
        eigvals += 1e-10  # Avoid log(0)
        geo_mean = np.exp(np.mean(np.log(eigvals)))
        arith_mean = np.mean(eigvals)
        features[f"spectral_flatness_f{k}"] = geo_mean / arith_mean

    return features

def compute_csd_matrix(data_matrix, fs, nperseg=256):
    n_channels = data_matrix.shape[0]
    csdm = np.zeros((n_channels, n_channels), dtype=np.complex64)

    # Compute the cross-spectral density for each pair of channels
    for i in range(n_channels):
        for j in range(i, n_channels):
            _, Pxy = csd(data_matrix[i], data_matrix[j], fs=fs, nperseg=nperseg)
            csdm[i, j] = np.mean(Pxy)
            csdm[j, i] = np.conj(csdm[i, j])  # Ensuring the matrix is Hermitian

    return csdm
