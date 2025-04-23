import numpy as np
from scipy.stats import entropy, kurtosis, skew

def extract_spectrogram_features(Sxx, freqs, times):
    features = {}

    # Basic stats
    features['mean_power'] = np.mean(Sxx)
    features['std_power'] = np.std(Sxx)
    features['max_power'] = np.max(Sxx)
    features['min_power'] = np.min(Sxx)

    # Temporal features
    features['temporal_variance'] = np.var(Sxx, axis=1).mean()
    features['spectral_variance'] = np.var(Sxx, axis=0).mean()

    # Frequency domain summary
    mean_spectrum = np.mean(Sxx, axis=1)
    features['spectral_centroid'] = np.sum(freqs * mean_spectrum) / np.sum(mean_spectrum)
    features['spectral_entropy'] = entropy(mean_spectrum / np.sum(mean_spectrum))
    features['spectral_kurtosis'] = kurtosis(mean_spectrum)
    features['spectral_skewness'] = skew(mean_spectrum)

    # Temporal domain summary
    mean_temporal = np.mean(Sxx, axis=0)
    features['temporal_entropy'] = entropy(mean_temporal / np.sum(mean_temporal))

    return features
