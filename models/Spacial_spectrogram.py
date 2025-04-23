import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def plot_spatial_spectrogram(stream, time_window, fs, spatial_sample, window_size=64, overlap=0.5):
    t0, t1 = time_window
    n_samples = stream[0].stats.npts
    total_time = n_samples / fs
    time_array = np.linspace(0, total_time, n_samples)

    # Select time window
    mask = (time_array >= t0) & (time_array <= t1)
    data = np.array([tr.data[mask] for tr in stream.traces])  # shape: (n_channels, time_window_samples)

    # Mean over time in the selected window
    snapshot = np.mean(data, axis=1)

    n_channels = len(snapshot)
    step = max(1, int(window_size * (1 - overlap)))
    spatial_freqs = fftshift(fftfreq(window_size, d=spatial_sample))

    spectro = []
    centers = []

    for start in range(0, n_channels - window_size + 1, step):
        window = snapshot[start:start + window_size]
        if len(window) < window_size:
            continue
        fft_result = fftshift(fft(window))
        power_db = 10 * np.log10(np.abs(fft_result)**2 + 1e-10)
        spectro.append(power_db)
        centers.append((start + window_size // 2) * spatial_sample)

    if len(spectro) < 2:
        print("⚠️ Not enough windows for spatial spectrogram. Try reducing `window_size` or overlap.")
        return

    spectro = np.array(spectro).T  # (n_freqs, n_windows)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(centers, spatial_freqs, spectro, shading='auto', cmap='Spectral_r')
    plt.xlabel("Fiber Position (m)")
    plt.ylabel("Spatial Frequency (1/m)")
    plt.title("Spatial Spectrogram")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()