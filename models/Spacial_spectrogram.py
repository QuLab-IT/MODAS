import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def plot_spatial_spectrogram(stream, time_window, fs, spatial_sample, window_size=64, overlap=0.5, target_time=None):
    
    t0, t1 = time_window
    n_samples = stream[0].stats.npts
    total_time = n_samples / fs
    time_array = np.linspace(0, total_time, n_samples)

    # Default to center of time window if not specified
    if target_time is None:
        target_time = (t0 + t1) / 2

    if not (t0 <= target_time <= t1):
        raise ValueError("target_time must be within the specified time_window.")

    # Find index closest to the desired time
    time_index = np.argmin(np.abs(time_array - target_time))

    # Extract one sample per trace at the chosen time
    snapshot = np.array([tr.data[time_index] for tr in stream.traces])  # shape: (n_channels,)

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
        print("Not enough windows for spatial spectrogram.")
        return

    spectro = np.array(spectro).T  # (n_freqs, n_windows)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(centers, spatial_freqs, spectro, shading='auto', cmap='Spectral_r')
    plt.xlabel("Fiber Position (m)")
    plt.ylabel("Spatial Frequency (1/m)")
    plt.title(f"Spatial Spectrogram at t = {target_time:.3f} s")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()