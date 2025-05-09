import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, fft2

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

def plot_spatial_spectrogram_interval(stream, time_window, fs, spatial_sample, window_size=64, overlap=0.5, target_time=None, interval=None):
    t0, t1 = time_window
    n_samples = stream[0].stats.npts
    total_time = n_samples / fs
    time_array = np.linspace(0, total_time, n_samples)

    # Default to center of time window if not specified
    if target_time is None:
        target_time = (t0 + t1) / 2

    if not (t0 <= target_time <= t1):
        raise ValueError("target_time must be within the specified time_window.")

    # Determine time range for the interval
    half_interval = interval / 2
    t_start = max(t0, target_time - half_interval)
    t_end = min(t1, target_time + half_interval)

    start_idx = int(t_start * fs)
    end_idx = int(t_end * fs)

    # Prepare spatial snapshots for each time sample in the interval
    snapshots = []
    time_axis = time_array[start_idx:end_idx]

    for time_idx in range(start_idx, end_idx):
        snapshot = np.array([tr.data[time_idx] for tr in stream.traces])
        snapshots.append(snapshot)
    
    snapshots = np.array(snapshots).T  # shape: (n_channels, n_time_samples)

    # Compute spatial spectrogram
    n_channels = snapshots.shape[0]
    step = max(1, int(window_size * (1 - overlap)))
    spatial_freqs = fftshift(fftfreq(window_size, d=spatial_sample))
    times = []

    spectro = []
    for t in range(0, snapshots.shape[1]):
        snapshot = snapshots[:, t]
        slice_spectro = []
        for start in range(0, n_channels - window_size + 1, step):
            window = snapshot[start:start + window_size]
            if len(window) < window_size:
                continue
            fft_result = fftshift(fft(window))
            power_db = 10 * np.log10(np.abs(fft_result)**2 + 1e-10)
            slice_spectro.append(power_db)
        if len(slice_spectro) > 0:
            spectro.append(np.mean(slice_spectro, axis=0))
            times.append(time_axis[t])
    
    if len(spectro) < 2:
        print("Not enough data for spatial spectrogram.")
        return

    spectro = np.array(spectro).T  # shape: (n_spatial_freqs, n_time_steps)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, spatial_freqs, spectro, shading='auto', cmap='Spectral_r')
    plt.xlabel("Time (s)")
    plt.ylabel("Spatial Frequency (1/m)")
    plt.title(f"Spatial Spectrogram Around t = {target_time:.3f} s")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()

def compute_fk_spectrum(stream, fs, spatial_sample, time_window, nfft_time=256, nfft_space=None):

    t0, t1 = time_window
    start_idx = int(t0 * fs)
    end_idx = int(t1 * fs)

    # Build 2D array: shape (n_channels, n_samples)
    data_2d = np.array([tr.data[start_idx:end_idx] for tr in stream.traces])
    n_channels, n_samples = data_2d.shape

    if nfft_space is None:
        nfft_space = n_channels

    # 2D FFT
    fk_spectrum = fftshift(np.abs(fft2(data_2d, s=(nfft_space, nfft_time)))**2)

    # Frequency and Wavenumber axes
    freqs = fftshift(fftfreq(nfft_time, d=1/fs))
    wavenumbers = fftshift(fftfreq(nfft_space, d=spatial_sample))
    wavenumbers_rad = 2 * np.pi * wavenumbers  # radians/m

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(wavenumbers_rad, freqs, 10 * np.log10(fk_spectrum.T + 1e-10), shading='nearest', cmap='viridis')
    plt.xlabel("Wavenumber (rad/m)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"FK Spectrum from t = {t0:.2f} to {t1:.2f} s")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()
