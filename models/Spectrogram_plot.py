import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def plot_spectrogram(stream, channel_index, fs, nperseg=256, cmap='Spectral_r',
                     start_time=None, time_window=None, freq_limit=20):
    """
    time_window: tuple of (start_time_offset, end_time_offset) in seconds
    freq_limit: max frequency to display in Hz
    """
    trace = stream.traces[channel_index]
    signal = trace.data

    # Compute spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')

    # Limit frequency axis
    freq_mask = f <= freq_limit
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # Convert power to dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Avoid log(0)

    if start_time is not None:
        time_axis = [start_time + timedelta(seconds=float(seconds)) for seconds in t]
    else:
        time_axis = t  # still in seconds

    # Apply time window if specified
    if time_window:
        t0, t1 = time_window  # in seconds
        if start_time is not None:
            mask = [(start_time + timedelta(seconds=float(sec)) >= start_time + timedelta(seconds=t0)) and 
                    (start_time + timedelta(seconds=float(sec)) <= start_time + timedelta(seconds=t1)) 
                    for sec in t]
        else:
            mask = (t >= t0) & (t <= t1)
        
        time_axis = np.array(time_axis)[mask]
        Sxx_dB = Sxx_dB[:, mask]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_axis, f, Sxx_dB, shading='auto', cmap=cmap)
    plt.colorbar(label="Power (dB)")

    if start_time is not None:
        plt.xlabel("Time")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()
        date_str = start_time.strftime('%Y-%m-%d')
        plt.title(f"Spectrogram - {trace.stats.channel} - {date_str}")
    else:
        plt.xlabel("Time (s)")
        plt.title(f"Spectrogram - {trace.stats.channel}")

    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()
