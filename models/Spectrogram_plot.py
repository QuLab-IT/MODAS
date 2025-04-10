import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_spectrogram(stream, channel_index, fs, nperseg=1024, cmap='Spectral'):
    # Extract data from the selected channel
    trace = stream.traces[channel_index]
    signal = trace.data

    # Compute spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)

    # Convert power spectral density to dB
    Sxx_dB = np.log10(Sxx)
    
    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx_dB, shading='auto', cmap=cmap)
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogram - {trace.stats.channel}")
    plt.show()
