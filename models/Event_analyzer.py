import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import norm

def analyze_event_dynamics(stream, fs, time_window, n_segments=10, channel_index=0, fit_pdf=True):
    tr = stream[channel_index]
    t0, t1 = time_window
    samples = tr.data
    n_total = len(samples)
    time = np.linspace(0, n_total / fs, n_total)

    # Slice signal in time
    segment_times = np.linspace(t0, t1, n_segments + 1)
    segment_samples = [np.where((time >= segment_times[i]) & (time < segment_times[i + 1]))[0]
                       for i in range(n_segments)]

    freqs = fftfreq(n_total, 1/fs)
    freqs_pos = freqs[freqs >= 0]

    plt.figure(figsize=(12, 6))

    # a) Amplitude vs Frequency (with time as a parameter)
    for i, indices in enumerate(segment_samples):
        segment_data = samples[indices]
        fft_vals = np.abs(fft(segment_data, n=n_total))[:len(freqs_pos)]
        plt.plot(freqs_pos, fft_vals, label=f"t={segment_times[i]:.1f}-{segment_times[i+1]:.1f}s")

    plt.title("Amplitude vs Frequency (at different time slices)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # b) Frequency distribution over time
    dominant_freqs = []
    for indices in segment_samples:
        segment_data = samples[indices]
        fft_vals = np.abs(fft(segment_data, n=n_total))[:len(freqs_pos)]
        dominant_freqs.append(freqs_pos[np.argmax(fft_vals)])

    times_centered = [(segment_times[i] + segment_times[i+1]) / 2 for i in range(n_segments)]

    plt.figure(figsize=(10, 4))
    plt.plot(times_centered, dominant_freqs, 'o-')
    plt.xlabel("Time (s)")
    plt.ylabel("Dominant Frequency (Hz)")
    plt.title("Dominant Frequency Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Fit PDF if needed
    if fit_pdf:
        from scipy.stats import gaussian_kde

        data = np.array(dominant_freqs)
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 300)
        pdf_vals = kde(x_range)

        plt.figure(figsize=(8, 4))
        plt.plot(x_range, pdf_vals)
        plt.title("PDF Fit of Dominant Frequencies")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()