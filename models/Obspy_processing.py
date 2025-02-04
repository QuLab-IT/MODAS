import numpy             as np
import matplotlib.pyplot as plt

from obspy                     import Stream, Trace, UTCDateTime


def create_stream(data_array, sampling_rate, starttime, n_channels=1, chan_lst=None, station='DAS1', network='XX', location='00'):
    
    if chan_lst == None:
        chan_lst = [f"CH{i:04}" for i in range(n_channels)]

    stream = Stream()
    for i in range(n_channels):
        trace = Trace(data=data_array[i])
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime     = UTCDateTime(starttime)
        trace.stats.channel       = chan_lst[i]  # Default channel names
        trace.stats.station       = station  # Default station name
        trace.stats.network       = network  # Default network code
        trace.stats.location      = location  # Default location
        stream.append(trace)
    return stream

def running_absolute_mean(data, window_samples):
    """
    Computes the running absolute mean of a signal.
    
    Parameters:
    -----------
    data : np.array
        1D NumPy array containing the seismic trace data.
    window_samples : int
        Window size in samples for running mean calculation.
    
    Returns:
    --------
    np.array
        Running absolute mean of the signal.
    """
    abs_data = np.abs(data)
    ram_values = np.convolve(abs_data, np.ones(window_samples) / window_samples, mode="same")
    return ram_values

def ram_normalization(st: Stream, window_sec: float):
    """
    Applies Running Absolute Mean (RAM) normalization to an ObsPy Stream.
    
    Parameters:
    -----------
    st : Stream
        ObsPy Stream containing multiple traces.
    window_sec : float
        Window size in seconds for the RAM calculation.
    
    Returns:
    --------
    Stream
        Stream with RAM-normalized traces.
    """
    st_normalized = st.copy()
    
    for tr in st_normalized:
        sampling_rate = tr.stats.sampling_rate
        window_samples = max(1, int(window_sec * sampling_rate))  # Convert time to samples
        
        # Compute RAM and normalize the trace
        ram_values = running_absolute_mean(tr.data, window_samples)
        tr.data = tr.data / (ram_values + 1e-6)  # Avoid division by zero
    
    return st_normalized

def show_sort_plot(stream: Stream, color='k'):
    # Create a figure with multiple subplots (one per trace)
    fig, axes = plt.subplots(len(stream), 1, figsize=(10, 6), sharex=True, sharey=True)

    # Ensure axes is a list if there is only one trace
    if len(stream) == 1:
        axes = [axes]

    # Loop through each trace and plot separately
    for i, tr in enumerate(stream):
        time = tr.times() / (60*60)  # Get time array for x-axis in [h]
        axes[i].plot(time, tr.data, color, label=f"{tr.stats.station}.{tr.stats.channel}")
        
        # Formatting
        axes[i].legend(loc="upper left")
        axes[i].grid()
        axes[i].set_ylabel("Amplitude")

    # Set common labels
    axes[-1].set_xlabel("Time [h]")
    # fig.suptitle("HDAS Traces")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def show_fft(stream: Stream, f_lim=0.2, color='k'):
    """
    Plots the FFT of each trace in an ObsPy Stream, limiting the frequency range to [0, f_lim] Hz.

    Parameters:
    -----------
    stream : Stream
        ObsPy Stream containing multiple traces.
    f_lim : float
        Maximum frequency to display in the plot (Hz).
    color : str
        Line color for the plots (default: 'k' for black).
    """
    # Set up the figure and subplots
    fig, axes = plt.subplots(len(stream), 1, figsize=(10, 6), sharex=True)

    # If only one channel exists, convert axes to a list for consistency
    if len(stream) == 1:
        axes = [axes]

    # Loop through each trace (channel) in the Stream
    for i, tr in enumerate(stream):
        # Extract raw data and sampling rate
        data = tr.data
        npts = tr.stats.npts  # Number of data points
        dt = tr.stats.delta  # Sampling interval (s)
        fs = tr.stats.sampling_rate  # Sampling frequency (Hz)

        # Compute FFT
        fft_values = np.fft.fft(data)
        freqs = np.fft.fftfreq(npts, d=dt)  # Frequency axis

        # Keep only positive frequencies
        positive_freqs = freqs[:npts // 2]
        positive_fft = np.abs(fft_values[:npts // 2])

        # Apply frequency limit
        mask = (positive_freqs >= 0) & (positive_freqs <= f_lim)
        filtered_freqs = positive_freqs[mask]
        filtered_fft = positive_fft[mask]

        # Plot FFT for this channel
        axes[i].plot(filtered_freqs, filtered_fft, color, label=f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}")
        axes[i].set_ylabel("Magnitude")
        axes[i].legend(loc="upper left")
        axes[i].grid()

    # Set common x-axis label
    axes[-1].set_xlabel("Frequency (Hz)")

    # Adjust layout
    plt.tight_layout()
    plt.show()
