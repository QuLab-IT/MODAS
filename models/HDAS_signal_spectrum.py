import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from tqdm         import tqdm

def _noise_filter_peaks(spect, peaks, limit = 0):
    noise_level = np.mean(spect[limit:]) + np.mean(spect[peaks])
    return np.array([p for p in peaks if spect[p] >= noise_level])

def _filter_peaks_byneighbour(spect, peaks, n_peaks):
    index_bin_size = len(spect) // n_peaks
    cc             = 1
    counts         = np.ones(len(peaks))

    while cc > 0:
        changes_made = False
        aux = np.ones(len(peaks), dtype=bool)

        for i in range(0, len(peaks)):
            if peaks[i] != -1:
                if i > 0 and peaks[i] - peaks[i-1] < cc * index_bin_size and spect[peaks[i]] > spect[peaks[i - 1]]:
                    counts[i]  += counts[i-1]
                    aux[i - 1] = False
                    changes_made = True
                if i < len(peaks)-1 and peaks[i + 1] - peaks[i] < cc * index_bin_size and spect[peaks[i]] > spect[peaks[i + 1]]:
                    counts[i]  += counts[i+1]
                    aux[i + 1] = False
                    changes_made = True

        peaks  = peaks[aux]
        counts = counts[aux]

        if changes_made:
            cc += 1
        else:
            cc = 0

    return peaks, counts

def _apply_optimal_filter( data, peaks, counts, freq_arr, fs, n_peaks):
    freq_bin_size = fs / (2 * n_peaks)
    filtered_data = np.zeros((len(peaks), data.size))

    for i, peak in enumerate(peaks):
        df     = max(1, freq_bin_size*counts[i])
        f_low  = max(0.09, freq_arr[peak] - df / 2)
        f_high = min(0.99*(fs/2), freq_arr[peak] + df / 2)
        
        if f_high - f_low < 1: 
            if f_high == 0.99*(fs/2): f_low = f_high - 1
            else: f_high = f_low + 1
    

        filtered_data[i,:] = BandPass_filter(data, f_low, f_high, fs)

    # Sum filtered data
    return np.sum(filtered_data, axis=0)

def _divide_into_bins(min_val, max_val, N):
    # Create N + 1 bin edges to define the N intervals
    bin_edges = np.linspace(min_val, max_val, N + 1)
    
    # Pair consecutive edges to define each window (min, max)
    bins = [(bin_edges[i], bin_edges[i+1]) for i in range(N)]
    
    return bins

def _sorted_peaks(data, accept_win = None):

    if accept_win is not None:
        peaks, _ = find_peaks(data[accept_win[0] : accept_win[1]])
        peaks    = accept_win[0] + peaks
    else: 
        peaks, _ = find_peaks(data)
    
    indices  = np.argsort(data[peaks])[:][::-1]

    return peaks[indices]

def _get_index(value, min_val, max_val, N):
    return int(round((value - min_val) / (max_val - min_val) * (N - 1)))

def select_peaks(data, n_timebins, min_val, max_val):
    bins  = _divide_into_bins(min_val, max_val, n_timebins)
    peaks = []
    for bin in bins:
        window = ( _get_index(bin[0], min_val, max_val, len(data)) , 
                   _get_index(bin[1], min_val, max_val, len(data)) )
        aux = _sorted_peaks(data, accept_win = window)
        if len(aux) > 0: peaks += [ aux[0] ]

    return peaks

def spectrum(data, axis=1, sampling_rate=50):
    # Perform FFT along the time axis (x-axis, axis=1)
    if len(data.shape) == 1: axis = 0
    fft_result = np.fft.fft(data, axis=axis)

    # Compute the amplitude spectrum (magnitude of the FFT result)
    amplitude_spectrum = np.abs(fft_result)

    # Calculate the frequencies in Hz
    n_timepoints = data.shape[axis]
    frequencies = np.fft.fftfreq(n_timepoints, d=1/sampling_rate)  # 'd' is the time step, 1/sampling_rate

    # Only plot the positive frequencies (real-world data only)
    positive_frequencies = frequencies[:n_timepoints // 2]
    if axis == 1:
        positive_amplitude = amplitude_spectrum[:, :n_timepoints // 2]
    else:
        positive_amplitude = amplitude_spectrum[:n_timepoints // 2]

    return positive_amplitude, positive_frequencies

def spectral_whitening(data, axis=1, epsilon=1e-6):
    """
    Apply spectral whitening to data along the specified axis.
    
    Parameters:
    - data: Input 2D array (e.g., strain data).
    - axis: Axis along which to apply the whitening (default: -1 for time axis).
    - epsilon: Small value to avoid division by zero.
    
    Returns:
    - whitened_data: The data after spectral whitening.
    """
    # Fourier Transform along the specified axis (time axis by default)
    data_fft = np.fft.fft(data, axis=axis)
    
    # Compute amplitude spectrum (absolute value of the complex FFT)
    amplitude = np.abs(data_fft)
    
    # Avoid division by zero by adding a small value (epsilon)
    amplitude[amplitude < epsilon] = epsilon
    
    # Whiten the spectrum: Normalize the amplitude to 1, keeping the phase
    whitened_fft = data_fft / amplitude
    
    # Inverse Fourier Transform to return to time domain
    whitened_data = np.fft.ifft(whitened_fft, axis=axis)
    
    # Take the real part of the inverse FFT (the data is real-valued)
    return np.real(whitened_data)

def BandPass_filter(Data_2D, fc_low, fc_high, frequency_sample):
    # Band-Pass Filter
    nyquist = frequency_sample * 0.5
    low     = fc_low / nyquist
    high    = fc_high / nyquist
    # Design Butterworth filter
    b, a = butter(5, [low, high], btype='band')
    # Apply the filter using filtfilt for zero phase shift
    return filtfilt(b, a, Data_2D, axis=-1)

def optimal_BandPass_filter(Data_2D, frequency_sample, n_peaks):
    spect_data, freq_array = spectrum(Data_2D)
    filtered_data          = np.zeros(Data_2D.shape)

    for i in tqdm( range(spect_data.shape[0]) ):

        spect            = spect_data[i]
        peaks            = select_peaks(spect, n_peaks, 0, frequency_sample/2)[1:]
        peaks            = _noise_filter_peaks(spect, peaks, limit = int( len(spect)*1/5 ))
        peaks, counts    = _filter_peaks_byneighbour(spect, peaks, n_peaks)
        filtered_data[i] = _apply_optimal_filter(Data_2D[i], peaks, counts, freq_array, frequency_sample, n_peaks)

    return filtered_data
