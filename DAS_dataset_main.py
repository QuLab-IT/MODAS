import numpy as np
import os

from obspy                    import Stream
from models.HDAS_file_convert import sampling_file_name, HDAS_meas_settings, read_bin_file
from models.Obspy_processing  import create_stream, ram_normalization, show_sort_plot, show_fft
from models.ASDF_file_convert import write_to_h5
from models.User_print        import print_header, print_small_header, print_update

#############################################################################################

Name          = 'Madeira'                   # Project Name
raw_data_file = './/Madeira//Data'          # Raw data files path

stop          = 1000                        # Total number of channels to be loaded
select        = [37, 173, 460, 580, 756]    # Selection of channels to monitor

window_norm   = 10.0                        # running average window for time-domain normalization [s]
band_freq     = [0.01, 0.2]                 # Band pass filter [f_cut_min, f_cut_high] [Hz]

# Metadata
start_time = '2025-01-01'                   # Starting time
station    = 'MAD'                          # Station Name
network    = 'MODAS'                        # Network Name
location   = 'MAD'                          # Location Name

#############################################################################################
# Step 0. Find Data
save_data_folder = os.path.join(os.getcwd(), 'Code', 'data', 'results')
raw_data_folder  = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'MODAS', Name)
sampling_files   = sampling_file_name(raw_data_folder)

meas_settings    = HDAS_meas_settings( sampling_files[0] )
n_time_samples   = meas_settings['N Time Samples']          # Determine the number of time samples
n_channels       = int(meas_settings['N Processed'])        # Determine the number of position samples
frequency_sample = meas_settings['Sample Frequency']        # Determine the frequency of aquisition
spatial_sample   = meas_settings['Spatial Sampling']        # Determine the lenght separation
offset           = meas_settings['Pos Offset']              # Determine the lenght offset

aquisition_time  = n_time_samples/frequency_sample          # Determine the total time of aquisition in [s]

###########################
# Step 0.1 Print Header
args = { 'Project name'   : Name,
         'Fiber lenght'   : f'{n_channels*spatial_sample + offset} m',
         'N channels'     : n_channels,
         'Channel spacing': f'{spatial_sample} m',
         'Spatial offset' : f'{offset} m',
         'N time samples' : n_time_samples,
         'Aquisition time': f'{aquisition_time/(60*60)} h',
         'Sampling rate'  : f'{frequency_sample} Hz' }
print_header('DAS Raw Data converter', args)


#############################################################################################
# Step 1. Convert data into Obspy.stream
print_small_header('Converting the Data into a stream')

raw_data_2D = np.empty((stop, n_time_samples))
# Process each file and fill the 2D array
for chan_count in range(stop):
    # Read the relevant data slice from each file
    _, FileData, _ = read_bin_file(sampling_files[chan_count],  
                                skip_header   = True)
    
    # Check and replace NaN values
    FileData = np.nan_to_num(FileData, nan=0)
    
    # Store data values
    raw_data_2D[chan_count] = FileData

channel_list = [f"CH{50+i*10}m" for i in range(stop)]
stream       = create_stream(raw_data_2D, frequency_sample, start_time, stop, channel_list, station, network, location)

# Delete previous data formats to free memory
del FileData, raw_data_2D

#############################################################################################
# Step 2. Single station analysis
print_small_header('Single station analysis')

# Select channels to monitor
chn_select = [channel_list[i] for i in select]
st_monitor = Stream()
for chan in chn_select:
    st_monitor += stream.select(channel=chan)

show_sort_plot(st_monitor)  # Uncomment to plot Raw data

print_update('Detrending data')
stream.detrend()
st_monitor.detrend()

show_sort_plot(st_monitor)  # Uncomment to plot Detrended data

print_update('Filtering data')
stream.filter("bandpass", freqmin=band_freq[0], freqmax=band_freq[1], corners=4, zerophase=True)
st_monitor.filter("bandpass", freqmin=band_freq[0], freqmax=band_freq[1], corners=4, zerophase=True)

show_fft(st_monitor)        # Uncomment to plot filtered data spectrum
show_sort_plot(st_monitor)  # Uncomment to plot filtered data

print_update('Normalize data')
stream     = ram_normalization(stream, window_norm)
st_monitor = ram_normalization(st_monitor, window_norm)

show_sort_plot(st_monitor)  # Uncomment to print normalized data

print_update('saving ASDF data')
write_to_h5(stream, 'DAS_SSprocess.h5')