import numpy as np
import psutil
import os

from tqdm import tqdm

TEMP_FOLDER = os.getcwd() + os.sep + "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

def _index_find(posStart, posStop, timeStart, timeStop, offset, spatial_sample, frequency_sample):
    # Calculate index for positions directly
    pos_index_start = int((posStart - offset) / spatial_sample)
    pos_index_stop = int((posStop - offset) / spatial_sample)
    
    # Calculate index for times directly
    time_index_start = int(timeStart * frequency_sample)
    time_index_stop = int(timeStop * frequency_sample)

    return (pos_index_start, pos_index_stop), (time_index_start, time_index_stop)

def sampling_file_name(folder_path_name: str):
    """Get a list of file paths in the given directory."""
    return [ os.path.join(folder_path_name, file) for file in os.listdir(folder_path_name)]

def Memory_check(required_mem):
    req_mem = 8 * required_mem # Estimate memory usage
    avail_mem = psutil.virtual_memory().available
    if avail_mem / req_mem < 1:
        raise MemoryError("Not enough memory to process the data ({} MB < {} MB)".format(int(avail_mem*1e-6/8), int(req_mem*1e-6/8)) )
    
    return req_mem/avail_mem

def read_bin_file(file_path: str, data_start=None, data_stop=None, dtype=np.float64, skip_header=False):
    # Define size of each double (np.float64) in bytes
    dtype_size = np.dtype(dtype).itemsize
    header_size = 200 * dtype_size  # Header is 200 doubles, each 8 bytes
    
    # Memory map the file
    with open(file_path, 'rb') as fileID:
        # If skipping the header, map the file starting from after the header
        if skip_header:
            # Create a memory map starting after the header
            FileData = np.memmap(file_path, dtype=dtype, mode='r', offset=header_size)
            FileHeader = None
        else:
            # Read the header (first 200 doubles)
            FileHeader = np.fromfile(fileID, dtype=dtype, count=200)
            # Map the rest of the file after the header
            FileData = np.memmap(file_path, dtype=dtype, mode='r', offset=header_size)
    
    # Return the header (if present), the required slice of data, and the number of samples
    return FileHeader, FileData[data_start: data_stop], FileData.size

def HDAS_meas_settings(sample_file_path: str):
    args = {}
    header, _, size = read_bin_file(sample_file_path)

    args['Spatial Sampling'] = header[1]
    args['Lenght Monitored'] = header[3]
    args['N Time Samples']   = size

    if header[101] == 0: #File type: HDAS_2DRawData_Strain
        args['File Type']        = '2D_Strain'
        args['Pos Offset']       = header[11]
        args['Sample Frequency'] = header[6]/header[15]/header[98]
        args['Ref Start']        = header[17] + 1
        args['Ref Stop']         = header[19] + 1
        args['Multi Point Pos']  = header[41:50]*args['Spatial Sampling'] + args['Pos Offset']
        args['Ref Update Win']   = header[23]*header[6]/header[15]/1000
        args['N Processed']      = header[14]-header[12]

    if header[101] == 1: #File type: HDAS_2DRawData_Temp
        args['File Type']        = '2D_Temp'
        args['Pos Offset']       = header[28]
        args['Sample Frequency'] = header[6]/header[32]/header[99]
        args['Ref Start']        = header[34] + 1
        args['Ref Stop']         = header[36] + 1
        args['Multi Point Pos']  = header[51:60]*args['Spatial Sampling'] + args['Pos Offset']
        args['Ref Update Win']   = header[40]*header[6]/header[32]/1000
        args['N Processed']      = header[31]-header[29]

    return args

def HDAS_2DMap(sampling_folder_path: str,  fiber_lenght: tuple[int, int], time: tuple[int, int]):

    sampling_files   = sampling_file_name(sampling_folder_path) # Select the data files from the respective directory
    
    meas_settings    = HDAS_meas_settings(sampling_files[0])    # Determine the measure settings
    N_time_samples   = meas_settings['N Time Samples']          # Determine the number of time samples
    N_pos_samples    = meas_settings['N Processed']             # Determine the number of position samples
    frequency_sample = meas_settings['Sample Frequency']        # Determine the frequency of aquisition
    spatial_sample   = meas_settings['Spatial Sampling']        # Determine the lenght separation
    offset           = meas_settings['Pos Offset']              # Determine the lenght offset

    FiberStart, FiberStop = fiber_lenght                        # Select Start/Stop Spatial Point
    TimeStart, TimeStop   = time                                # Select Start/Stop Time Sample

    # Get indices for positions and times
    posIndex, timeIndex   = _index_find(FiberStart, FiberStop, TimeStart, TimeStop, offset, spatial_sample, frequency_sample)
    if posIndex[1] >= N_pos_samples:
        raise Warning("Warning: fiber lenght exceeds maximum")
    if timeIndex[1] >= N_time_samples:
        raise Warning("Warning: time stop exceeds maximum")

    # Precompute the fiber positions and time arrays
    position_samples = int( np.ceil(1 + (FiberStop - FiberStart) / spatial_sample) )
    time_samples     = int( np.ceil(1 + (TimeStop - TimeStart) * frequency_sample) )
    
    # Check the memory available to process the data
    Memory_check(required_mem = position_samples * time_samples)

    # Initialize raw data array
    # raw_data_2D = np.zeros((position_samples, time_samples))
    file_path = TEMP_FOLDER + os.sep + "temp_datamap.dat"
    if os.path.exists(file_path): os.remove(file_path)
    raw_data_2D    = np.memmap(file_path, dtype=np.float64, mode='w+', shape=(position_samples, time_samples))

    # Process each file and fill the 2D array
    i = 0
    for pos_i in tqdm(range(posIndex[0], posIndex[1]+1)):
        # Read the relevant data slice from each file
        _, FileData, _ = read_bin_file(sampling_files[pos_i], 
                                    data_start    = timeIndex[0], 
                                    data_stop     = timeIndex[1] + 1, 
                                    skip_header   = True)
        
        # Check and replace NaN values
        FileData = np.nan_to_num(FileData, nan=0)
        
        # Store data values
        raw_data_2D[i] = FileData
        i += 1

    raw_data_2D.flush()
    
    return raw_data_2D, meas_settings
