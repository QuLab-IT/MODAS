import h5py
from obspy                     import Stream, Trace, UTCDateTime

# 1. Create a Stream from a numpy array
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

# 3. Write Stream to HDF5
def write_to_h5(stream, filename):
    """
    Writes an ObsPy Stream object to an HDF5 file.
    
    Args:
        stream (Stream): ObsPy Stream object.
        filename (str): Path to the HDF5 file.
    """
    with h5py.File(filename, "w") as f:
        data_group = f.create_group("data")
        metadata_group = f.create_group("metadata")
        
        for trace in stream:
            channel_name = trace.stats.channel
            # Save data
            data_group.create_dataset(channel_name, data=trace.data)
            # Save metadata as attributes
            for key, value in trace.stats.items():
                if key in ["starttime", "endtime"]:  # Convert UTCDateTime to string
                    value = str(value)
                metadata_group.attrs[f"{channel_name}_{key}"] = value
    
# 4. Read HDF5 into Stream
def read_from_h5(filename):
    """
    Reads an HDF5 file and converts it back into an ObsPy Stream object.
    
    Args:
        filename (str): Path to the HDF5 file.
    
    Returns:
        Stream: ObsPy Stream object.
    """
    stream = Stream()
    with h5py.File(filename, "r") as f:
        data_group = f["data"]
        metadata_group = f["metadata"]
        
        for channel_name, dataset in data_group.items():
            # Create a trace
            trace = Trace(data=dataset[:])
            # Populate metadata from attributes
            for key, value in metadata_group.attrs.items():
                if key.startswith(channel_name):
                    attr_name = key[len(channel_name) + 1:]  # Strip channel name prefix
                    if attr_name == "starttime":  # Convert string back to UTCDateTime
                        value = UTCDateTime(value)
                    elif attr_name != "endtime":
                        setattr(trace.stats, attr_name, value)
            stream.append(trace)
    return stream
