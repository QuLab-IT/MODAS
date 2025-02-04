import os
import xml.etree.ElementTree as ET
from   xml.dom import minidom
from   datetime import datetime

def _generate_stationxml(
    sender, module, created, network_code, network_start_date,
    network_description, network_identifier, station_code, station_start_date,
    station_description, station_latitude, station_longitude, station_elevation,
    site_name, channel_code, channel_location_code, channel_start_date,
    channel_latitude, channel_longitude, channel_elevation, channel_depth,
    channel_azimuth, channel_dip, channel_sample_rate, sensor_description,
    sensitivity_value, sensitivity_frequency, input_units, output_units, output_file
):
    # Create the root element
    root = ET.Element("FDSNStationXML")

    # Add child elements to root
    ET.SubElement(root, "Source").text = ""
    ET.SubElement(root, "Sender").text = sender
    ET.SubElement(root, "Module").text = module
    ET.SubElement(root, "Created").text = created

    # Create the Network element
    network = ET.SubElement(root, "Network", attrib={
        "code": network_code,
        "startDate": network_start_date
    })
    ET.SubElement(network, "Description").text = network_description
    identifier = ET.SubElement(network, "Identifier", attrib={"type": "DOI"})
    identifier.text = network_identifier

    # Create the Station element
    station = ET.SubElement(network, "Station", attrib={
        "code": station_code,
        "startDate": station_start_date
    })
    ET.SubElement(station, "Description").text = station_description
    ET.SubElement(station, "Latitude").text = str(station_latitude)
    ET.SubElement(station, "Longitude").text = str(station_longitude)
    ET.SubElement(station, "Elevation").text = str(station_elevation)
    site = ET.SubElement(station, "Site")
    ET.SubElement(site, "Name").text = site_name

    # Create the Channel element
    channel = ET.SubElement(station, "Channel", attrib={
        "code": channel_code,
        "locationCode": channel_location_code,
        "startDate": channel_start_date
    })
    ET.SubElement(channel, "Latitude").text = str(channel_latitude)
    ET.SubElement(channel, "Longitude").text = str(channel_longitude)
    ET.SubElement(channel, "Elevation").text = str(channel_elevation)
    ET.SubElement(channel, "Depth").text = str(channel_depth)
    ET.SubElement(channel, "Azimuth").text = str(channel_azimuth)
    ET.SubElement(channel, "Dip").text = str(channel_dip)
    ET.SubElement(channel, "SampleRate").text = str(channel_sample_rate)

    # Add Sensor element
    sensor = ET.SubElement(channel, "Sensor")
    ET.SubElement(sensor, "Description").text = sensor_description

    # Add Response element
    response = ET.SubElement(channel, "Response")
    instrument_sensitivity = ET.SubElement(response, "InstrumentSensitivity")
    ET.SubElement(instrument_sensitivity, "Value").text = str(sensitivity_value)
    ET.SubElement(instrument_sensitivity, "Frequency").text = str(sensitivity_frequency)
    input_units_elem = ET.SubElement(instrument_sensitivity, "InputUnits")
    ET.SubElement(input_units_elem, "Name").text = input_units
    output_units_elem = ET.SubElement(instrument_sensitivity, "OutputUnits")
    ET.SubElement(output_units_elem, "Name").text = output_units

    # Pretty-print the XML
    xml_string = ET.tostring(root, encoding="unicode")
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

def _date_to_day_of_year(date_str: str):
    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday
    # Format it as a three-digit code
    return f"{day_of_year:03d}"

def SDEC_file_path(dir:str, date: str):
    
    curr_path  = dir + os.sep + 'continuous_waveforms'
    year, _, _ = date.split('-')
    dyear      = _date_to_day_of_year(date)
    curr_path  += os.sep + f"{year}" + os.sep + f"{year}_{dyear}"
    
    os.makedirs(curr_path, exist_ok=True)
    return curr_path

def SDEC_file_naming(network, station, channel, location, date):

    year, _, _ = date.split('-')
    dyear      = _date_to_day_of_year(date)
    
    p_lst = [network, station, channel, location]
    p_lim = [ 2, 5, 3, 2]
    
    for i in range(4):
        size = len(p_lst[i])
        if size < p_lim[i]: p_lst[i] += (p_lim[i]-size)*'_'

    return f"{p_lst[0][:2]}{p_lst[1][:5]}{p_lst[2][:3]}{p_lst[3][:2]}_{year}{dyear}.mseed"

def SDEC_station_path(dir:str, station):
    curr_path  = dir + os.sep + 'FDSNstationXML'
    if len(station) < 2: station += (2-len(station))*'_'
    curr_path  += os.sep + f"{station[:2]}"

    os.makedirs(curr_path, exist_ok=True)
    return curr_path

def SDEC_stationxml_naming(network, station):

    p_lst = [network, station]
    p_lim = [ 2, 5]
    
    for i in range(2):
        size = len(p_lst[i])
        if size < p_lim[i]: p_lst[i] += (p_lim[i]-size)*'_'

    return f"{p_lst[0][:2]}_{p_lst[1][:5]}.xml"

_generate_stationxml(
    sender = "FAKE-DC",
    module = "Some WEB SERVICE",
    created=datetime.utcnow().isoformat() + "Z",
    network_code = "IU",
    network_start_date="1988-01-01T00:00:00Z",
    network_description="Global Seismograph Network - IRIS/USGS (GSN)",
    network_identifier="10.7914/SN/IU",
    station_code="ANMO",
    station_start_date="2002-11-19T21:07:00Z",
    station_description="(GSN) IRIS/USGS (IU) and ANSS",
    station_latitude=34.94591,
    station_longitude=-106.4572,
    station_elevation=1820.0,
    site_name="Albuquerque, New Mexico, USA",
    channel_code="BHZ",
    channel_location_code="00",
    channel_start_date="2018-07-09T20:45:00Z",
    channel_latitude=34.94591,
    channel_longitude=-106.4572,
    channel_elevation=1632.7,
    channel_depth=188,
    channel_azimuth=0,
    channel_dip=-90,
    channel_sample_rate=40,
    sensor_description="Streckeisen STS-6A VBB Seismometer",
    sensitivity_value=1.98475E9,
    sensitivity_frequency=0.02,
    input_units="m/s",
    output_units="count",
    output_file="FDSNStationXML.xml"
)