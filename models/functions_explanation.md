# Imports done in main file 

- import numpy as np - for numerical operations
- import os - for retrieving data from drive
- import time - to measure how long each step takes
- import matplotlib.pyplot as plt - for plotting

- from sklearn.datasets                   import fetch_openml
- from sklearn.model_selection            import train_test_split
- from sklearn.metrics                    import classification_report, accuracy_score
- from obspy                              import Stream
- from models.HDAS_file_convert           import sampling_file_name, HDAS_meas_settings, read_bin_file
- from models.Obspy_processing            import create_stream, ram_normalization, show_sort_plot, show_fft
- from models.ASDF_file_convert           import write_to_h5
- from models.User_print                  import print_header, print_small_header, print_update
- from models.Spectrogram_plot            import plot_spectrogram
- from models.Spacial_spectrogram         import plot_spatial_spectrogram, plot_spatial_spectrogram_interval, compute_fk_spectrum
- from models.Event_analyzer              import analyze_event_dynamics 
- from models.Feature_extraction_ML       import extract_csd_vector #not being used anymore
- from models.Logistic_Regression         import run_logistic_regression, predict_logistic_regression
- from models.PCA                         import fit_pca, transform_pca
- from datetime                           import datetime - also for time tracking

#######################################################################################################

# models.HDAS_file_convert

- _index_find – Converts spatial and temporal coordinates to index ranges for array slicing.
- sampling_file_name – Returns a list of full file paths in the specified folder.
- Memory_check – Ensures enough system memory is available to process the data.
- read_bin_file – Reads a binary file with an optional header and returns data slices.
- HDAS_meas_settings – Extracts measurement and acquisition settings from a sample HDAS file header.
- HDAS_2DMap – Constructs a 2D spatial-temporal data array from multiple HDAS binary files.

#######################################################################################################

# models.Obspy_processing

- create_stream – Converts a NumPy array into an ObsPy Stream with metadata for each channel.
- running_absolute_mean – Computes a moving average of the absolute values of a 1D signal.
- ram_normalization – Normalizes each trace in an ObsPy Stream using a running absolute mean.
- show_sort_plot – Plots each trace in a stream over time in a vertically stacked layout.
- show_fft – Computes and plots the FFT of each trace in a stream, limited to low frequencies.

#######################################################################################################

# models.ASDF_file_convert

- create_stream – Builds an ObsPy Stream from a NumPy array with metadata for each trace.
- write_to_h5 – Saves all trace data and metadata from an ObsPy Stream to an HDF5 file.
- read_from_h5 – Loads trace data and metadata from an HDF5 file into an ObsPy Stream.

#######################################################################################################

# models.User_print

- print_header – Prints a styled header with an optional dictionary of arguments.
- print_arguments – Prints a dictionary of arguments with aligned keys and values.
- print_small_header – Prints a compact, styled header without arguments.
- print_separator – Prints a full-width line separator.
- print_update – Prints a single update line.

#######################################################################################################

# models.Spectrogram_plot

- plot_spectrogram — Computes and plots a time-frequency spectrogram of a selected channel from a stream, optionally limited by time and frequency ranges, with time axis formatted as datetime if provided.

#######################################################################################################

# models.Spacial_spectrogram

- plot_spatial_spectrogram — Computes and plots a spatial spectrogram snapshot at a target time by FFT over spatial channels within a window along the fiber.
- plot_spatial_spectrogram_interval — Computes and plots a time-evolving spatial spectrogram over a specified time interval by averaging FFT power spectra along spatial windows for each time sample.
- compute_fk_spectrum — Computes and plots the 2D frequency–wavenumber (f–k) power spectrum from a time window of multi-channel data using a 2D FFT.

#######################################################################################################

# models.Event_analyzer

- analyze_event_dynamics - Analyzes a chosen channel’s frequency content over time by segmenting the signal, plotting amplitude spectra, tracking dominant frequencies, and optionally fitting their PDF.

#######################################################################################################

# models.Feature_extraction_ML

- extract_spectrogram_features - Extracts diverse statistical and spectral features from a spectrogram matrix to characterize signal properties in time and frequency domains.

- extract_csd_vector - Computes and flattens cross-spectral density matrices across channel pairs and selected frequencies into a single feature vector for multichannel signals.

#######################################################################################################

# models.Logistic_Regression

- run_logistic_regression - Trains and evaluates a logistic regression model on given features and labels, printing a classification report.
- predict_logistic_regression - Uses a trained logistic regression model to predict probabilities and binary labels from new features.

#######################################################################################################

# models.PCA

- fit_pca - Fits PCA on standardized input features, automatically choosing or using a specified number of components to reduce dimensionality while preserving variance.
- transform_pca - Applies the previously fitted scaler and PCA transformation to new input features.
- get_explained_variance - Returns the percentage of variance explained by each principal component from the fitted PCA.
- plot_explained_variance - Plots individual and cumulative explained variance ratios of the fitted PCA components.
- _prepare_matrix - Converts various input formats (NumPy array, DataFrame, dict) into a 2D NumPy array suitable for PCA.