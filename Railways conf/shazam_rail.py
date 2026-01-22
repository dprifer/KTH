import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import get_window
from matplotlib.ticker import AutoMinorLocator
from dataclasses import dataclass
from collections import namedtuple, defaultdict



""" Exploration """

def hdf5_to_dict(self, obj):
    """
    Recursively load an HDF5 group or dataset into
    nested Python dictionaries and NumPy arrays.
    """
    if isinstance(obj, h5py.Dataset):
        return np.array(obj)

    elif isinstance(obj, h5py.Group):
        return {key: self.hdf5_to_dict(obj[key]) for key in obj.keys()}

    else:
        raise TypeError("Unknown HDF5 object type")


def load_data(path, scenario):

    # HARDCODED STUFF!!!!!
    #######################
    sensor_locations = ["SRS_CB_B1", "SRS_CB_cab", "SRS_B1_R", "SRS_B1_WS1_R"]
    sensor_names = ["carbody_B1", "carbody_cab", "B1_right", "WS1_right"]

    #######################


    mat_file = f"{path}/{scenario}.mat"

    sensor_data = {}

    with h5py.File(mat_file, "r") as f:
        for sensor_location, sensor_name in zip(sensor_locations, sensor_names):
            acceleration = np.array(
                f["timeInt"]["RS_result"]["SRS_G_Accelerometers"][sensor_location]["ch_003"]["values"]
            )  # ch_003: vertical acceleration

            time = np.array(f["timeInt"]["time"]["values"])
            distance = np.array(f["timeInt"]["bodyPosTrans"]["SB_CarBody"]["x"]["values"])

            # Flatten
            acceleration = np.squeeze(acceleration).T
            time = np.squeeze(time).T
            distance = np.squeeze(distance).T

            # Clip first 5 seconds: excitation fade-in length plus some
            t0 = time[0]
            mask = time >= (t0 + 5.0)

            acceleration = acceleration[mask]
            time = time[mask]
            time = time - time[0]  # shift back to 0
            distance = distance[mask]

            sensor_data[sensor_name] = (acceleration, time, distance)

    return sensor_data


def stft(sensor_data, fs=50, window_length = 4, window_type = 'hann', overlap = 0.9, fft_size = 256):
    """

    :param fs: sampling rate of simulation Hz
    :param window_length: seconds
    :param window_type:
    :param overlap: "percent" overlap of consecutive windows
    :param fft_size: must be power of 2 for efficiency
    :return:
    """

    # pre-computing
    Nx = int(window_length * fs)  # number of samples in a window
    window = get_window(window_type, Nx)  # precompute window
    stride = int(Nx * (1 - overlap))  # number of samples the sliding window slides forward
    noverlap = Nx - stride
    freqs = np.fft.rfftfreq(fft_size, 1.0 / fs)

    print(f'Number of samples in a window: {Nx}')
    print(f'Stride: {stride}')
    print(f'Number of samples overlapping per window: {noverlap}')

    # detrending
    sensor_data_detrended = {}
    for sensor_name, (acceleration, time, distance) in sensor_data.items():
        # Convert and detrend acceleration
        signal_array = np.asarray(acceleration).flatten()
        signal_array = signal.detrend(signal_array)

        # Store back the transformed acceleration, keep others unchanged
        sensor_data_detrended[sensor_name] = (signal_array, time, distance)

    stft_results = {}

    for sensor_name, (signal_array, time, distance) in sensor_data_detrended.items():
        # Compute STFT
        f, t, Zxx = signal.stft(signal_array,
                                fs=fs,
                                window=window,
                                nperseg=Nx,
                                noverlap=noverlap,
                                nfft=fft_size,
                                )

        stft_magnitude = np.abs(Zxx)

        stft_results[sensor_name] = {"f": f, "t": t, "Zxx": Zxx, "stft_magnitude": stft_magnitude}

    return stft_results, freqs, sensor_data_detrended


def peaks(stft_magnitude, time_frames, peaks_per_frame, freqs):
    """

    :param stft_magnitude: coming from the stft() function - stft_results[sensor_name]["stft_magnitude"]
    :param time_frames: coming from the stft() function - stft_results[sensor_name]["t"]
    :param peaks_per_frame: how many peaks to take from each frame, e.g. top 5
    :param freqs: coming from the stft() function, frequencies corresponding to FFT bins
    :return:
    """

    # HARDCODED STUFF!!!!!!
    #######################
    body_mode = (0.5, 2.0)  # Carbody body bounce mode
    suspension = (5.0, 9.0)  # Bogie bounce modes
    high_freq = (12.0, 25.0)  # High-frequency

    body_mode_weight: float = 1.5
    suspension_weight: float = 1.2
    high_freq_weight: float = 0.5  # giving less importance to high frequency "noise"

    #######################

    # precomputing
    bands = [
        (body_mode, body_mode_weight),
        (suspension, suspension_weight),
        (high_freq, high_freq_weight),
    ]

    band_masks = {}
    band_weights = {}

    for (f_min, f_max), weight in bands:
        band_name = f"{f_min:.1f}_{f_max:.1f}_Hz"

        # Boolean mask for frequencies in this band
        mask = (freqs >= f_min) & (freqs <= f_max)
        band_masks[band_name] = mask
        band_weights[band_name] = weight

    peaks = []  # Process each frame independently
    for frame_idx in range(stft_magnitude.shape[1]):
        frame = stft_magnitude[:, frame_idx]
        frame_time = time_frames[frame_idx]

        # Find peaks using local maxima (robust method) --> separate function?
        # For each frequency, check if it's local max vs. neighbors
        window = 3
        half_window = window // 2  # how many samples to look to the left and right
        local_max = np.zeros(len(frame), dtype=bool)  # initializing output array

        for i in range(half_window, len(frame) - half_window):
            is_max = frame[i] == np.max(frame[i - half_window:i + half_window + 1])
            if is_max and frame[i] > 0:  # Positive values only
                local_max[i] = True

            peak_freq_indices = np.where(local_max)[0]  # Extract peak values and apply weighting

        if len(peak_freq_indices) > 0:
            peak_magnitudes = frame[peak_freq_indices]  # Apply frequency band weights
            weights = np.ones(len(peak_freq_indices))

            for band_name, band_mask in band_masks.items():
                band_weight = band_weights[band_name]
                weights[band_mask[peak_freq_indices]] = band_weight
                weighted_magnitudes = peak_magnitudes * weights

            # Sort by weighted magnitude and take top N
            top_indices = np.argsort(-weighted_magnitudes)[:peaks_per_frame]

            # Convert to (freq, time, magnitude) tuples
            for idx in top_indices:
                freq_idx = peak_freq_indices[idx]
                freq_hz = freqs[freq_idx]
                magnitude = frame[freq_idx]
                peaks.append((freq_hz, frame_time, magnitude))

    # Sort by magnitude (descending) for overall ranking
    peaks.sort(key=lambda x: x[2], reverse=True)

    return peaks, band_masks, band_weights


def fingerprinting(time_window_sec, freq_quantization_hz, time_quantization_sec, peaks):
    """

    :param time_window_sec: time window forward from the anchor
    :param freq_quantization_hz: e.g. 1 Hz bins (0.5-1.5 Hz values fall into it)
    :param time_quantization_sec: e.g. 100 ms
    :param peaks: peaks extracted by the peaks() function
    :return:
    """

    # precomputing
    time_quantization_bins = int(time_window_sec / time_quantization_sec)  # Number of time bins
    freq_quantization_bins = int(
        (25.0 - 0.5) / freq_quantization_hz) + 1  # Number of frequency bins in 0.5-25 Hz range.

    print(f'Number of time bins: {time_quantization_bins}')
    print(f'Number of frequency bins: {freq_quantization_bins}')

    Peak = namedtuple("Peak", ["freq_hz", "time_sec", "magnitude"])

    peak_objs = [Peak(freq, time, mag) for freq, time, mag in peaks]

    fingerprints = {}

    # For each peak as anchor
    for i, anchor in enumerate(peak_objs):
        # Find target peaks within time window
        time_max = anchor.time_sec + time_window_sec

        for j, target in enumerate(peak_objs):
            # Skip self-pairing
            if i == j:
                continue

            # Check time constraint
            if target.time_sec <= anchor.time_sec or target.time_sec > time_max:
                continue

            # Create peak pair
            time_delta = target.time_sec - anchor.time_sec

            # Quantize frequencies
            freq_1_bin = int(anchor.freq_hz / freq_quantization_hz)
            freq_2_bin = int(target.freq_hz / freq_quantization_hz)

            # Quantize time delta
            time_bin = int(time_delta / time_quantization_sec)

            # Clamp to valid ranges
            freq_1_bin = np.clip(freq_1_bin, 0, freq_quantization_bins - 1)
            freq_2_bin = np.clip(freq_2_bin, 0, freq_quantization_bins - 1)
            time_bin = np.clip(time_bin, 0, time_quantization_bins - 1)

            hash_tuple = (freq_1_bin, freq_2_bin, time_bin)

            # Store anchor time for later matching
            if hash_tuple not in fingerprints:
                fingerprints[hash_tuple] = []
            fingerprints[hash_tuple].append((anchor.time_sec, time_delta))

    print(f"Generated {len(fingerprints)} unique fingerprints from {len(peak_objs)} peaks.")

    occurrences = [len(times) for times in fingerprints.values()]
    stats = {
        "total_hashes": sum(occurrences),
        "unique_hashes": len(fingerprints),
        "avg_occurrences": np.mean(occurrences),
        "max_occurrences": np.max(occurrences),
        "min_occurrences": np.min(occurrences),
    }

    return fingerprints, stats



""" Real stuff """
# Compute signal STFT
def compute_stft(signal_array):

    # HARDCODED STUFF
    #################
    fs = 50
    window_length = 4
    window_type = 'hann'
    overlap = 0.9
    fft_size = 256

    freqs = np.fft.rfftfreq(fft_size, 1.0 / fs)

    Nx = int(window_length * fs)  # number of samples in a window
    window = get_window(window_type, Nx)  # precompute window
    stride = int(Nx * (1 - overlap))  # number of samples the sliding window slides forward
    noverlap = Nx - stride

    #################

    signal_array = np.asarray(signal_array).flatten()

    signal_array = signal.detrend(signal_array)

    f, t, Zxx = signal.stft(signal_array,
                            fs=fs,
                            window=window,
                            nperseg=Nx,
                            noverlap=noverlap,
                            nfft=fft_size,
                            )

    # Return magnitude spectrogram and time vector
    stft_magnitude = np.abs(Zxx)

    return stft_magnitude, t, freqs


# Evaluate segment
def process_signal(signal_array):

    stft_magnitude, time_frames, freqs = compute_stft(signal_array)
    pks = peaks(stft_magnitude, time_frames, 5, freqs)

    return stft_magnitude, pks, time_frames


# Database building
class FingerprintDatabase:

    def __init__(self):

        # Main storage: {sensor -> {hash_tuple -> [(scenario, segment_idx, time), ...]}}

        self.db = {}  # Separate dbs per sensor

        # Metadata for reconstruction
        self.scenario_names = set()
        self.sensor_names = set()
        self.segments_per_scenario = defaultdict(int)

    def add_fingerprint(self, hash_tuple, scenario_name, segment_index, sensor_name, anchor_time_sec, frame_time_sec):
        """

        :param hash_tuple: Quantized fingerprint hash
        :param scenario_name: Name of simulation scenario (e.g., "S1_ERRI1_K1")
        :param segment_index: Index of segment within simulation
        :param sensor_name: Sensor location (e.g., "CB_B1", "B1_right", "B1_WS1_right")
        :param anchor_time_sec: Relative time of anchor peak within segment
        :param frame_time_sec: Absolute time in simulation
        :return:
        """

        self.scenario_names.add(scenario_name)
        self.sensor_names.add(sensor_name)

        record = (scenario_name, segment_index, sensor_name, anchor_time_sec, frame_time_sec)

        if sensor_name not in self.db:
            self.db[sensor_name] = defaultdict(list)
        self.db[sensor_name][hash_tuple].append(record)





