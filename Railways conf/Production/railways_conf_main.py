import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import library modules
from config import FingerprinterConfig
from stft_processor import STFTProcessor
from fingerprint import FingerprintGenerator, ConstellationMap
from database import FingerprintDatabase, MatchingScorer
from sim_io import SimulationLoader, SegmentationHelper
from plotting import plot_time_series, plot_spectrogram_with_peaks
import time




def load_and_explore():

    print("\n" + "=" * 70)
    print("Load and Visualize")
    print("=" * 70)

    # Initialize configuration
    config = FingerprinterConfig()
    config.print_summary()

    # Initialize loader
    loader = SimulationLoader(
        r"C:\Users\prife\OneDrive - KTH\KTH\04 Research\01 Conferences\2026\Railways\01 Simulations\main_model\output")

    # Load one scenario
    scenario_name = "S1_ERRI1_K1"
    print(f"\nLoading {scenario_name}...")

    try:
        sensor_data = loader.load_scenario(scenario_name)
        print(f"Loaded {len(sensor_data)} sensors")

        for sensor_name, (accel, time, dist) in sensor_data.items():
            print(f"  {sensor_name}: {len(accel)} samples, {time[-1]:.2f} s duration")

        return sensor_data, config, loader

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Note: Adjust the path in SimulationLoader and sensor channel names to match your .mat structure.")
        return None, config, loader


def stft_and_peaks(sensor_data, config):
    """Compute STFT, detect peaks, visualize"""

    print("\n" + "=" * 70)
    print("STFT and Peak Detection")
    print("=" * 70)

    # Initialize STFT processor
    processor = STFTProcessor(
        config.stft,
        config.freq_bands,
        config.peak_detection,
    )

    # Process first sensor
    sensor_name = list(sensor_data.keys())[0]
    acceleration, time, distance = sensor_data[sensor_name]

    print(f"\nProcessing {sensor_name}...")

    # Compute STFT
    stft_magnitude, time_frames = processor.compute_stft(acceleration, detrend=True)
    print(f"STFT shape: {stft_magnitude.shape}")
    print(f"Frequency resolution: {processor.stft_cfg.freq_resolution_hz:.4f} Hz")
    print(f"Time resolution: {config.stft.stride_samples / config.stft.sampling_rate_hz:.3f} s")

    # Extract peaks
    peaks = processor.extract_peaks_from_spectrogram(stft_magnitude, time_frames)
    print(f"Total peaks extracted: {len(peaks)}")

    # Print first few peaks
    print("\nFirst 10 peaks (freq [Hz], time [s], magnitude):")
    for i, (freq, t, mag) in enumerate(peaks[:10]):
        print(f"  {i + 1:2d}: {freq:6.2f} Hz, {t:6.3f} s, mag={mag:10.2e}")

    return processor, stft_magnitude, peaks, time_frames


def fingerprinting(config, processor, peaks):
    """Generate fingerprints from peaks."""

    print("\n" + "=" * 70)
    print("Fingerprint Generation")
    print("=" * 70)

    # Generate fingerprints
    fp_gen = FingerprintGenerator(config.peak_pairing)
    fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)

    print(f"Generated {len(fingerprints)} unique fingerprints")

    # Compute statistics
    stats = fp_gen.hash_statistics(fingerprints)
    print(f"  Total hashes: {stats['total_hashes']}")
    print(f"  Unique hashes: {stats['unique_hashes']}")
    print(f"  Avg occurrences per hash: {stats['avg_occurrences']:.2f}")
    print(f"  Max occurrences: {stats['max_occurrences']}")

    print("\nExample fingerprint hashes (first 5):")
    for i, hash_tuple in enumerate(list(fingerprints.keys())[:5]):
        times = fingerprints[hash_tuple]
        display_str = fp_gen.format_hash_for_display(hash_tuple)
        print(f"  {i + 1}: {display_str}, appears at {len(times)} anchor times")

    return fingerprints, fp_gen

def main():

    t0 = time.perf_counter()

    print("\n")
    print("#" * 70)
    print("# VIBRATION FINGERPRINTING PIPELINE - COMPLETE EXAMPLE")
    print("#" * 70)

    sensor_data, config, loader = load_and_explore()

    processor, stft_magnitude, peaks, time_frames = stft_and_peaks(sensor_data, config)

    fingerprints, fp_gen = fingerprinting(config, processor, peaks)

    acceleration, t, distance = sensor_data[list(sensor_data.keys())[0]]

    # plot_time_series(acceleration, t, title="Carbody acceleration")
    # plot_spectrogram_with_peaks(stft_magnitude, processor.freqs, time_frames, peaks, max_peaks=200)
    #
    # plt.show()


    dt = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(f"Finished in {dt:.3f} seconds!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

