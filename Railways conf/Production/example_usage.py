"""
example_usage.py

Complete example demonstrating the fingerprinting pipeline.
Run this script to:
1. Load simulations
2. Extract STFT + peaks
3. Generate fingerprints
4. Build database
5. Perform matching
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import library modules
from config import FingerprinterConfig
from stft_processor import STFTProcessor
from fingerprint import FingerprintGenerator, ConstellationMap
from database import FingerprintDatabase, MatchingScorer
from sim_io import SimulationLoader, SegmentationHelper


def example_1_load_and_explore():
    """Example 1: Load simulation, visualize spectrogram and peaks."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Load and Explore")
    print("="*70)
    
    # Initialize configuration
    config = FingerprinterConfig()
    config.print_summary()
    
    # Initialize loader
    loader = SimulationLoader(r"C:\Users\prife\OneDrive - KTH\KTH\04 Research\01 Conferences\2026\Railways\01 Simulations\main_model\output")
    
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


def example_2_stft_and_peaks(sensor_data, config):
    """Example 2: Compute STFT, detect peaks, visualize."""
    
    if sensor_data is None:
        print("Skipping Example 2 (no data)")
        return None
    
    print("\n" + "="*70)
    print("EXAMPLE 2: STFT and Peak Detection")
    print("="*70)
    
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
        print(f"  {i+1:2d}: {freq:6.2f} Hz, {t:6.3f} s, mag={mag:10.2e}")
    
    return processor, stft_magnitude, peaks, time_frames


def example_3_fingerprinting(processor, peaks):
    """Example 3: Generate fingerprints from peaks."""
    
    if processor is None or not peaks:
        print("Skipping Example 3 (no peaks)")
        return None
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Fingerprint Generation")
    print("="*70)
    
    # Generate fingerprints
    fp_gen = FingerprintGenerator(processor.pairing_cfg)
    fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)
    
    print(f"Generated {len(fingerprints)} unique fingerprints")
    
    # Compute statistics
    stats = fp_gen.hash_statistics(fingerprints)
    print(f"  Total hashes: {stats['total_hashes']}")
    print(f"  Unique hashes: {stats['unique_hashes']}")
    print(f"  Avg occurrences per hash: {stats['avg_occurrences']:.2f}")
    print(f"  Max occurrences: {stats['max_occurrences']}")
    
    # Show examples
    print("\nExample fingerprint hashes (first 5):")
    for i, hash_tuple in enumerate(list(fingerprints.keys())[:5]):
        times = fingerprints[hash_tuple]
        display_str = fp_gen.format_hash_for_display(hash_tuple)
        print(f"  {i+1}: {display_str}, appears at {len(times)} anchor times")
    
    return fingerprints, fp_gen


def example_4_database(processor, sensor_data, config):
    """Example 4: Build database from multiple scenarios."""
    
    if sensor_data is None:
        print("Skipping Example 4 (no data)")
        return None
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Database Building")
    print("="*70)
    
    # Initialize database
    db = FingerprintDatabase(combine_sensors=config.database.combine_sensors)
    
    # For demonstration, process just one scenario
    scenario_name = list(sensor_data.keys())[0]
    accel, time, dist = sensor_data[scenario_name]
    
    # Extract segments
    segments = SegmentationHelper.extract_segments(
        accel, time,
        config.stft.window_length_sec,
        config.stft.overlap,
        config.segmentation.subsample_factor,
    )
    
    print(f"\nProcessing {scenario_name} with {len(segments)} segments...")
    
    # Process segments
    processor_instance = STFTProcessor(
        config.stft,
        config.freq_bands,
        config.peak_detection,
    )
    fp_gen = FingerprintGenerator(config.peak_pairing)
    
    for seg_idx, (seg_signal, seg_time, _) in enumerate(segments[:10]):  # First 10 segments
        stft_mag, peaks, _ = processor_instance.process_signal(seg_signal)
        fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)
        
        # Add to database
        db.add_fingerprints_batch(
            fingerprints,
            scenario_name=scenario_name,
            segment_index=seg_idx,
            sensor_name="carbody",
            frame_time_sec=seg_time,
        )
    
    # Print statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    print(f"  Scenarios: {stats['num_scenarios']}")
    print(f"  Sensors: {stats['num_sensors']}")
    print(f"  Unique hashes: {stats['unique_hashes']}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Avg entries per hash: {stats['avg_entries_per_hash']:.2f}")
    
    return db


def example_5_matching(db, processor, peaks):
    """Example 5: Match fingerprints against database."""
    
    if db is None or processor is None:
        print("Skipping Example 5 (no database or processor)")
        return
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Fingerprint Matching")
    print("="*70)
    
    # Generate fingerprints for a sample
    fp_gen = FingerprintGenerator(processor.pairing_cfg)
    sample_fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)
    
    print(f"Sample has {len(sample_fingerprints)} unique fingerprints")
    
    # Match against database
    match_counts = db.match_fingerprints(sample_fingerprints)
    
    print(f"\nMatching results (by scenario):")
    for scenario, count in sorted(match_counts.items(), key=lambda x: -x[1]):
        print(f"  {scenario}: {count} matches")
    
    if match_counts:
        best_scenario = max(match_counts, key=match_counts.get)
        print(f"\nBest match: {best_scenario}")


def main():
    """Run all examples."""
    
    print("\n")
    print("#" * 70)
    print("# VIBRATION FINGERPRINTING PIPELINE - COMPLETE EXAMPLE")
    print("#" * 70)
    
    # Example 1: Load and explore
    sensor_data, config, loader = example_1_load_and_explore()
    
    # Example 2: STFT and peak detection
    processor, stft_magnitude, peaks, time_frames = example_2_stft_and_peaks(
        sensor_data, config
    )
    
    # Example 3: Fingerprinting
    fingerprints, fp_gen = example_3_fingerprinting(processor, peaks)
    
    # Example 4: Database
    db = example_4_database(processor, sensor_data, config)
    
    # Example 5: Matching
    example_5_matching(db, processor, peaks)
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
