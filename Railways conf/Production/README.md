"""
README - VIBRATION FINGERPRINTING LIBRARY

A Python library for sparse vibration fingerprinting inspired by Shazam,
adapted for railway vehicle condition monitoring using multibody simulations.

## QUICK START

### 1. Install Dependencies

pip install numpy scipy scikit-signal pyyaml scikit-learn matplotlib

### 2. Configure Parameters

The library uses a hierarchical configuration system with sensible defaults:

```python
from config import FingerprinterConfig

# Use defaults (recommended parameters already set)
config = FingerprinterConfig()

# Print configuration
config.print_summary()

# Customize if needed
config.stft.window_length_sec = 4.0
config.peak_detection.peaks_per_frame = 5

# Save/load from YAML
config.to_yaml("my_config.yaml")
config_loaded = FingerprinterConfig.from_yaml("my_config.yaml")
```

### 3. Load Simulations

```python
from io import SimulationLoader

loader = SimulationLoader("path/to/mat/files")

# Load one scenario
sensor_data = loader.load_scenario("S1_ERRI1_K1")
# Returns: {"carbody": (accel, time, distance), "bogie": (...), "axlebox": (...)}

# Load all scenarios
all_data = loader.load_all_scenarios()
```

**NOTE**: You may need to adjust the data extraction paths in `io.py:SimulationLoader.load_scenario()`
to match your actual .mat file structure. Look for lines with:
```python
acceleration = np.squeeze(
    mat_data["RS_result"]["SRS_G_Accelerometers"]["SRS_B1_WS1_R"][sensor_info["channel_name"]]["values"]
)
```

### 4. Process Signals: STFT + Peak Detection

```python
from stft_processor import STFTProcessor

processor = STFTProcessor(
    config.stft,
    config.freq_bands,
    config.peak_detection,
)

# Process a signal (returns spectrogram and peaks)
accel, time, dist = sensor_data["carbody"]
stft_magnitude, peaks, time_frames = processor.process_signal(accel)

# peaks is a list of (freq_hz, time_sec, magnitude) tuples
# Peaks are already sorted by magnitude (descending)
print(f"Extracted {len(peaks)} peaks")
```

### 5. Generate Fingerprints

```python
from fingerprint import FingerprintGenerator

fp_gen = FingerprintGenerator(config.peak_pairing)

# Generate fingerprints from peaks
fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)
# Returns: {(f1_bin, f2_bin, t_bin) -> [anchor_times]}

# Get statistics
stats = fp_gen.hash_statistics(fingerprints)
print(f"Unique hashes: {stats['unique_hashes']}")
print(f"Total occurrences: {stats['total_hashes']}")
```

### 6. Build Database

```python
from database import FingerprintDatabase
from io import SegmentationHelper

db = FingerprintDatabase(combine_sensors=True)

# Process segments from a scenario
segments = SegmentationHelper.extract_segments(
    accel, time,
    config.stft.window_length_sec,
    config.stft.overlap,
)

# For each segment, generate fingerprints and add to database
for seg_signal, seg_time, seg_idx in segments:
    stft_mag, peaks, _ = processor.process_signal(seg_signal)
    fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)
    
    db.add_fingerprints_batch(
        fingerprints,
        scenario_name="S1_ERRI1_K1",
        segment_index=seg_idx,
        sensor_name="carbody",
        frame_time_sec=seg_time,
    )

# Check database
stats = db.get_statistics()
print(f"Database: {stats['unique_hashes']} unique hashes")

# Save database
db.save("database.pkl")

# Load database
db_loaded = FingerprintDatabase.load("database.pkl")
```

### 7. Match Fingerprints

```python
# Match an unknown sample against database
sample_fingerprints = fp_gen.generate_fingerprints_from_peaks(peaks)
match_counts = db.match_fingerprints(sample_fingerprints)

# Results: {scenario_name -> number_of_matches}
print(match_counts)

# Identify best match
if match_counts:
    best_scenario = max(match_counts, key=match_counts.get)
    print(f"Best match: {best_scenario} ({match_counts[best_scenario]} matches)")
```

### 8. Compute Metrics for Experiments

```python
from database import MatchingScorer

# For Experiment 1: Condition Separability
# Collect results: [(true_scenario, match_counts_dict), ...]
results = [
    ("S1_ERRI1_K1", {"S1_ERRI1_K1": 150, "S2_ERRI1_K0d7": 5}),
    ("S2_ERRI1_K0d7", {"S2_ERRI1_K0d7": 120, "S1_ERRI1_K1": 35}),
    # ... more results
]

# Compute confusion matrix
confusion_matrix, scenario_names = MatchingScorer.compute_confusion_matrix(results)
print(f"Accuracy: {np.trace(confusion_matrix) / np.sum(confusion_matrix):.2%}")
```

## LIBRARY STRUCTURE

```
fingerprinting_lib/
├── config.py              # Configuration classes (STFTConfig, PeakPairingConfig, etc.)
├── stft_processor.py      # STFT, windowing, peak detection with frequency weighting
├── fingerprint.py         # Peak pairing, quantization, hash generation
├── database.py            # Database building, matching, scoring
├── io.py                  # Simulation loading, segmentation, metadata
├── plotting.py            # (Optional) Visualization functions
└── example_usage.py       # Complete working example
```

## KEY PARAMETERS AND DEFAULTS

### STFT Parameters
- **Window length**: 4.0 seconds
- **Overlap**: 0.9 (90%)
- **Window type**: Hann
- **FFT size**: 256 (zero-padded)
- **Sampling rate**: 50 Hz
- **Frequency range**: 0.5–25 Hz

### Frequency Bands (Physics-Informed)
- **Body mode (0.5–3 Hz)**: Weight 1.5× - most sensitive to damage
- **Suspension (3–10 Hz)**: Weight 1.2× - sensitive to damage
- **High freq (10–25 Hz)**: Weight 0.8× - less sensitive

### Peak Detection
- **Peaks per frame**: 5
- **Detection method**: Local maxima (robust)
- **Weighted selection**: Yes (applies frequency band weights)
- **Band restriction**: Yes (peaks only in 0.5–25 Hz)

### Peak Pairing
- **Time window**: 2.0 seconds (forward from anchor)
- **Frequency quantization**: 1.0 Hz
- **Time quantization**: 0.1 seconds
- **Hash format**: (freq_bin_1, freq_bin_2, time_bin) → integer

### Database
- **Combine sensors**: True (single database)
- **Separate per scenario**: False (combined database for all scenarios)

## ADVANCED: CUSTOMIZATION

### Change STFT Parameters

```python
config.stft.window_length_sec = 5.0  # Use 5-second windows
config.stft.overlap = 0.95           # 95% overlap
config.stft.fft_size = 512           # Higher frequency resolution
```

### Change Frequency Bands

```python
# Define custom bands
config.freq_bands.body_mode = (0.3, 2.0)
config.freq_bands.body_mode_weight = 2.0  # Higher weight
config.freq_bands.high_freq_weight = 0.5  # Lower weight
```

### Use Individual Sensor Databases

```python
# Instead of combining all sensors
db = FingerprintDatabase(combine_sensors=False)

# Then match per sensor
match_counts = db.match_fingerprints(sample_fp, sensor_name="carbody")
match_counts_bogie = db.match_fingerprints(sample_fp, sensor_name="bogie")
```

### Custom Quantization

```python
config.peak_pairing.freq_quantization_hz = 0.5   # Finer quantization
config.peak_pairing.time_quantization_sec = 0.05 # Finer time resolution
# Warning: Finer quantization = larger hash space, less robust
```

## TROUBLESHOOTING

### "No peaks detected"
- Check signal amplitude (should be normalized or have reasonable variance)
- Increase `config.peak_detection.peaks_per_frame`
- Check frequency band limits (may be excluding all peaks)

### "Low match rates"
- Verify peak detection is working (plot spectrogram with peaks)
- Check quantization parameters (may be too coarse/fine)
- Ensure database has sufficient fingerprints from same scenario

### "AttributeError: 'dict' object has no attribute 'values'"
- Likely issue in loading .mat file structure. Verify `io.py` data extraction paths.
- Print the MATLAB data structure: `print(mat_data.keys())` to debug

### Memory issues with large simulations
- Use `config.segmentation.subsample_factor > 1` to skip segments
- Process scenarios one at a time instead of all at once
- Use separate databases per sensor (`combine_sensors=False`)

## NEXT STEPS

1. **Implement plotting functions** in `plotting.py`:
   - `plot_spectrogram_with_peaks(stft_magnitude, peaks, freq_grid, time_grid)`
   - `plot_constellation_map(peaks, fingerprints)`
   - `plot_confusion_matrix(confusion_matrix, scenario_names)`

2. **Run Experiment 1 (Condition Separability)**:
   - Load each scenario
   - Extract and fingerprint all segments
   - Build separate database from training set
   - Test on held-out test set
   - Compute confusion matrix

3. **Run Experiment 2 (Robustness to Noise)**:
   - Add synthetic Gaussian noise to signals at various SNR levels
   - Recompute fingerprints and match rates
   - Compare spectrogram SNR degradation vs. fingerprint robustness

4. **Run Experiment 3 (Severity Sensitivity)**:
   - Generate stiffness sweep (K = 1.0, 0.9, 0.8, ..., 0.5)
   - Compute similarity metric between each degraded case and baseline
   - Plot similarity vs. stiffness factor

## CITATION

If using this library, please cite the original Shazam algorithm:
Wang, A. L., "An Industrial-Strength Audio Search Algorithm," 
in Proceedings of the International Society for Music Information 
Retrieval Conference (ISMIR), 2003, pp. 7–13.

And reference this work as:
[Your Name], "Exploring Sparse Vibration Fingerprints for Rail Vehicle 
Condition Monitoring Using Multibody Simulations," [Conference], 2026.
"""

# This is a markdown README, save as README.md or README.txt
