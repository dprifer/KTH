from dataclasses import dataclass, field
from typing import Tuple, List
import yaml


@dataclass
class STFTConfig:

    # Window parameters
    window_length_sec: float = 4.0  # seconds
    
    overlap: float = 0.9  # overlap ratio, range (0,1)
    
    window_type: str = "hann"  # can be 'hann', 'hamming', 'blackman', etc.
    
    # FFT parameters
    fft_size: int = 256  # must be power of 2 for efficiency
    
    sampling_rate_hz: float = 50.0  # sampling rate of simulation

    freq_min_hz: float = 0.5  # Nyquist theorem
    
    freq_max_hz: float = 25.0  # Nyquist frequency
    
    def __post_init__(self):
        """Validate parameters."""
        nyquist = self.sampling_rate_hz / 2
        if self.freq_max_hz > nyquist:
            raise ValueError(
                f"freq_max_hz ({self.freq_max_hz}) exceeds Nyquist "
                f"({nyquist}) for sampling rate {self.sampling_rate_hz} Hz"
            )
        if self.overlap <= 0 or self.overlap >= 1:
            raise ValueError("overlap must be in range (0, 1)")
    
    @property
    def window_length_samples(self) -> int:
        """Compute window length in samples."""
        return int(self.window_length_sec * self.sampling_rate_hz)
    
    @property
    def stride_samples(self) -> int:
        """Compute stride (hop length) in samples."""
        return int(self.window_length_samples * (1 - self.overlap))
    
    @property
    def freq_resolution_hz(self) -> float:
        """Frequency resolution of STFT in Hz."""
        return self.sampling_rate_hz / self.fft_size


@dataclass
class FrequencyBandConfig:
    """Physics-informed frequency band definitions."""
    # Decoupled bogie frequency: 6.875 Hz (k=4*1.2 kN/mm, m=2615 kg)
    # CB frequency: 1.167 Hz (k=4*430 N/mm, m=32000 kg)
    
    # Band boundaries for vertical suspension (Hz)
    body_mode: Tuple[float, float] = (0.5, 3.0)  # Carbody body bounce mode
    
    suspension: Tuple[float, float] = (3.0, 10.0)  # Bogie bounce modes
    
    high_freq: Tuple[float, float] = (10.0, 25.0)  # High-frequency content
    
    # Band weighting for peak detection
    body_mode_weight: float = 1.5
    
    suspension_weight: float = 1.2
    
    high_freq_weight: float = 0.8  # giving less importance to high frequency "noise"
    
    def get_bands(self) -> List[Tuple[Tuple[float, float], float]]:
        """Provide list of (frequency_range, weight) tuples."""
        return [
            (self.body_mode, self.body_mode_weight),
            (self.suspension, self.suspension_weight),
            (self.high_freq, self.high_freq_weight),
        ]


@dataclass
class PeakDetectionConfig:
    
    peaks_per_frame: int = 5  # Number of peaks to extract per STFT frame
    
    method: str = "local_max"  # Peak detection method: 'local_max' or 'scipy_peaks'
    
    min_distance_bins: int = 2  # Minimum distance between peaks in time-frequency bins (scipy_peaks)
    
    use_weighted_selection: bool = True  # Apply frequency-band weighting when selecting peaks, see FrequencyBandConfig
    
    apply_band_restriction: bool = True  # Restrict peaks to frequency_bands (discard outside [body, suspension, high_freq])


@dataclass
class PeakPairingConfig:

    time_window_sec: float = 2.0  # Time window forward of anchor peak for finding targets (seconds)

    freq_quantization_hz: float = 1.0  # Frequency quantization bin size (Hz)
    
    time_quantization_sec: float = 0.1  # Time quantization bin size (seconds): 100 ms
    
    # Pairing options
    require_same_band: bool = False  # If True, only pair peaks from same frequency band. If False, pair any peaks.
    
    @property
    def time_quantization_bins(self) -> int:
        """Number of time bins for quantization."""
        return int(self.time_window_sec / self.time_quantization_sec)
    
    @property
    def freq_quantization_bins(self) -> int:
        """Number of frequency bins in 0.5-25 Hz range."""
        return int((25.0 - 0.5) / self.freq_quantization_hz) + 1  # UPDATE when min-max changes


@dataclass
class SegmentationConfig:
    
    subsample_factor: int = 1  # Use every Nth segment. 1 = all segments, 2 = every other, etc
    
    min_segments_per_scenario: int = 10  # Minimum number of segments to extract per scenario (after subsampling)


@dataclass
class DatabaseConfig:
    
    combine_sensors: bool = True
    """If True, combine fingerprints from all sensors into single database.
    If False, create separate database per sensor."""
    
    separate_databases_per_scenario: bool = False
    """If True, create separate database for each scenario.
    If False, combine all scenarios into single database."""


@dataclass
class FingerprinterConfig:
    """Main configuration container."""
    
    stft: STFTConfig = field(default_factory=STFTConfig)
    freq_bands: FrequencyBandConfig = field(default_factory=FrequencyBandConfig)
    peak_detection: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)
    peak_pairing: PeakPairingConfig = field(default_factory=PeakPairingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "FingerprinterConfig":
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses from dict
        config = cls()
        
        if 'stft' in data:
            config.stft = STFTConfig(**data['stft'])
        if 'freq_bands' in data:
            config.freq_bands = FrequencyBandConfig(**data['freq_bands'])
        if 'peak_detection' in data:
            config.peak_detection = PeakDetectionConfig(**data['peak_detection'])
        if 'peak_pairing' in data:
            config.peak_pairing = PeakPairingConfig(**data['peak_pairing'])
        if 'segmentation' in data:
            config.segmentation = SegmentationConfig(**data['segmentation'])
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        
        return config
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'stft': self.stft.__dict__,
            'freq_bands': self.freq_bands.__dict__,
            'peak_detection': self.peak_detection.__dict__,
            'peak_pairing': self.peak_pairing.__dict__,
            'segmentation': self.segmentation.__dict__,
            'database': self.database.__dict__,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def print_summary(self) -> None:
        """Print human-readable summary of configuration."""
        print("=" * 70)
        print("FINGERPRINTER CONFIGURATION SUMMARY")
        print("=" * 70)
        
        print("\nSTFT PARAMETERS:")
        print(f"  Window length: {self.stft.window_length_sec} s "
              f"({self.stft.window_length_samples} samples)")
        print(f"  Overlap: {self.stft.overlap * 100:.0f}% "
              f"(stride: {self.stft.stride_samples} samples)")
        print(f"  Window type: {self.stft.window_type}")
        print(f"  FFT size: {self.stft.fft_size}")
        print(f"  Frequency resolution: {self.stft.freq_resolution_hz:.3f} Hz")
        print(f"  Frequency range: {self.stft.freq_min_hz}–{self.stft.freq_max_hz} Hz")
        
        print("\nFREQUENCY BANDS:")
        for (f_min, f_max), weight in self.freq_bands.get_bands():
            print(f"  {f_min:5.1f}–{f_max:5.1f} Hz (weight: {weight:.1f})")
        
        print("\nPEAK DETECTION:")
        print(f"  Peaks per frame: {self.peak_detection.peaks_per_frame}")
        print(f"  Method: {self.peak_detection.method}")
        print(f"  Weighted selection: {self.peak_detection.use_weighted_selection}")
        print(f"  Band restriction: {self.peak_detection.apply_band_restriction}")
        
        print("\nPEAK PAIRING:")
        print(f"  Time window: {self.peak_pairing.time_window_sec} s")
        print(f"  Freq quantization: {self.peak_pairing.freq_quantization_hz} Hz")
        print(f"  Time quantization: {self.peak_pairing.time_quantization_sec} s")
        print(f"  Hash space size: "
              f"~{self.peak_pairing.freq_quantization_bins**2 * self.peak_pairing.time_quantization_bins:,} "
              f"(freq² × time)")
        
        print("\nSEGMENTATION:")
        print(f"  Subsample factor: {self.segmentation.subsample_factor}")
        print(f"  Min segments per scenario: {self.segmentation.min_segments_per_scenario}")
        
        print("\nDATABASE:")
        print(f"  Combine sensors: {self.database.combine_sensors}")
        print(f"  Separate per scenario: {self.database.separate_databases_per_scenario}")
        
        print("=" * 70 + "\n")
