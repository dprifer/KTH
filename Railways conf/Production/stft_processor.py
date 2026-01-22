"""
stft_processor.py

STFT computation, windowing, peak detection with frequency-band weighting.
"""

import numpy as np
from scipy import signal
from scipy.signal import get_window
from typing import Tuple, List, Dict
import warnings

from config import STFTConfig, FrequencyBandConfig, PeakDetectionConfig


class STFTProcessor:
    """Computes STFT and extracts peaks with physics-informed weighting."""
    
    def __init__(
        self,
        stft_config: STFTConfig,
        freq_bands_config: FrequencyBandConfig,
        peak_config: PeakDetectionConfig,
    ):
        """Initialize STFT processor.
        
        Args:
            stft_config: STFT parameters (window, FFT, frequency limits)
            freq_bands_config: Frequency band definitions and weights
            peak_config: Peak detection parameters
        """
        self.stft_cfg = stft_config
        self.freq_bands_cfg = freq_bands_config
        self.peak_cfg = peak_config
        
        # Precompute window
        self.window = get_window(
            self.stft_cfg.window_type,
            self.stft_cfg.window_length_samples
        )
        
        # Frequency grid for FFT output
        self.freqs = np.fft.rfftfreq(
            self.stft_cfg.fft_size,
            1.0 / self.stft_cfg.sampling_rate_hz
        )
        
        # Find indices corresponding to freq_min and freq_max
        self.freq_min_idx = np.searchsorted(self.freqs, self.stft_cfg.freq_min_hz)
        self.freq_max_idx = np.searchsorted(self.freqs, self.stft_cfg.freq_max_hz) + 1
        
        # Precompute band masks and weights
        self._compute_band_masks()
    
    def _compute_band_masks(self) -> None:
        """Precompute boolean masks for each frequency band."""
        self.band_masks = {}
        self.band_weights = {}
        
        for (f_min, f_max), weight in self.freq_bands_cfg.get_bands():
            band_name = f"{f_min:.1f}_{f_max:.1f}_Hz"
            
            # Boolean mask for frequencies in this band
            mask = (self.freqs >= f_min) & (self.freqs <= f_max)
            self.band_masks[band_name] = mask
            self.band_weights[band_name] = weight
    
    def compute_stft(
        self,
        signal_array: np.ndarray,
        detrend: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute STFT of a signal.
        
        Args:
            signal_array: 1D array of acceleration samples
            detrend: If True, apply linear detrending before STFT
        
        Returns:
            stft_magnitude: 2D array of shape (n_freqs, n_frames), magnitude spectrogram
            time_frames: 1D array of frame times in seconds
        """
        signal_array = np.asarray(signal_array).flatten()
        
        if detrend:
            signal_array = signal.detrend(signal_array)
        
        # Compute STFT using scipy (more stable than manual implementation)
        f, t, Zxx = signal.stft(
            signal_array,
            fs=self.stft_cfg.sampling_rate_hz,
            window=self.window,
            nperseg=self.stft_cfg.window_length_samples,
            noverlap=self.stft_cfg.window_length_samples - self.stft_cfg.stride_samples,
            nfft=self.stft_cfg.fft_size,
        )
        
        # Return magnitude spectrogram and time vector
        stft_magnitude = np.abs(Zxx)
        
        return stft_magnitude, t
    
    def extract_peaks_from_spectrogram(
        self,
        stft_magnitude: np.ndarray,
        time_frames: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """Extract peaks from spectrogram using percentile thresholding.
        
        Args:
            stft_magnitude: 2D magnitude spectrogram from compute_stft()
            time_frames: 1D time vector from compute_stft()
        
        Returns:
            peaks: List of (frequency_hz, time_sec, magnitude) tuples
                   sorted by magnitude (descending)
        """
        peaks = []
        
        # Process each frame independently
        for frame_idx in range(stft_magnitude.shape[1]):
            frame = stft_magnitude[:, frame_idx]
            frame_time = time_frames[frame_idx]
            
            # Restrict to frequency range if requested
            if self.peak_cfg.apply_band_restriction:
                freq_mask = np.zeros(len(frame), dtype=bool)
                for band_mask in self.band_masks.values():
                    freq_mask |= band_mask
                frame_restricted = frame.copy()
                frame_restricted[~freq_mask] = -np.inf
            else:
                frame_restricted = frame
            
            # Find peaks using local maxima (robust method)
            if self.peak_cfg.method == "local_max":
                # For each frequency, check if it's local max vs. neighbors
                local_max = self._find_local_maxima_1d(frame_restricted)
                peak_freq_indices = np.where(local_max)[0]
            
            elif self.peak_cfg.method == "scipy_peaks":
                # Use scipy.signal.find_peaks
                peak_freq_indices, _ = signal.find_peaks(
                    frame_restricted,
                    distance=self.peak_cfg.min_distance_bins,
                )
            else:
                raise ValueError(f"Unknown peak detection method: {self.peak_cfg.method}")
            
            # Extract peak values and apply weighting if requested
            if len(peak_freq_indices) > 0:
                peak_magnitudes = frame[peak_freq_indices]
                
                if self.peak_cfg.use_weighted_selection:
                    # Apply frequency band weights
                    weights = np.ones(len(peak_freq_indices))
                    for band_name, band_mask in self.band_masks.items():
                        band_weight = self.band_weights[band_name]
                        weights[band_mask[peak_freq_indices]] = band_weight
                    
                    weighted_magnitudes = peak_magnitudes * weights
                    # Sort by weighted magnitude and take top N
                    top_indices = np.argsort(-weighted_magnitudes)[:self.peak_cfg.peaks_per_frame]
                else:
                    # Sort by raw magnitude and take top N
                    top_indices = np.argsort(-peak_magnitudes)[:self.peak_cfg.peaks_per_frame]
                
                # Convert to (freq, time, magnitude) tuples
                for idx in top_indices:
                    freq_idx = peak_freq_indices[idx]
                    freq_hz = self.freqs[freq_idx]
                    magnitude = frame[freq_idx]
                    peaks.append((freq_hz, frame_time, magnitude))
        
        # Sort by magnitude (descending) for overall ranking
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        return peaks
    
    @staticmethod
    def _find_local_maxima_1d(signal_1d: np.ndarray, window: int = 3) -> np.ndarray:
        """Find local maxima in 1D signal.
        
        Args:
            signal_1d: 1D array
            window: Size of local neighborhood (must be odd)
        
        Returns:
            Boolean array marking local maxima
        """
        if window % 2 == 0:
            window += 1
        
        half_window = window // 2
        local_max = np.zeros(len(signal_1d), dtype=bool)
        
        for i in range(half_window, len(signal_1d) - half_window):
            is_max = signal_1d[i] == np.max(
                signal_1d[i - half_window:i + half_window + 1]
            )
            if is_max and signal_1d[i] > 0:  # Positive values only
                local_max[i] = True
        
        return local_max
    
    def process_signal(
        self,
        signal_array: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]], np.ndarray]:
        """End-to-end pipeline: STFT → peak detection.
        
        Args:
            signal_array: 1D acceleration signal
        
        Returns:
            stft_magnitude: Magnitude spectrogram
            peaks: List of (freq, time, magnitude) tuples
            time_frames: Time vector for spectrogram frames
        """
        stft_magnitude, time_frames = self.compute_stft(signal_array, detrend=True)
        peaks = self.extract_peaks_from_spectrogram(stft_magnitude, time_frames)
        
        return stft_magnitude, peaks, time_frames
