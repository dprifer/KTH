"""
Peak pairing, quantization, and hash generation for fingerprinting.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass

from config import PeakPairingConfig


@dataclass
class Peak:
    """Single spectral peak."""
    freq_hz: float
    time_sec: float
    magnitude: float


@dataclass
class PeakPair:
    """Pair of peaks for fingerprinting."""
    anchor_freq_hz: float
    anchor_time_sec: float
    target_freq_hz: float
    target_time_sec: float
    time_delta_sec: float
    
    def to_quantized(self, pairing_cfg: PeakPairingConfig) -> Tuple[int, int, int]:
        """Quantize peak pair into discrete bins.
        
        Args:
            pairing_cfg: Configuration with quantization parameters
        
        Returns:
            Tuple of (freq_bin_1, freq_bin_2, time_bin) ready for hashing
        """
        # Quantize frequencies
        freq_1_bin = int(self.anchor_freq_hz / pairing_cfg.freq_quantization_hz)
        freq_2_bin = int(self.target_freq_hz / pairing_cfg.freq_quantization_hz)
        
        # Quantize time delta
        time_bin = int(self.time_delta_sec / pairing_cfg.time_quantization_sec)
        
        # Clamp to valid ranges
        freq_1_bin = np.clip(freq_1_bin, 0, pairing_cfg.freq_quantization_bins - 1)
        freq_2_bin = np.clip(freq_2_bin, 0, pairing_cfg.freq_quantization_bins - 1)
        time_bin = np.clip(time_bin, 0, pairing_cfg.time_quantization_bins - 1)
        
        return (freq_1_bin, freq_2_bin, time_bin)


class FingerprintGenerator:
    """Generates constellation fingerprints from spectral peaks."""
    
    def __init__(self, pairing_cfg: PeakPairingConfig):
        """Initialize fingerprint generator"""

        self.pairing_cfg = pairing_cfg
    
    def generate_fingerprints_from_peaks(
        self,
        peaks: List[Tuple[float, float, float]],  # (freq, time, mag)
    ) -> Dict[Tuple[int, int, int], List[float]]:
        """Generate fingerprints (peak pairs) from a list of peaks.
        
        Args:
            peaks: List of (freq_hz, time_sec, magnitude) tuples
        
        Returns:
            Dictionary mapping quantized hash tuples to list of anchor times
        """
        if len(peaks) < 2:
            return {}
        
        # Convert to Peak objects
        peak_objs = [Peak(freq, time, mag) for freq, time, mag in peaks]
        
        fingerprints = {}
        
        # For each peak as anchor
        for i, anchor in enumerate(peak_objs):
            # Find target peaks within time window
            time_max = anchor.time_sec + self.pairing_cfg.time_window_sec
            
            for j, target in enumerate(peak_objs):
                # Skip self-pairing
                if i == j:
                    continue
                
                # Check time constraint
                if target.time_sec <= anchor.time_sec or target.time_sec > time_max:
                    continue
                
                # Create peak pair
                time_delta = target.time_sec - anchor.time_sec
                pair = PeakPair(
                    anchor_freq_hz=anchor.freq_hz,
                    anchor_time_sec=anchor.time_sec,
                    target_freq_hz=target.freq_hz,
                    target_time_sec=target.time_sec,
                    time_delta_sec=time_delta,
                )
                
                # Quantize to hash
                hash_tuple = pair.to_quantized(self.pairing_cfg)
                
                # Store anchor time for later matching
                if hash_tuple not in fingerprints:
                    fingerprints[hash_tuple] = []
                fingerprints[hash_tuple].append(anchor.time_sec)
        
        return fingerprints
    
    def hash_to_integer(
        self,
        hash_tuple: Tuple[int, int, int],
    ) -> int:
        """Convert quantized hash tuple to single integer for fast lookup.
        
        Uses Morton encoding (Z-order curve) for better spatial locality.
        
        Args:
            hash_tuple: (freq_bin_1, freq_bin_2, time_bin)
        
        Returns:
            Single integer hash
        """
        f1, f2, t = hash_tuple
        
        # Simple approach: pack into 32-bit integer
        # Assumes freq bins < 256, time bins < 256
        # Format: [f1 (8 bits) | f2 (8 bits) | t (8 bits) | padding (8 bits)]
        
        int_hash = (f1 << 16) | (f2 << 8) | t
        
        return int_hash
    
    def integer_to_hash(self, int_hash: int) -> Tuple[int, int, int]:
        """Reverse operation: convert integer back to hash tuple."""
        f1 = (int_hash >> 16) & 0xFF
        f2 = (int_hash >> 8) & 0xFF
        t = int_hash & 0xFF
        return (f1, f2, t)
    
    def format_hash_for_display(
        self,
        hash_tuple: Tuple[int, int, int],
    ) -> str:
        """Human-readable hash representation.
        
        Args:
            hash_tuple: (freq_bin_1, freq_bin_2, time_bin)
        
        Returns:
            String representation
        """
        f1, f2, t = hash_tuple
        return f"F({f1:3d},{f2:3d})_Δt({t:3d})"
    
    @staticmethod
    def hash_statistics(fingerprints: Dict[Tuple[int, int, int], List[float]]) -> Dict:
        """Compute statistics on generated fingerprints.
        
        Args:
            fingerprints: From generate_fingerprints_from_peaks()
        
        Returns:
            Dictionary with statistics
        """
        if not fingerprints:
            return {
                "total_hashes": 0,
                "unique_hashes": 0,
                "avg_occurrences": 0,
                "max_occurrences": 0,
            }
        
        occurrences = [len(times) for times in fingerprints.values()]
        
        return {
            "total_hashes": sum(occurrences),
            "unique_hashes": len(fingerprints),
            "avg_occurrences": np.mean(occurrences),
            "max_occurrences": np.max(occurrences),
            "min_occurrences": np.min(occurrences),
        }


class ConstellationMap:
    """Visualization-friendly representation of peak constellation."""
    
    def __init__(
        self,
        peaks: List[Tuple[float, float, float]],  # (freq, time, mag)
        fingerprints: Dict[Tuple[int, int, int], List[float]],
    ):
        """Initialize constellation map.
        
        Args:
            peaks: List of spectral peaks
            fingerprints: Generated peak-pair fingerprints
        """
        self.peaks = peaks
        self.fingerprints = fingerprints
        
        # Build list of pairs for visualization
        self.pairs = []
        for hash_tuple, anchor_times in fingerprints.items():
            for anchor_time in anchor_times:
                self.pairs.append((hash_tuple, anchor_time))
    
    def get_peak_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract peak coordinates and magnitudes.
        
        Returns:
            times: 1D array of peak times (seconds)
            freqs: 1D array of peak frequencies (Hz)
            magnitudes: 1D array of peak magnitudes
        """
        times = np.array([p[1] for p in self.peaks])
        freqs = np.array([p[0] for p in self.peaks])
        mags = np.array([p[2] for p in self.peaks])
        
        return times, freqs, mags
    
    def get_pairing_statistics(self) -> Dict:
        """Compute statistics on peak pairings.
        
        Returns:
            Dictionary with pairing statistics
        """
        num_pairs = len(self.pairs)
        num_unique_hashes = len(self.fingerprints)
        
        # Distribution of pair occurrences
        occurrences = [len(times) for times in self.fingerprints.values()]
        
        return {
            "total_pairs": num_pairs,
            "unique_hashes": num_unique_hashes,
            "pairs_per_hash_mean": np.mean(occurrences) if occurrences else 0,
            "pairs_per_hash_std": np.std(occurrences) if occurrences else 0,
        }
