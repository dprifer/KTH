"""
database.py

Fingerprint database building, matching, and scoring.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from fingerprint import FingerprintGenerator


@dataclass
class FingerprintDatabaseEntry:
    """Single entry in the fingerprint database."""
    hash_tuple: Tuple[int, int, int]
    scenario_name: str
    segment_index: int
    sensor_name: str
    anchor_time_sec: float
    frame_time_sec: float


class FingerprintDatabase:
    """Database for storing and matching fingerprints."""
    
    def __init__(self):
        """Initialize fingerprint database"""

        # Or if not combining sensors: {sensor -> {hash_tuple -> [(scenario, segment_idx, time), ...]}}
        

        self.db = {}  # Separate dbs per sensor
        
        # Metadata for reconstruction
        self.scenario_names = set()
        self.sensor_names = set()
        self.segments_per_scenario = defaultdict(int)
    
    def add_fingerprint(
        self,
        hash_tuple: Tuple[int, int, int],
        scenario_name: str,
        segment_index: int,
        sensor_name: str,
        anchor_time_sec: float,
        frame_time_sec: float,
    ) -> None:
        """Add a single fingerprint to the database.
        
        Args:
            hash_tuple: Quantized fingerprint hash
            scenario_name: Name of simulation scenario (e.g., "S1_ERRI1_K1")
            segment_index: Index of segment within simulation
            sensor_name: Sensor location (e.g., "carbody", "bogie", "axlebox")
            anchor_time_sec: Time of anchor peak within segment
            frame_time_sec: Absolute time in simulation
        """
        self.scenario_names.add(scenario_name)
        self.sensor_names.add(sensor_name)
        
        record = (scenario_name, segment_index, sensor_name, anchor_time_sec, frame_time_sec)

        if sensor_name not in self.db:
            self.db[sensor_name] = defaultdict(list)
        self.db[sensor_name][hash_tuple].append(record)
    
    def add_fingerprints_batch(
        self,
        fingerprints: Dict[Tuple[int, int, int], List[float]],
        scenario_name: str,
        segment_index: int,
        sensor_name: str,
        frame_time_sec: float,
    ) -> None:
        """Add multiple fingerprints from one segment.
        
        Args:
            fingerprints: From FingerprintGenerator.generate_fingerprints_from_peaks()
            scenario_name: Scenario identifier
            segment_index: Segment index
            sensor_name: Sensor identifier
            frame_time_sec: Time of first frame in segment
        """
        for hash_tuple, anchor_times in fingerprints.items():
            for anchor_time in anchor_times:
                self.add_fingerprint(
                    hash_tuple,
                    scenario_name,
                    segment_index,
                    sensor_name,
                    anchor_time,
                    frame_time_sec,
                )
    
    def match_fingerprints(
        self,
        sample_fingerprints: Dict[Tuple[int, int, int], List[float]],
        sensor_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """Match a sample's fingerprints against database.
        
        Args:
            sample_fingerprints: Fingerprints from unknown sample
            sensor_name: If not combining sensors, specify which sensor database to query
        
        Returns:
            Dictionary mapping scenario name to number of matching hashes
        """
        match_counts = defaultdict(int)

        if sensor_name is None:
            raise ValueError("sensor_name required when not combining sensors")
        if sensor_name not in self.db:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        query_db = self.db[sensor_name]
        
        # Count matches per scenario
        for hash_tuple in sample_fingerprints.keys():
            if hash_tuple in query_db:
                # This sample hash matches database hashes
                records = query_db[hash_tuple]
                
                for scenario_name, segment_idx, sensor, anchor_time, frame_time in records:
                    match_counts[scenario_name] += 1
        
        return dict(match_counts)
    
    def match_fingerprints_detailed(
        self,
        sample_fingerprints: Dict[Tuple[int, int, int], List[float]],
        sensor_name: Optional[str] = None,
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """Match with detailed records (segment index, count, time).
        
        Args:
            sample_fingerprints: Fingerprints from unknown sample
            sensor_name: If not combining sensors, specify which sensor database to query
        
        Returns:
            Dictionary mapping scenario name to list of (segment_idx, match_count, frame_time) tuples
        """
        matches = defaultdict(int)

        if sensor_name is None:
            raise ValueError("sensor_name required when not combining sensors")
        if sensor_name not in self.db:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        query_db = self.db[sensor_name]
        
        # Build detailed matches
        for hash_tuple in sample_fingerprints.keys():
            if hash_tuple in query_db:
                records = query_db[hash_tuple]
                
                for scenario_name, segment_idx, sensor, anchor_time, frame_time in records:
                    key = (scenario_name, segment_idx)
                    matches[key] += 1
        
        # Reshape: {scenario -> [(segment_idx, count, time), ...]}
        result = defaultdict(list)
        for (scenario, segment_idx), count in matches.items():
            result[scenario].append((segment_idx, count))
        
        return dict(result)
    
    def get_statistics(self) -> Dict:
        """Compute database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        num_unique_hashes = sum(len(db) for db in self.db.values())
        num_entries = sum(
            sum(len(records) for records in db.values())
            for db in self.db.values()
        )
        
        stats = {
            "num_scenarios": len(self.scenario_names),
            "scenario_names": sorted(self.scenario_names),
            "num_sensors": len(self.sensor_names),
            "sensor_names": sorted(self.sensor_names),
            "unique_hashes": num_unique_hashes,
            "total_entries": num_entries,
            "avg_entries_per_hash": num_entries / num_unique_hashes if num_unique_hashes > 0 else 0,
        }
        
        return stats
    
    def save(self, filepath: str) -> None:
        """Save database to disk.
        
        Args:
            filepath: Path to save (pickle format)
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'db': dict(self.db) if isinstance(self.db, defaultdict) else self.db,
                'scenario_names': self.scenario_names,
                'sensor_names': self.sensor_names,
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> "FingerprintDatabase":
        """Load database from disk.
        
        Args:
            filepath: Path to saved database
        
        Returns:
            Loaded FingerprintDatabase instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        db = cls()
        db.db = data['db']
        db.scenario_names = data['scenario_names']
        db.sensor_names = data['sensor_names']
        
        return db


class MatchingScorer:
    """Score matching results across multiple scenarios."""
    
    @staticmethod
    def compute_confusion_matrix(
        match_results: List[Tuple[str, Dict[str, int]]],
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute confusion matrix from matching results.
        
        Args:
            match_results: List of (true_scenario, match_counts_dict) tuples
        
        Returns:
            confusion_matrix: 2D numpy array (true scenario × predicted scenario)
            scenario_names: List of scenario names
        """
        # Collect all unique scenario names
        all_scenarios = set()
        for true_scenario, match_counts in match_results:
            all_scenarios.add(true_scenario)
            all_scenarios.update(match_counts.keys())
        
        scenario_list = sorted(all_scenarios)
        n_scenarios = len(scenario_list)
        
        confusion_matrix = np.zeros((n_scenarios, n_scenarios))
        
        for true_scenario, match_counts in match_results:
            true_idx = scenario_list.index(true_scenario)
            
            # Find best match (scenario with most matches)
            if match_counts:
                pred_scenario = max(match_counts, key=match_counts.get)
                pred_idx = scenario_list.index(pred_scenario)
            else:
                # No matches found
                pred_idx = true_idx  # Default to diagonal (incorrect)
            
            confusion_matrix[true_idx, pred_idx] += 1
        
        return confusion_matrix, scenario_list
    
    @staticmethod
    def compute_match_ratios(
        match_results: List[Tuple[str, Dict[str, int]]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute match ratios (matches per scenario vs. total).
        
        Args:
            match_results: List of (true_scenario, match_counts_dict) tuples
        
        Returns:
            Dictionary mapping scenario name to per-scenario ratios
        """
        # Group by scenario
        by_scenario = defaultdict(list)
        for true_scenario, match_counts in match_results:
            by_scenario[true_scenario].append(match_counts)
        
        ratios = {}
        for scenario_name, match_list in by_scenario.items():
            total_matches = [sum(m.values()) for m in match_list]
            
            if total_matches:
                avg_total = np.mean(total_matches)
                
                # Average ratio for each matched scenario
                scenario_ratios = defaultdict(list)
                for match_counts in match_list:
                    total = sum(match_counts.values())
                    if total > 0:
                        for matched_scenario, count in match_counts.items():
                            scenario_ratios[matched_scenario].append(count / total)
                
                ratios[scenario_name] = {
                    s: np.mean(r) for s, r in scenario_ratios.items()
                }
        
        return ratios


