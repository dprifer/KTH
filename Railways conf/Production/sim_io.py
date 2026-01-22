"""
input/output definition

Loading simulations, organizing metadata, managing files.
"""

import scipy.io
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import h5py


@dataclass
class SimulationMetadata:
    """Metadata for a simulation scenario."""
    scenario_name: str
    track_excitation: str
    primary_stiffness: float
    speed_kmh: float
    duration_sec: float
    track_length_m: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'scenario_name': self.scenario_name,
            'track_excitation': self.track_excitation,
            'primary_stiffness': self.primary_stiffness,
            'speed_kmh': self.speed_kmh,
            'duration_sec': self.duration_sec,
            'track_length_m': self.track_length_m,
        }


class SimulationLoader:
    """Load and parse simulation data from .mat files."""
    
    # Mapping from scenario naming convention to metadata
    SCENARIO_METADATA = {
        "S1_ERRI1_K1": SimulationMetadata(
            scenario_name="S1_ERRI1_K1",
            track_excitation="ERRI high, 1×",
            primary_stiffness=1.0,
            speed_kmh=100,
            duration_sec=120,
            track_length_m=3300,
        ),
        "S2_ERRI1_K0d7": SimulationMetadata(
            scenario_name="S2_ERRI1_K0d7",
            track_excitation="ERRI high, 1×",
            primary_stiffness=0.7,
            speed_kmh=100,
            duration_sec=120,
            track_length_m=3300,
        ),
        "S3_ERRI1_K0d5": SimulationMetadata(
            scenario_name="S3_ERRI1_K0d5",
            track_excitation="ERRI high, 1×",
            primary_stiffness=0.5,
            speed_kmh=100,
            duration_sec=120,
            track_length_m=3300,
        ),
        "S4_ERRI1_K2": SimulationMetadata(
            scenario_name="S4_ERRI1_K1d35",
            track_excitation="ERRI high, 1×",
            primary_stiffness=1.35,  # Broken coil spring (increased stiffness by 35% due to reduced number of coils)
            speed_kmh=100,
            duration_sec=120,
            track_length_m=3300,
        ),
        "S5_ERRI1d5_K1": SimulationMetadata(
            scenario_name="S5_ERRI1d5_K1",
            track_excitation="ERRI high, 1.5×",
            primary_stiffness=1.0,
            speed_kmh=100,
            duration_sec=120,
            track_length_m=3300,
        ),
        "S6_ERRI2_K1": SimulationMetadata(
            scenario_name="S6_ERRI2_K1",
            track_excitation="ERRI high, 2×",
            primary_stiffness=1.0,
            speed_kmh=100,
            duration_sec=120,
            track_length_m=3300,
        ),
    }

    SENSOR_DEFINITIONS = {
        "carbody_B1": {
            "sensor_location": "SRS_CB_B1",
            "description": "Carbody acceleration above B1",
        },
        "carbody_cab": {
            "sensor_location": "SRS_CB_cab",
            "description": "Carbody acceleration in driver's cab",
        },
        "bogie": {
            "sensor_location": "SRS_B1_R",
            "description": "Front bogie frame acceleration on right side vert. sec. damper bracket",
        },
        "axlebox": {
            "sensor_location": "SRS_B1_WS1_R",
            "description": "Axlebox acceleration on WS1 right side",
        },
    }
    
    def __init__(self, mat_dir: str):

        self.mat_dir = Path(mat_dir)
        if not self.mat_dir.exists():
            raise FileNotFoundError(f"Simulation directory not found: {mat_dir}")

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

    def load_scenario(
        self,
        scenario_name: str,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load a single scenario's simulation data based on scenario identifier (e.g., "S1_ERRI1_K1")
        
        Returns:
            Dictionary mapping sensor name to acceleration, time, distance
        """
        mat_file = self.mat_dir / f"{scenario_name}.mat"
        
        if not mat_file.exists():
            raise FileNotFoundError(f"Simulation file not found: {mat_file}")


        sensor_data = {}
        
        for sensor_name, sensor_info in self.SENSOR_DEFINITIONS.items():
            try:
                with h5py.File(mat_file, "r") as f:

                    acceleration = np.array(f["timeInt"]["RS_result"]["SRS_G_Accelerometers"]
                        [sensor_info["sensor_location"]]
                        ["ch_003"]["values"]  # ch_003: vertical acceleration
                    )

                    time = np.array(f["timeInt"]["time"]["values"]
                    )

                    distance = np.array(f["timeInt"]["bodyPosTrans"]["SB_CarBody"]["x"]["values"]
                    )

                # Flatten
                acceleration = np.squeeze(acceleration).T
                time = np.squeeze(time).T
                distance = np.squeeze(distance).T
                
                sensor_data[sensor_name] = (acceleration, time, distance)
            
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not load {sensor_name} for {scenario_name}: {e}")
        
        return sensor_data
    
    # not used yet
    def load_all_scenarios(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Load all scenarios into a directory.
        
        Returns:
            Dictionary mapping scenario name to sensor data
        """
        all_data = {}
        
        for scenario_name in self.SCENARIO_METADATA.keys():
            try:
                sensor_data = self.load_scenario(scenario_name)
                all_data[scenario_name] = sensor_data
                print(f"✓ Loaded {scenario_name}")
            except FileNotFoundError as e:
                print(f"✗ {e}")
        
        return all_data
    
    def get_metadata(self, scenario_name: str) -> SimulationMetadata:
        """Get simulation parameters for a scenario"""

        if scenario_name not in self.SCENARIO_METADATA:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        return self.SCENARIO_METADATA[scenario_name]


    # not used yet
    @staticmethod
    def save_metadata(filepath: str, metadata_dict: Dict[str, SimulationMetadata]) -> None:
        """Save scenario metadata to JSON file.
        
        Args:
            filepath: Output JSON file path
            metadata_dict: Dictionary of SimulationMetadata instances
        """
        data = {
            name: metadata.to_dict()
            for name, metadata in metadata_dict.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metadata saved to {filepath}")

    # not used yet
    @staticmethod
    def load_metadata(filepath: str) -> Dict[str, SimulationMetadata]:
        """Load scenario metadata from JSON file.
        
        Args:
            filepath: Input JSON file path
        
        Returns:
            Dictionary of SimulationMetadata instances
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata_dict = {
            name: SimulationMetadata(
                scenario_name=name,
                track_excitation=info['track_excitation'],
                primary_stiffness=info['primary_stiffness'],
                speed_kmh=info['speed_kmh'],
                duration_sec=info['duration_sec'],
                track_length_m=info['track_length_m'],
            )
            for name, info in data.items()
        }
        
        return metadata_dict


class SegmentationHelper:
    """Extract overlapping segments from full simulation signals."""
    
    @staticmethod
    def extract_segments(
        acceleration: np.ndarray,
        time: np.ndarray,
        window_length_sec: float,
        overlap: float = 0.9,
        subsample_factor: int = 1,
    ) -> List[Tuple[np.ndarray, float, int]]:
        """Extract overlapping segments from signal.
        
        Args:
            from config.py
        
        Returns:
            List of (segment_signal, segment_time, segment_index) tuples
        """
        sampling_rate = 1.0 / np.mean(np.diff(time))
        window_samples = int(window_length_sec * sampling_rate)
        stride_samples = int(window_samples * (1 - overlap))
        
        segments = []
        segment_idx = 0
        
        for start_idx in range(0, len(acceleration) - window_samples, stride_samples):
            if (segment_idx % subsample_factor) == 0:
                end_idx = start_idx + window_samples
                segment_signal = acceleration[start_idx:end_idx].copy()
                segment_time = time[start_idx]
                segments.append((segment_signal, segment_time, segment_idx))
            
            segment_idx += 1
        
        return segments
    
    @staticmethod
    def extract_segments_all_sensors(
        sensor_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        window_length_sec: float,
        overlap: float = 0.9,
        subsample_factor: int = 1,
    ) -> Dict[str, List[Tuple[np.ndarray, float, int]]]:
        """Extract segments from all sensors in a scenario.
        
        Args:
            sensor_data: From SimulationLoader.load_scenario()
            window_length_sec: Window length in seconds
            overlap: Overlap ratio
            subsample_factor: Subsampling factor
        
        Returns:
            Dictionary mapping sensor name to list of segments
        """
        all_segments = {}
        
        for sensor_name, (accel, time, dist) in sensor_data.items():
            segments = SegmentationHelper.extract_segments(
                accel, time, window_length_sec, overlap, subsample_factor
            )
            all_segments[sensor_name] = segments
        
        return all_segments
