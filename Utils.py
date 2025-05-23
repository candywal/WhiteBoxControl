#!/usr/bin/env python3
"""
Utils.py - Utility functions for WhiteBoxControl.

This module provides helper functions for:
- Downloading and processing datasets
- Model loading and validation
- Data format conversion
- Common operations
"""

import os
import json
import requests
import subprocess
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_hf_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        cache_dir: Local directory to cache the model (optional)
    
    Returns:
        str: Path to the downloaded model
    """
    print(f"📥 Downloading model: {model_name}")
    
    try:
        # Set cache directory if provided
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✓ Model downloaded successfully")
        return model_name
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}: {e}")

def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path exists and contains required files.
    
    Args:
        model_path: Path to model directory or HuggingFace model name
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not model_path:
        return False
    
    # Check if it's a HuggingFace model name
    if "/" in model_path and not os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return True
        except:
            return False
    
    # Check if it's a local path
    if os.path.exists(model_path):
        required_files = ["config.json"]
        return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
    
    return False

def load_trajectory_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load trajectory data from JSON file.
    
    Args:
        data_path: Path to JSON file containing trajectories
    
    Returns:
        List of trajectory dictionaries
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Data must be a list of trajectories")
        
        print(f"✓ Loaded {len(data)} trajectories from {data_path}")
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {data_path}: {e}")

def convert_data_format(data: List[Any], source_format: str, target_format: str) -> List[Any]:
    """
    Convert between different data formats.
    
    Args:
        data: Input data
        source_format: Source format ("trajectories", "actions", "conversations")
        target_format: Target format ("trajectories", "actions", "conversations")
    
    Returns:
        Converted data
    """
    if source_format == target_format:
        return data
    
    if source_format == "actions" and target_format == "trajectories":
        # Convert flat list of actions to trajectories
        # Group by trajectory_index if available
        trajectories = {}
        for action in data:
            traj_idx = action.get("trajectory_index", 0)
            if traj_idx not in trajectories:
                trajectories[traj_idx] = []
            trajectories[traj_idx].append(action)
        
        return list(trajectories.values())
    
    elif source_format == "trajectories" and target_format == "actions":
        # Flatten trajectories to actions
        actions = []
        for traj_idx, trajectory in enumerate(data):
            if isinstance(trajectory, list):
                for action in trajectory:
                    action["trajectory_index"] = traj_idx
                    actions.append(action)
            else:
                trajectory["trajectory_index"] = traj_idx
                actions.append(trajectory)
        
        return actions
    
    else:
        raise ValueError(f"Conversion from {source_format} to {target_format} not supported")

def combine_datasets(dataset_paths: List[str], output_path: str, 
                    labels: Optional[List[str]] = None) -> None:
    """
    Combine multiple datasets into a single file.
    
    Args:
        dataset_paths: List of paths to dataset files
        output_path: Path to save combined dataset
        labels: Optional labels to assign to each dataset
    """
    combined_data = []
    
    for i, data_path in enumerate(dataset_paths):
        print(f"📁 Loading dataset {i+1}/{len(dataset_paths)}: {data_path}")
        data = load_trajectory_data(data_path)
        
        # Add label if provided
        if labels and i < len(labels):
            for trajectory in data:
                if isinstance(trajectory, list):
                    for action in trajectory:
                        action["dataset_label"] = labels[i]
                else:
                    trajectory["dataset_label"] = labels[i]
        
        combined_data.extend(data)
    
    # Save combined dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"✓ Combined {len(combined_data)} trajectories saved to: {output_path}")

def split_dataset(data_path: str, train_ratio: float = 0.8, 
                 output_dir: str = "data/splits", seed: int = 42) -> Dict[str, str]:
    """
    Split dataset into train/test sets.
    
    Args:
        data_path: Path to dataset file
        train_ratio: Ratio of data for training (default: 0.8)
        output_dir: Directory to save splits
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with paths to train and test files
    """
    import random
    random.seed(seed)
    
    # Load data
    data = load_trajectory_data(data_path)
    
    # Shuffle and split
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    train_path = os.path.join(output_dir, f"{dataset_name}_train.json")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✓ Dataset split: {len(train_data)} train, {len(test_data)} test")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    return {"train": train_path, "test": test_path}

def add_trajectory_indices(data_path: str, output_path: Optional[str] = None) -> str:
    """
    Add trajectory_index field to all actions in a dataset.
    
    Args:
        data_path: Path to input dataset
        output_path: Path to save output (if None, overwrites input)
    
    Returns:
        Path to output file
    """
    data = load_trajectory_data(data_path)
    
    # Add trajectory indices
    for traj_idx, trajectory in enumerate(data):
        if isinstance(trajectory, list):
            for action in trajectory:
                action["trajectory_index"] = traj_idx
        else:
            trajectory["trajectory_index"] = traj_idx
    
    # Save
    if output_path is None:
        output_path = data_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Added trajectory indices to {len(data)} trajectories")
    return output_path

def validate_dataset_format(data_path: str, required_fields: Optional[List[str]] = None) -> bool:
    """
    Validate dataset format and required fields.
    
    Args:
        data_path: Path to dataset file
        required_fields: List of required fields in each action
    
    Returns:
        bool: True if valid, False otherwise
    """
    if required_fields is None:
        required_fields = ["messages", "classification"]
    
    try:
        data = load_trajectory_data(data_path)
        
        if len(data) == 0:
            print("❌ Dataset is empty")
            return False
        
        # Check first trajectory/action
        sample_trajectory = data[0]
        if isinstance(sample_trajectory, list):
            if len(sample_trajectory) == 0:
                print("❌ First trajectory is empty")
                return False
            sample_action = sample_trajectory[0]
        else:
            sample_action = sample_trajectory
        
        # Check required fields
        missing_fields = []
        for field in required_fields:
            if field not in sample_action:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        # Check classification values
        classifications = set()
        for trajectory in data:
            if isinstance(trajectory, list):
                for action in trajectory:
                    classifications.add(action.get("classification"))
            else:
                classifications.add(trajectory.get("classification"))
        
        valid_classifications = {"safe", "unsafe"}
        invalid_classifications = classifications - valid_classifications
        
        if invalid_classifications:
            print(f"❌ Invalid classification values: {invalid_classifications}")
            print(f"   Valid values are: {valid_classifications}")
            return False
        
        print(f"✓ Dataset format validation passed")
        print(f"  {len(data)} trajectories")
        print(f"  Classifications: {sorted(classifications)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False

def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpus": []
    }
    
    if torch.cuda.is_available():
        gpu_info["gpu_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info["gpus"].append({
                "id": i,
                "name": gpu_props.name,
                "memory_total": gpu_props.total_memory,
                "memory_available": torch.cuda.mem_get_info(i)[0],
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
            })
    
    return gpu_info

def setup_directories(base_dir: str = ".") -> Dict[str, str]:
    """
    Set up the standard directory structure for WhiteBoxControl.
    
    Args:
        base_dir: Base directory for the project
    
    Returns:
        Dictionary mapping directory names to paths
    """
    directories = {
        "data": os.path.join(base_dir, "data"),
        "datasets": os.path.join(base_dir, "data", "datasets"),
        "models": os.path.join(base_dir, "data", "models"),
        "outputs": os.path.join(base_dir, "outputs"),
        "trained_probes": os.path.join(base_dir, "outputs", "trained_probes"),
        "evaluation": os.path.join(base_dir, "outputs", "evaluation"),
        "run_scripts": os.path.join(base_dir, "run_scripts"),
    }
    
    # Create directories
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        print(f"📁 {name}: {path}")
    
    return directories

def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Returns:
        Dictionary mapping package names to installation status
    """
    required_packages = [
        "torch",
        "transformers",
        "accelerate", 
        "sklearn",
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "pandas"
    ]
    
    status = {}
    
    for package in required_packages:
        try:
            __import__(package)
            status[package] = True
        except ImportError:
            status[package] = False
    
    return status

def print_system_info():
    """Print comprehensive system information for debugging."""
    print("🖥️  System Information")
    print("=" * 50)
    
    # Python version
    import sys
    print(f"Python: {sys.version}")
    
    # PyTorch info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Dependencies
    print("\n📦 Dependencies:")
    deps = check_dependencies()
    for package, installed in deps.items():
        status = "✓" if installed else "❌"
        print(f"  {status} {package}")
    
    # Disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    print(f"\n💾 Disk Space:")
    print(f"  Total: {total // (1024**3)} GB")
    print(f"  Used: {used // (1024**3)} GB") 
    print(f"  Free: {free // (1024**3)} GB")

if __name__ == "__main__":
    # Demo usage
    print("🛠️  WhiteBoxControl Utilities")
    print_system_info()
    print("\n📁 Setting up directories...")
    directories = setup_directories()
    print(f"\n✓ Setup complete!") 