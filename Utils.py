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
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

def download_hf_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-Coder-32B-Instruct")
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

def load_hf_dataset(dataset_name: str, 
                   subset: Optional[str] = None,
                   split: str = "train",
                   classification_mapping: Optional[Dict[str, str]] = None,
                   classification_field: str = "label",
                   messages_field: str = "messages") -> List[Dict[str, Any]]:
    """
    Load a Hugging Face dataset and convert it to WhiteBoxControl format.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "Anthropic/hh-rlhf")
        subset: Dataset subset/config name (optional)
        split: Dataset split to load (default: "train")
        classification_mapping: Mapping from dataset labels to "safe"/"unsafe"
                               If None, assumes labels are already "safe"/"unsafe"
        classification_field: Field name containing the classification (default: "label")
        messages_field: Field name containing the messages (default: "messages")
    
    Returns:
        List of trajectories in WhiteBoxControl format
    
    Example:
        # For a dataset with "safe"/"unsafe" labels
        data = load_hf_dataset("my-org/safety-dataset")
        
        # For a dataset with custom labels
        data = load_hf_dataset(
            "my-org/other-dataset",
            classification_mapping={"positive": "safe", "negative": "unsafe"}
        )
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets>=2.14.0")
    
    print(f"📥 Loading HuggingFace dataset: {dataset_name}")
    if subset:
        print(f"   Subset: {subset}")
    print(f"   Split: {split}")
    
    try:
        # Load dataset
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        trajectories = []
        
        for idx, example in enumerate(dataset):
            # Extract messages
            if messages_field not in example:
                raise ValueError(f"Messages field '{messages_field}' not found in dataset. "
                               f"Available fields: {list(example.keys())}")
            
            messages = example[messages_field]
            
            # Validate messages format
            if not isinstance(messages, list):
                raise ValueError(f"Messages field must be a list, got {type(messages)}")
            
            for message in messages:
                if not isinstance(message, dict) or "role" not in message or "content" not in message:
                    raise ValueError("Each message must be a dict with 'role' and 'content' fields")
            
            # Extract classification
            if classification_field in example:
                classification = example[classification_field]
                
                # Apply mapping if provided
                if classification_mapping:
                    if classification not in classification_mapping:
                        raise ValueError(f"Classification '{classification}' not found in mapping. "
                                       f"Available mappings: {list(classification_mapping.keys())}")
                    classification = classification_mapping[classification]
                
                # Validate classification
                if classification not in ["safe", "unsafe"]:
                    raise ValueError(f"Classification must be 'safe' or 'unsafe', got '{classification}'")
            
            else:
                # If no classification field, assume we need manual labeling
                print(f"⚠️  No classification field '{classification_field}' found. "
                      f"Available fields: {list(example.keys())}")
                print("You'll need to manually add classifications to use this dataset.")
                classification = "safe"  # Default value
            
            # Create action in WhiteBoxControl format
            action = {
                "messages": messages,
                "classification": classification,
                "trajectory_index": idx,
                "action_id": 1,
                # Add original dataset fields as extra metadata
                "hf_dataset_name": dataset_name,
                "hf_dataset_subset": subset,
                "hf_dataset_split": split,
                "hf_original_index": idx,
            }
            
            # Add any additional fields from the dataset
            for key, value in example.items():
                if key not in [messages_field, classification_field]:
                    action[f"hf_{key}"] = value
            
            # Wrap action in trajectory (each example is its own trajectory)
            trajectories.append([action])
        
        print(f"✓ Loaded {len(trajectories)} trajectories from HuggingFace dataset")
        return trajectories
        
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace dataset {dataset_name}: {e}")

def load_trajectory_data(data_path: str, 
                        hf_dataset_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Load trajectory data from JSON file or HuggingFace dataset.
    
    Args:
        data_path: Path to JSON file OR HuggingFace dataset name (if hf_dataset_config provided)
        hf_dataset_config: Configuration for loading HuggingFace dataset
                          Example: {
                              "subset": "helpful-base", 
                              "split": "train",
                              "classification_mapping": {"chosen": "safe", "rejected": "unsafe"},
                              "classification_field": "preference",
                              "messages_field": "conversations"
                          }
    
    Returns:
        List of trajectory dictionaries
    """
    # If HuggingFace config is provided, load from HF
    if hf_dataset_config is not None:
        return load_hf_dataset(data_path, **hf_dataset_config)
    
    # Check if it looks like a HuggingFace dataset name
    if "/" in data_path and not os.path.exists(data_path):
        print(f"🤔 '{data_path}' looks like a HuggingFace dataset name but no config provided.")
        print("   To load from HuggingFace, use the --hf_dataset flag with TrainProbe.py")
        print("   or provide hf_dataset_config to this function.")
        print("   Treating as local file path...")
    
    # Load from local JSON file
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

def setup_gpu(gpu_id: Optional[int]) -> None:
    """
    Setup GPU configuration.
    
    Args:
        gpu_id: Specific GPU ID to use (0-7), if None uses all available
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔧 Using GPU {gpu_id}")
    else:
        print(f"🔧 Using all available GPUs")

def validate_data_file_basic(data_path: str) -> int:
    """
    Basic validation of JSON data file.
    
    Args:
        data_path: Path to JSON data file
        
    Returns:
        int: Number of trajectories in the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Data file must contain a list of trajectories")
        
        if len(data) == 0:
            raise ValueError("Data file is empty")
        
        # Check first trajectory structure
        if isinstance(data[0], list):
            # Format: [[action1, action2, ...], [action1, action2, ...], ...]
            sample_action = data[0][0] if data[0] else {}
        else:
            # Format: [action1, action2, ...]
            sample_action = data[0]
        
        required_fields = ["messages", "classification"]
        for field in required_fields:
            if field not in sample_action:
                raise ValueError(f"Action missing required field: {field}")
        
        print(f"✓ Data validation passed: {len(data)} trajectories loaded")
        return len(data)
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in data file")

def validate_hf_dataset(data_path: str, args: Any) -> int:
    """
    Validate HuggingFace dataset.
    
    Args:
        data_path: HuggingFace dataset name
        args: Arguments containing HF dataset configuration
        
    Returns:
        int: Number of examples in the dataset
        
    Raises:
        ImportError: If datasets library not available
        RuntimeError: If dataset validation fails
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets>=2.14.0")
    
    print(f"🔍 Validating HuggingFace dataset: {data_path}")
    
    # Parse classification mapping if provided
    classification_mapping = None
    if hasattr(args, 'hf_classification_mapping') and args.hf_classification_mapping:
        try:
            classification_mapping = json.loads(args.hf_classification_mapping)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in classification mapping: {args.hf_classification_mapping}")
    
    try:
        # Try to load a small sample to validate
        split = getattr(args, 'hf_split', 'train')
        subset = getattr(args, 'hf_subset', None)
        
        if subset:
            dataset = load_dataset(data_path, subset, split=f"{split}[:5]")
        else:
            dataset = load_dataset(data_path, split=f"{split}[:5]")
        
        if len(dataset) == 0:
            raise ValueError(f"Dataset split '{split}' is empty")
        
        # Check if required fields exist
        messages_field = getattr(args, 'hf_messages_field', 'messages')
        sample = dataset[0]
        
        if messages_field not in sample:
            raise ValueError(f"Messages field '{messages_field}' not found. Available fields: {list(sample.keys())}")
        
        # Check messages format
        messages = sample[messages_field]
        if not isinstance(messages, list):
            raise ValueError(f"Messages field must be a list, got {type(messages)}")
        
        if len(messages) > 0:
            if not isinstance(messages[0], dict) or "role" not in messages[0] or "content" not in messages[0]:
                raise ValueError("Messages must be a list of dicts with 'role' and 'content' fields")
        
        # Get the full dataset size
        if subset:
            full_dataset = load_dataset(data_path, subset, split=split)
        else:
            full_dataset = load_dataset(data_path, split=split)
        
        print(f"✓ HuggingFace dataset validation passed")
        print(f"  Dataset: {data_path}")
        if subset:
            print(f"  Subset: {subset}")
        print(f"  Split: {split}")
        print(f"  Messages field: {messages_field}")
        
        return len(full_dataset)
        
    except Exception as e:
        raise RuntimeError(f"Failed to validate HuggingFace dataset {data_path}: {e}")

def prepare_hf_dataset_for_training(data_path: str, args: Any) -> str:
    """
    Load HuggingFace dataset and convert to temporary JSON file for training/evaluation.
    
    Args:
        data_path: HuggingFace dataset name
        args: Arguments containing HF dataset configuration
        
    Returns:
        str: Path to temporary JSON file
    """
    import tempfile
    
    print(f"\n🔄 Loading HuggingFace dataset and converting to training format...")
    
    # Parse classification mapping if provided
    classification_mapping = None
    if hasattr(args, 'hf_classification_mapping') and args.hf_classification_mapping:
        try:
            classification_mapping = json.loads(args.hf_classification_mapping)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in classification mapping: {args.hf_classification_mapping}")
    
    # Load HuggingFace dataset
    trajectories = load_hf_dataset(
        dataset_name=data_path,
        subset=getattr(args, 'hf_subset', None),
        split=getattr(args, 'hf_split', 'train'),
        classification_mapping=classification_mapping,
        classification_field=getattr(args, 'hf_classification_field', 'label'),
        messages_field=getattr(args, 'hf_messages_field', 'messages')
    )
    
    # Create temporary JSON file
    temp_json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(trajectories, temp_json_file, indent=2)
    temp_json_file.close()
    
    print(f"✓ HuggingFace dataset converted and saved to temporary file")
    return temp_json_file.name

if __name__ == "__main__":
    # Demo usage
    print("🛠️  WhiteBoxControl Utilities")
    print_system_info()
    print("\n📁 Setting up directories...")
    directories = setup_directories()
    print(f"\n✓ Setup complete!") 