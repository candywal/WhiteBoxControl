#!/usr/bin/env python3
"""
TrainProbe.py - Main script for training neural probes on model activations.

This script provides a unified interface for training different types of probes:
- Logistic Probe: Standard logistic regression probe
- Attention Probe: Probe with learned attention mechanism
- Mean Difference Probe: Non-parametric probe based on class mean differences

Usage:
    python TrainProbe.py --probe_type logistic --model /path/to/model --data /path/to/data.json
    python TrainProbe.py --probe_type attention --model /path/to/model --data /path/to/data.json --lr 0.002
    python TrainProbe.py --probe_type mean_diff --model /path/to/model --data /path/to/data.json
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add probes directory to path
sys.path.append(str(Path(__file__).parent / "probes"))

from probes.logistic_probe import LogisticProbe
from probes.attention_probe import LearnedAttentionProbe
from probes.mean_diff_probe import MeanDiffProbe
import torch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural probes on model activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train logistic probe
  python TrainProbe.py --probe_type logistic --model meta-llama/Llama-3.1-8B-Instruct --data data/train.json

  # Train attention probe with custom hyperparameters
  python TrainProbe.py --probe_type attention --model /path/to/model --data data/train.json \\
                       --lr 0.002 --epochs 15 --use_normalization

  # Train mean difference probe (no training required)
  python TrainProbe.py --probe_type mean_diff --model /path/to/model --data data/train.json
        """
    )
    
    # Required arguments
    parser.add_argument("--probe_type", type=str, required=True, 
                       choices=["logistic", "attention", "mean_diff"],
                       help="Type of probe to train")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path (HuggingFace model or local path)")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data (JSON file with trajectories) OR HuggingFace dataset name")
    
    # HuggingFace dataset arguments
    parser.add_argument("--hf_dataset", action="store_true",
                       help="Load data from HuggingFace dataset instead of local JSON file")
    parser.add_argument("--hf_subset", type=str, default=None,
                       help="HuggingFace dataset subset/config name")
    parser.add_argument("--hf_split", type=str, default="train",
                       help="HuggingFace dataset split to load (default: train)")
    parser.add_argument("--hf_messages_field", type=str, default="messages",
                       help="Field name containing messages in HF dataset (default: messages)")
    parser.add_argument("--hf_classification_field", type=str, default="label",
                       help="Field name containing classification in HF dataset (default: label)")
    parser.add_argument("--hf_classification_mapping", type=str, default=None,
                       help="JSON string mapping HF labels to safe/unsafe (e.g., '{\"positive\": \"safe\", \"negative\": \"unsafe\"}')")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="outputs/trained_probes",
                       help="Output directory for trained probe")
    parser.add_argument("--layer", type=int, default=15,
                       help="Transformer layer to extract activations from")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="Specific GPU to use (0-7), if None uses all available")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (if None, uses probe-specific defaults)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    
    # Attention probe specific arguments
    parser.add_argument("--use_normalization", action="store_true",
                       help="Use layer normalization (attention probe)")
    parser.add_argument("--attention_scaling", action="store_true", default=True,
                       help="Scale attention by sqrt(hidden_dim) (attention probe)")
    parser.add_argument("--initialization_scale", type=float, default=2.0,
                       help="Weight initialization scale (attention probe)")
    
    # Logistic probe specific arguments
    parser.add_argument("--fit_method", type=str, default="adam", choices=["adam", "lbfgs"],
                       help="Optimization method (logistic probe)")
    parser.add_argument("--token_aggregation", type=str, default="mean", 
                       choices=["mean", "soft_max", "last_only"],
                       help="Token position aggregation strategy")
    
    # Other options
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear activation cache before training")
    parser.add_argument("--save_config", action="store_true", default=True,
                       help="Save training configuration")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()

def setup_gpu(gpu_id: Optional[int]) -> None:
    """Setup GPU configuration."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔧 Using GPU {gpu_id}")
    else:
        print(f"🔧 Using all available GPUs")

def validate_data_file(data_path: str, args: argparse.Namespace) -> None:
    """Validate that the data file exists and has correct format, or that HF dataset is accessible."""
    
    # If loading from HuggingFace dataset
    if args.hf_dataset:
        try:
            # Import here to avoid dependency issues if not using HF datasets
            from datasets import load_dataset
            
            print(f"🔍 Validating HuggingFace dataset: {data_path}")
            
            # Parse classification mapping if provided
            classification_mapping = None
            if args.hf_classification_mapping:
                try:
                    classification_mapping = json.loads(args.hf_classification_mapping)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in classification mapping: {args.hf_classification_mapping}")
            
            # Create HF dataset config
            hf_config = {
                "subset": args.hf_subset,
                "split": args.hf_split,
                "classification_mapping": classification_mapping,
                "classification_field": args.hf_classification_field,
                "messages_field": args.hf_messages_field,
            }
            
            # Try to load a small sample to validate
            if args.hf_subset:
                dataset = load_dataset(data_path, args.hf_subset, split=f"{args.hf_split}[:5]")
            else:
                dataset = load_dataset(data_path, split=f"{args.hf_split}[:5]")
            
            if len(dataset) == 0:
                raise ValueError(f"Dataset split '{args.hf_split}' is empty")
            
            # Check if required fields exist
            sample = dataset[0]
            if args.hf_messages_field not in sample:
                raise ValueError(f"Messages field '{args.hf_messages_field}' not found. Available fields: {list(sample.keys())}")
            
            # Check messages format
            messages = sample[args.hf_messages_field]
            if not isinstance(messages, list):
                raise ValueError(f"Messages field must be a list, got {type(messages)}")
            
            if len(messages) > 0:
                if not isinstance(messages[0], dict) or "role" not in messages[0] or "content" not in messages[0]:
                    raise ValueError("Messages must be a list of dicts with 'role' and 'content' fields")
            
            print(f"✓ HuggingFace dataset validation passed")
            print(f"  Dataset: {data_path}")
            if args.hf_subset:
                print(f"  Subset: {args.hf_subset}")
            print(f"  Split: {args.hf_split}")
            print(f"  Messages field: {args.hf_messages_field}")
            print(f"  Classification field: {args.hf_classification_field}")
            if classification_mapping:
                print(f"  Classification mapping: {classification_mapping}")
            
        except ImportError:
            raise ImportError("datasets library not available. Install with: pip install datasets>=2.14.0")
        except Exception as e:
            raise RuntimeError(f"Failed to validate HuggingFace dataset {data_path}: {e}")
        
        return
    
    # Original JSON file validation
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
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in data file")

def create_probe(probe_type: str, args: argparse.Namespace) -> Any:
    """Create and configure the specified probe type."""
    
    # Create output directory
    probe_output_dir = os.path.join(args.output_dir, f"{probe_type}_probe_layer{args.layer}")
    os.makedirs(probe_output_dir, exist_ok=True)
    
    # Common arguments for all probes
    common_args = {
        "model_name": args.model,
        "monitor_dir": probe_output_dir,
        "inference_batch_size_per_device": args.batch_size,
        "dtype": torch.bfloat16,
        "layer": args.layer,
    }
    
    if probe_type == "logistic":
        probe = LogisticProbe(
            **common_args,
            fit_method=args.fit_method,
            token_position_aggregation_strategy=args.token_aggregation,
        )
        
    elif probe_type == "attention":
        probe = LearnedAttentionProbe(
            **common_args,
            use_normalization=args.use_normalization,
            attention_scaling=args.attention_scaling,
            initialization_scale=args.initialization_scale,
        )
        
    elif probe_type == "mean_diff":
        probe = MeanDiffProbe(
            **common_args,
            token_position_aggregation_strategy=args.token_aggregation,
        )
    
    return probe, probe_output_dir

def save_training_config(args: argparse.Namespace, output_dir: str) -> None:
    """Save training configuration to JSON file."""
    config = {
        "probe_type": args.probe_type,
        "model": args.model,
        "data": args.data,
        "layer": args.layer,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Add HuggingFace dataset config if used
    if args.hf_dataset:
        config.update({
            "hf_dataset": True,
            "hf_subset": args.hf_subset,
            "hf_split": args.hf_split,
            "hf_messages_field": args.hf_messages_field,
            "hf_classification_field": args.hf_classification_field,
            "hf_classification_mapping": args.hf_classification_mapping,
        })
    
    # Add probe-specific config
    if args.probe_type == "attention":
        config.update({
            "use_normalization": args.use_normalization,
            "attention_scaling": args.attention_scaling,
            "initialization_scale": args.initialization_scale,
        })
    elif args.probe_type == "logistic":
        config.update({
            "fit_method": args.fit_method,
            "token_aggregation": args.token_aggregation,
        })
    elif args.probe_type == "mean_diff":
        config.update({
            "token_aggregation": args.token_aggregation,
        })
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Training configuration saved: {config_path}")

def main():
    """Main training function."""
    print("🚀 WhiteBoxControl - Neural Probe Training")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Setup
    setup_gpu(args.gpu_id)
    validate_data_file(args.data, args)
    
    print(f"\n📋 Training Configuration:")
    print(f"  Probe Type: {args.probe_type}")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Layer: {args.layer}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr or 'auto'}")
    print(f"  Output: {args.output_dir}")
    
    # Create probe
    print(f"\n🔧 Initializing {args.probe_type} probe...")
    start_time = time.time()
    
    probe, output_dir = create_probe(args.probe_type, args)
    
    init_time = time.time() - start_time
    print(f"✓ Probe initialized in {init_time:.2f} seconds")
    
    # Save configuration
    if args.save_config:
        save_training_config(args, output_dir)
    
    # Prepare data path for training
    data_path_for_training = args.data
    temp_json_file = None
    
    # If using HuggingFace dataset, load and convert to JSON
    if args.hf_dataset:
        import tempfile
        from Utils import load_hf_dataset
        
        print(f"\n🔄 Loading HuggingFace dataset and converting to training format...")
        
        # Parse classification mapping if provided
        classification_mapping = None
        if args.hf_classification_mapping:
            try:
                classification_mapping = json.loads(args.hf_classification_mapping)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in classification mapping: {args.hf_classification_mapping}")
        
        # Load HuggingFace dataset
        trajectories = load_hf_dataset(
            dataset_name=args.data,
            subset=args.hf_subset,
            split=args.hf_split,
            classification_mapping=classification_mapping,
            classification_field=args.hf_classification_field,
            messages_field=args.hf_messages_field
        )
        
        # Create temporary JSON file
        temp_json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(trajectories, temp_json_file, indent=2)
        temp_json_file.close()
        
        data_path_for_training = temp_json_file.name
        print(f"✓ HuggingFace dataset converted and saved to temporary file: {data_path_for_training}")
    
    try:
        # Train probe
        print(f"\n🎯 Starting training...")
        start_time = time.time()
        
        training_kwargs = {
            "clear_activation_cache": args.clear_cache,
            "num_epochs": args.epochs,
        }
        
        if args.lr is not None:
            training_kwargs["lr"] = args.lr
        
        probe.train(data_path_for_training, **training_kwargs)
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Save model parameters
        model_path = os.path.join(output_dir, "weights.pt")
        probe.save_model_params(model_path)
        print(f"💾 Model weights saved: {model_path}")
        
        print(f"\n🎉 Training Complete!")
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"⏱️  Total time: {init_time + training_time:.2f} seconds")
        print(f"\n🔍 Next steps:")
        print(f"  Evaluate: python EvalProbe.py --probe_dir {output_dir} --data /path/to/test.json")
    
    finally:
        # Clean up temporary file if created
        if temp_json_file is not None:
            try:
                os.unlink(temp_json_file.name)
                print(f"🗑️  Cleaned up temporary file: {temp_json_file.name}")
            except OSError:
                pass  # File may already be deleted

if __name__ == "__main__":
    main() 