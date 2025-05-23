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
                       help="Path to training data (JSON file with trajectories)")
    
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

def validate_data_file(data_path: str) -> None:
    """Validate that the data file exists and has correct format."""
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
    validate_data_file(args.data)
    
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
    
    # Train probe
    print(f"\n🎯 Starting training...")
    start_time = time.time()
    
    training_kwargs = {
        "clear_activation_cache": args.clear_cache,
        "num_epochs": args.epochs,
    }
    
    if args.lr is not None:
        training_kwargs["lr"] = args.lr
    
    probe.train(args.data, **training_kwargs)
    
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

if __name__ == "__main__":
    main() 