#!/usr/bin/env python3
"""
EvalProbe.py - Main script for evaluating trained neural probes.

This script loads a trained probe and evaluates it on test data, providing
comprehensive metrics and visualizations.

Usage:
    python EvalProbe.py --probe_dir outputs/trained_probes/logistic_probe_layer15 --data data/test.json
    python EvalProbe.py --probe_dir outputs/trained_probes/attention_probe_layer15 --data data/test.json --save_predictions
"""

import os
import sys
import argparse
import json
import time
import pickle
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
        description="Evaluate trained neural probes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate logistic probe
  python EvalProbe.py --probe_dir outputs/trained_probes/logistic_probe_layer15 --data data/test.json

  # Evaluate attention probe and save predictions
  python EvalProbe.py --probe_dir outputs/trained_probes/attention_probe_layer15 --data data/test.json \\
                      --save_predictions --output_dir outputs/evaluation/my_experiment

  # Evaluate probe on multiple test sets
  python EvalProbe.py --probe_dir outputs/trained_probes/mean_diff_probe_layer15 \\
                      --data data/test1.json data/test2.json --batch_eval
        """
    )
    
    # Required arguments
    parser.add_argument("--probe_dir", type=str, required=True,
                       help="Directory containing trained probe (with config.json and weights.pt)")
    parser.add_argument("--data", type=str, nargs="+", required=True,
                       help="Path(s) to test data (JSON file(s) with trajectories) OR HuggingFace dataset name(s)")
    
    # HuggingFace dataset arguments
    parser.add_argument("--hf_dataset", action="store_true",
                       help="Load data from HuggingFace dataset instead of local JSON file")
    parser.add_argument("--hf_subset", type=str, default=None,
                       help="HuggingFace dataset subset/config name")
    parser.add_argument("--hf_split", type=str, default="test",
                       help="HuggingFace dataset split to load (default: test)")
    parser.add_argument("--hf_messages_field", type=str, default="messages",
                       help="Field name containing messages in HF dataset (default: messages)")
    parser.add_argument("--hf_classification_field", type=str, default="label",
                       help="Field name containing classification in HF dataset (default: label)")
    parser.add_argument("--hf_classification_mapping", type=str, default=None,
                       help="JSON string mapping HF labels to safe/unsafe (e.g., '{\"positive\": \"safe\", \"negative\": \"unsafe\"}')")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for evaluation results (default: probe_dir/evaluation)")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="Specific GPU to use (0-7), if None uses all available")
    
    # Output options
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save detailed predictions to CSV file")
    parser.add_argument("--save_plots", action="store_true",
                       help="Generate and save evaluation plots")
    parser.add_argument("--batch_eval", action="store_true",
                       help="Evaluate on multiple datasets and compare")
    
    # Evaluation options
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear activation cache before evaluation")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()

def load_probe_config(probe_dir: str) -> Dict[str, Any]:
    """Load probe configuration from config.json."""
    config_path = os.path.join(probe_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_fields = ["probe_type", "model", "layer"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Configuration missing required field: {field}")
    
    return config

def setup_gpu(gpu_id: Optional[int]) -> None:
    """Setup GPU configuration."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔧 Using GPU {gpu_id}")
    else:
        print(f"🔧 Using all available GPUs")

def validate_data_file(data_path: str, args: argparse.Namespace) -> int:
    """Validate data file and return number of trajectories."""
    
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
            
            # Get the full dataset size
            if args.hf_subset:
                full_dataset = load_dataset(data_path, args.hf_subset, split=args.hf_split)
            else:
                full_dataset = load_dataset(data_path, split=args.hf_split)
            
            print(f"✓ HuggingFace dataset validation passed: {len(full_dataset)} examples")
            return len(full_dataset)
            
        except ImportError:
            raise ImportError("datasets library not available. Install with: pip install datasets>=2.14.0")
        except Exception as e:
            raise RuntimeError(f"Failed to validate HuggingFace dataset {data_path}: {e}")
    
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
        
        return len(data)
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in data file")

def load_trained_probe(probe_dir: str, config: Dict[str, Any]) -> Any:
    """Load the trained probe based on configuration."""
    
    # Common arguments for all probes
    common_args = {
        "model_name": config["model"],
        "monitor_dir": probe_dir,
        "inference_batch_size_per_device": config.get("batch_size", 1),
        "dtype": torch.bfloat16,
        "layer": config["layer"],
    }
    
    probe_type = config["probe_type"]
    
    if probe_type == "logistic":
        probe = LogisticProbe(
            **common_args,
            fit_method=config.get("fit_method", "adam"),
            token_position_aggregation_strategy=config.get("token_aggregation", "mean"),
        )
        
    elif probe_type == "attention":
        probe = LearnedAttentionProbe(
            **common_args,
            use_normalization=config.get("use_normalization", False),
            attention_scaling=config.get("attention_scaling", True),
            initialization_scale=config.get("initialization_scale", 2.0),
        )
        
    elif probe_type == "mean_diff":
        probe = MeanDiffProbe(
            **common_args,
            token_position_aggregation_strategy=config.get("token_aggregation", "mean"),
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    # Load trained parameters if they exist
    weights_path = os.path.join(probe_dir, "weights.pt")
    if os.path.exists(weights_path) and probe_type != "mean_diff":
        probe.load_model_params(weights_path)
        print(f"✓ Loaded trained weights from: {weights_path}")
    
    return probe

def save_predictions_csv(predictions, labels, output_path: str) -> None:
    """Save predictions to CSV file."""
    import pandas as pd
    
    df = pd.DataFrame({
        'predicted_probability': predictions.predicted_probabilities,
        'true_label': labels,
        'predicted_label': (predictions.predicted_probabilities > 0.5).astype(int),
        'correct': (predictions.predicted_probabilities > 0.5).astype(int) == labels
    })
    
    df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to: {output_path}")

def generate_evaluation_plots(predictions, output_dir: str) -> None:
    """Generate evaluation plots."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve
        import numpy as np
        
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(predictions.labels, predictions.predicted_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {predictions.auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(predictions.labels, predictions.predicted_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction histogram
        plt.figure(figsize=(10, 6))
        safe_probs = predictions.predicted_probabilities[predictions.labels == 0]
        unsafe_probs = predictions.predicted_probabilities[predictions.labels == 1]
        
        plt.hist(safe_probs, bins=30, alpha=0.7, label='Safe', density=True)
        plt.hist(unsafe_probs, bins=30, alpha=0.7, label='Unsafe', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "prediction_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plots saved to: {os.path.join(output_dir, 'plots')}")
        
    except ImportError:
        print("⚠️  matplotlib not available, skipping plot generation")

def print_evaluation_metrics(predictions, dataset_name: str = "Test") -> None:
    """Print comprehensive evaluation metrics."""
    print(f"\n📊 {dataset_name} Set Evaluation Results:")
    print("=" * 50)
    print(f"📈 Overall Performance:")
    print(f"  Accuracy: {predictions.accuracy:.4f}")
    print(f"  AUC: {predictions.auc:.4f}")
    
    if hasattr(predictions, 'cross_entropy_loss') and predictions.cross_entropy_loss is not None:
        print(f"  Cross-Entropy Loss: {predictions.cross_entropy_loss:.4f}")
    
    if hasattr(predictions, 'tpr_at_fpr') and predictions.tpr_at_fpr:
        print(f"\n🎯 True Positive Rate at Fixed False Positive Rates:")
        for fpr, tpr in sorted(predictions.tpr_at_fpr.items()):
            print(f"  TPR @ FPR={fpr}: {tpr:.4f}")
    
    # Classification report
    y_pred = (predictions.predicted_probabilities > 0.5).astype(int)
    y_true = predictions.labels
    
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=['Safe', 'Unsafe'], output_dict=True)
    
    print(f"\n📋 Classification Report:")
    print(f"  Safe    - Precision: {report['Safe']['precision']:.4f}, Recall: {report['Safe']['recall']:.4f}, F1: {report['Safe']['f1-score']:.4f}")
    print(f"  Unsafe  - Precision: {report['Unsafe']['precision']:.4f}, Recall: {report['Unsafe']['recall']:.4f}, F1: {report['Unsafe']['f1-score']:.4f}")

def evaluate_single_dataset(probe, data_path: str, output_dir: str, args: argparse.Namespace) -> Any:
    """Evaluate probe on a single dataset."""
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"\n🔍 Evaluating on: {dataset_name}")
    
    num_trajectories = validate_data_file(data_path, args)
    print(f"✓ Data validation passed: {num_trajectories} trajectories")
    
    # Prepare data path for evaluation
    data_path_for_evaluation = data_path
    temp_json_file = None
    
    # If using HuggingFace dataset, load and convert to JSON
    if args.hf_dataset:
        import tempfile
        from Utils import load_hf_dataset
        
        print(f"🔄 Loading HuggingFace dataset and converting to evaluation format...")
        
        # Parse classification mapping if provided
        classification_mapping = None
        if args.hf_classification_mapping:
            try:
                classification_mapping = json.loads(args.hf_classification_mapping)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in classification mapping: {args.hf_classification_mapping}")
        
        # Load HuggingFace dataset
        trajectories = load_hf_dataset(
            dataset_name=data_path,
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
        
        data_path_for_evaluation = temp_json_file.name
        print(f"✓ HuggingFace dataset converted and saved to temporary file")
    
    try:
        # Run evaluation
        results_path = os.path.join(output_dir, f"{dataset_name}_results.pkl")
        start_time = time.time()
        
        predictions = probe.evaluate(
            data_path_for_evaluation,
            out_path=results_path,
            clear_activation_cache=args.clear_cache,
        )
        
        eval_time = time.time() - start_time
        print(f"✓ Evaluation completed in {eval_time:.2f} seconds")
        
        # Print metrics
        print_evaluation_metrics(predictions, dataset_name)
        
        # Save additional outputs
        if args.save_predictions:
            csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
            save_predictions_csv(predictions, predictions.labels, csv_path)
        
        if args.save_plots:
            generate_evaluation_plots(predictions, output_dir)
        
        return predictions
    
    finally:
        # Clean up temporary file if created
        if temp_json_file is not None:
            try:
                os.unlink(temp_json_file.name)
                print(f"🗑️  Cleaned up temporary file")
            except OSError:
                pass  # File may already be deleted

def main():
    """Main evaluation function."""
    print("🔍 WhiteBoxControl - Neural Probe Evaluation")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Setup
    setup_gpu(args.gpu_id)
    
    # Load probe configuration
    print(f"\n📋 Loading probe configuration...")
    config = load_probe_config(args.probe_dir)
    
    print(f"  Probe Type: {config['probe_type']}")
    print(f"  Model: {config['model']}")
    print(f"  Layer: {config['layer']}")
    print(f"  Trained: {config.get('timestamp', 'Unknown')}")
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.probe_dir, "evaluation")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output Directory: {output_dir}")
    
    # Load trained probe
    print(f"\n🔧 Loading trained probe...")
    start_time = time.time()
    
    probe = load_trained_probe(args.probe_dir, config)
    
    load_time = time.time() - start_time
    print(f"✓ Probe loaded in {load_time:.2f} seconds")
    
    # Evaluate on dataset(s)
    all_results = []
    
    for data_path in args.data:
        result = evaluate_single_dataset(probe, data_path, output_dir, args)
        all_results.append((os.path.basename(data_path), result))
    
    # Batch evaluation comparison
    if args.batch_eval and len(args.data) > 1:
        print(f"\n📊 Batch Evaluation Comparison:")
        print("=" * 70)
        print(f"{'Dataset':<20} {'Accuracy':<10} {'AUC':<8} {'TPR@0.01':<10} {'TPR@0.05':<10}")
        print("-" * 70)
        
        for dataset_name, result in all_results:
            tpr_01 = result.tpr_at_fpr.get(0.01, 0) if hasattr(result, 'tpr_at_fpr') else 0
            tpr_05 = result.tpr_at_fpr.get(0.05, 0) if hasattr(result, 'tpr_at_fpr') else 0
            print(f"{dataset_name:<20} {result.accuracy:<10.4f} {result.auc:<8.4f} {tpr_01:<10.4f} {tpr_05:<10.4f}")
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"⏱️  Total time: {load_time + sum([0] * len(args.data)):.2f} seconds")

if __name__ == "__main__":
    main() 