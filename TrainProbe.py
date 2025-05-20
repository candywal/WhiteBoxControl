import argparse
import os
import torch # Keep torch import for type hints and potential direct use
from Utils import load_and_prepare_hf_datasets, save_json, get_device
from probes import get_probe_class

def main():
    parser = argparse.ArgumentParser(description="Train a specified probe.")
    # Data Args
    parser.add_argument("--safe_dataset_name", required=True, type=str, help="HF name for the safe dataset.")
    parser.add_argument("--safe_dataset_config_name", type=str, default=None, help="Config for safe dataset.")
    parser.add_argument("--unsafe_dataset_name", required=True, type=str, help="HF name for the unsafe dataset.")
    parser.add_argument("--unsafe_dataset_config_name", type=str, default=None, help="Config for unsafe dataset.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use (e.g., train, test).")
    parser.add_argument("--dataset_cache_dir", type=str, default=None, help="Cache directory for Hugging Face datasets.")

    # Probe General Args
    parser.add_argument("--probe_type", required=True, choices=["logistic_probe", "mean_diff_probe", "learned_attention_probe"], help="Type of probe.")
    parser.add_argument("--base_model_name", required=True, type=str, help="HF name for the base model.")
    parser.add_argument("--layer", required=True, type=int, help="Layer from base model for activations.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the trained probe.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (ignored by mean_diff_probe).")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (ignored by mean_diff_probe).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and activation computation.")
    parser.add_argument("--l2_penalty", type=float, default=0.001, help="L2 penalty (weight decay).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "SGD"], help="Optimizer type.")
    parser.add_argument("--max_token_length", type=int, default=512, help="Max token length for tokenizer.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor.")


    # Hardware/Precision Args
    parser.add_argument("--device_str", type=str, default=None, help="Device ('cuda', 'cpu'). Auto-detected if None.")
    parser.add_argument("--dtype_str", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type for model and probe.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit via BitsAndBytes.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit via BitsAndBytes.")


    # Probe-Specific Args
    # For LogisticProbe & MeanDiffProbe
    parser.add_argument("--token_aggregation_strategy", type=str, default="last_only", choices=["mean", "soft_max", "last_only"])
    # For LearnedAttentionProbe
    parser.add_argument("--attn_use_normalization", action="store_true", help="Use LayerNorm in AttentionProbe.")
    parser.add_argument("--attn_attention_scaling", type=lambda x: (str(x).lower() == 'true'), default=True, help="Scale attention scores in AttentionProbe.")
    parser.add_argument("--attn_initialization_scale", type=float, default=1.0, help="Initial weight scale for AttentionProbe.")
    parser.add_argument("--attn_temperature", type=float, default=2.0, help="Softmax temperature for AttentionProbe.")


    args = parser.parse_args()

    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both --load_in_8bit and --load_in_4bit.")

    # Load and prepare data
    # This is done once, before probe initialization, on the main process if feasible, or by all.
    # For now, let each process load, HF datasets handles caching.
    print("Loading and preparing datasets...")
    train_messages, train_labels = load_and_prepare_hf_datasets(
        args.safe_dataset_name, args.unsafe_dataset_name,
        args.safe_dataset_config_name, args.unsafe_dataset_config_name,
        args.dataset_split, args.dataset_cache_dir
    )
    print(f"Loaded {len(train_messages)} training examples.")
    if not train_messages:
        print("No training data loaded. Exiting.")
        return

    ProbeClass = get_probe_class(args.probe_type)
    
    probe_specific_constructor_args = {}
    if args.probe_type in ["logistic_probe", "mean_diff_probe"]:
        probe_specific_constructor_args["token_position_aggregation_strategy"] = args.token_aggregation_strategy
    elif args.probe_type == "learned_attention_probe":
        probe_specific_constructor_args["use_normalization"] = args.attn_use_normalization
        probe_specific_constructor_args["attention_scaling"] = args.attn_attention_scaling
        probe_specific_constructor_args["initialization_scale"] = args.attn_initialization_scale
        probe_specific_constructor_args["temperature"] = args.attn_temperature

    print(f"Initializing probe: {args.probe_type}")
    probe = ProbeClass(
        base_model_name=args.base_model_name,
        layer=args.layer,
        probe_specific_kwargs=probe_specific_constructor_args, # This goes into BaseProbe's saved config
        # These are direct args for BaseProbe and then specific probe's __init__
        **(probe_specific_constructor_args), # Unpack for specific probe's __init__
        device_str=args.device_str,
        dtype_str=args.dtype_str,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        output_dir_for_saving=args.output_dir
    )

    print(f"Starting training for {args.probe_type}...")
    probe.train_on_data(
        train_messages=train_messages,
        train_labels=train_labels,
        # val_messages/labels can be added here if validation data is prepared
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        l2_penalty=args.l2_penalty,
        max_grad_norm=args.max_grad_norm,
        optimizer_type=args.optimizer_type,
        max_token_length=args.max_token_length,
        label_smoothing=args.label_smoothing,
    )

    # Saving is handled by the probe instance itself using accelerator.is_main_process
    probe.save_probe() # Uses output_dir_for_saving set during init

    if probe.accelerator.is_main_process:
        print(f"Probe training complete. Saved to {args.output_dir}")
        # Save training script args for reproducibility
        train_run_config = vars(args)
        save_json(train_run_config, os.path.join(args.output_dir, "train_script_args.json"))

if __name__ == "__main__":
    main()