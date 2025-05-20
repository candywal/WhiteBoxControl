import argparse
import os
from Utils import load_and_prepare_hf_datasets, save_json, load_json
from probes import get_probe_class # To get the class for loading

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained probe.")
    parser.add_argument("--probe_dir", required=True, type=str, help="Directory of the trained probe.")
    
    # Eval Data Args
    parser.add_argument("--eval_safe_dataset_name", required=True, type=str, help="HF name for safe eval data.")
    parser.add_argument("--eval_safe_dataset_config_name", type=str, default=None)
    parser.add_argument("--eval_unsafe_dataset_name", required=True, type=str, help="HF name for unsafe eval data.")
    parser.add_argument("--eval_unsafe_dataset_config_name", type=str, default=None)
    parser.add_argument("--eval_dataset_split", type=str, default="test", help="Dataset split for evaluation.")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)

    parser.add_argument("--results_output_dir", required=True, type=str, help="Directory to save evaluation metrics.")
    
    # Eval Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--max_token_length", type=int, default=512, help="Max token length for tokenizer.")
    
    # Hardware (usually inferred from probe_dir's config or auto-detected by BaseProbe)
    parser.add_argument("--device_str", type=str, default=None, help="Override device ('cuda', 'cpu').")


    args = parser.parse_args()
    os.makedirs(args.results_output_dir, exist_ok=True)

    # Load probe config to get its type, then use the class method to load
    probe_meta_config_path = os.path.join(args.probe_dir, "config.json")
    probe_meta_config = load_json(probe_meta_config_path)
    probe_type_to_load = probe_meta_config["probe_type"]
    ProbeClass = get_probe_class(probe_type_to_load)

    print(f"Loading probe of type '{probe_type_to_load}' from {args.probe_dir}...")
    # Pass device_str if specified, otherwise BaseProbe.load_probe will use its logic
    probe = ProbeClass.load_probe(probe_dir=args.probe_dir, device_str=args.device_str)
    
    print("Loading and preparing evaluation datasets...")
    eval_messages, eval_labels = load_and_prepare_hf_datasets(
        args.eval_safe_dataset_name, args.eval_unsafe_dataset_name,
        args.eval_safe_dataset_config_name, args.eval_unsafe_dataset_config_name,
        args.eval_dataset_split, args.dataset_cache_dir
    )
    print(f"Loaded {len(eval_messages)} evaluation examples.")
    if not eval_messages:
        print("No evaluation data loaded. Exiting.")
        if probe.accelerator.is_main_process: # Save empty results if main
             save_json({"error": "No evaluation data"}, os.path.join(args.results_output_dir, "evaluation_results.json"))
        return

    print(f"Starting evaluation...")
    evaluation_metrics = probe.evaluate_on_data(
        eval_messages=eval_messages,
        eval_labels=eval_labels,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length
    )

    if probe.accelerator.is_main_process:
        if evaluation_metrics:
            print("Evaluation complete. Metrics:")
            for k, v in evaluation_metrics.items():
                if isinstance(v, list) and len(v) > 10: # Don't print huge lists of labels/preds
                    print(f"  {k}: List of length {len(v)}")
                else:
                    print(f"  {k}: {v}")
            save_json(evaluation_metrics, os.path.join(args.results_output_dir, "evaluation_results.json"))
            print(f"Evaluation results saved to {args.results_output_dir}")
        else:
            print("Evaluation did not return metrics on main process (this might be expected if no eval data).")
        
        # Save eval script args
        eval_run_config = vars(args)
        save_json(eval_run_config, os.path.join(args.results_output_dir, "eval_script_args.json"))

if __name__ == "__main__":
    main()