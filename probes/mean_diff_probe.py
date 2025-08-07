# %%
"""Difference in means probe"""

import os
import time

import argparse
import logging
import torch
from torch import Tensor
from typing import List, Literal, Any, Dict
from torch import nn
import torch.distributed as dist
from .base_probe import Monitor
from transformers import AutoConfig
from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

from huggingface_hub import whoami

# whoami()

import os

# %%


# Add a simple ParameterHolder class to make self.params compatible with save_params
class ParameterHolder(nn.Module):
    def __init__(self, tensor=None, hidden_size=None, device=None, dtype=None):
        super().__init__()
        if tensor is not None:
            self.weight = nn.Parameter(tensor)
        elif hidden_size is not None:
            self.weight = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        else:
            raise ValueError("Either tensor or hidden_size must be provided")

    def forward(self, x):
        return torch.matmul(x, self.weight)


class MeanDiffProbe(Monitor):
    name = "mean_diff_probe"

    def __init__(
        self,
        model_name: str,
        layer: int,
        token_position_aggregation_strategy: Literal["mean", "soft_max", "last_only"] = "last_only",
        on_policy_model_organism: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.layer = layer
        self.token_position_aggregation_strategy = token_position_aggregation_strategy
        self.params = None
        self.class_means = {}
        self.on_policy_model_organism = on_policy_model_organism

    def init_params(self):
        if self.accelerator.is_main_process:
            print("===== init_params called =====")
        config = AutoConfig.from_pretrained(self.model_name)
        hidden_size = config.hidden_size
        if self.accelerator.is_main_process:
            print(f"Creating direction vector of size {hidden_size}")
        # For mean diff probe, we only need a direction vector
        self.params = ParameterHolder(hidden_size=hidden_size, device=self.device, dtype=self.dtype)
        if self.accelerator.is_main_process:
            print(f"Parameters have been created on device={self.device}, dtype={self.dtype}")

    def soft_max(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a differentiable approximation of the maximum using the LogSumExp trick.
        Lower temperature values make this closer to true max but less smooth.
        Higher temperature values make this smoother but more like an average.
        """
        temperature = 0.1
        # Scale inputs by temperature
        scaled = x / temperature
        # Use LogSumExp trick for numerical stability
        max_val = torch.max(scaled)  # For numerical stability
        exp_sum = torch.exp(scaled - max_val).sum()
        logsumexp = temperature * (torch.log(exp_sum) + max_val)

        n = x.numel()  # number of elements in x
        max_possible = temperature * (torch.log(torch.tensor(n)) + 1 / temperature)
        return logsumexp / max_possible

    def forward(self, trajectories, trajectory_activations, extra_info=None):
        if self.params is None:
            raise ValueError(
                "The probe parameters have not been initialized. "
                "Please call train() or init_params() before forward()."
            )

        if extra_info is not None:
            print(f"Extra info provided: {extra_info}")
            raise Exception("stop")

        predictions = torch.zeros(len(trajectories), device=self.device, dtype=self.dtype)

        # Remove tqdm and just use simple enumeration
        for i, (trajectory, activations) in enumerate(zip(trajectories, trajectory_activations)):
            # Move activations to device
            activations = activations.to(self.device)
            # Remove problematic assertion and add flexible shape validation
            if activations.dim() == 3 and activations.shape[0] == 1:
                # Standard case: (1, num_tokens, hidden_size)
                activations_2d = activations[0, :, :]
            elif activations.dim() == 2:
                # Already 2D: (num_tokens, hidden_size)
                activations_2d = activations
            else:
                raise ValueError(f"Unexpected activation shape: {activations.shape}. Expected (1, num_tokens, hidden_size) or (num_tokens, hidden_size)")

            # Project activations along the direction vector
            token_projections = torch.matmul(activations_2d, self.params.weight)

            match self.token_position_aggregation_strategy:
                case "mean":
                    prediction = token_projections.mean()
                case "soft_max":
                    prediction = self.soft_max(token_projections)
                case "last_only":
                    prediction = token_projections[-1]
                case _:
                    raise ValueError(
                        f"Unknown token_position_aggregation_strategy {self.token_position_aggregation_strategy}"
                    )

            predictions[i] = prediction

        # Apply sigmoid to ensure values are between 0 and 1
        return torch.sigmoid(predictions)

    def save_model_params(self, file_path: str):
        if self.accelerator.is_main_process:
            if self.params is not None:
                torch.save(
                    {"direction_vector": self.params.weight, "class_means": self.class_means},
                    file_path,
                )
                print(f"Model parameters saved to {file_path}")
            else:
                print("Model parameters are not initialized.")

    def compute_class_means(self, activations, labels):
        """Compute mean activation vectors for each class."""
        unique_labels = set(labels)
        class_means = {}

        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            class_activations = [activations[i] for i in indices]

            # Aggregate activations based on token strategy
            aggregated_activations = []
            for act in class_activations:
                # Handle activation shapes flexibly
                if act.dim() == 3 and act.shape[0] == 1:
                    # Standard case: (1, num_tokens, hidden_size)
                    act_2d = act[0]
                elif act.dim() == 2:
                    # Already 2D: (num_tokens, hidden_size)
                    act_2d = act
                else:
                    raise ValueError(f"Unexpected activation shape: {act.shape}. Expected (1, num_tokens, hidden_size) or (num_tokens, hidden_size)")
                    
                match self.token_position_aggregation_strategy:
                    case "mean":
                        aggregated_activations.append(act_2d.mean(dim=0))
                    case "soft_max":
                        # Apply softmax across tokens for each feature
                        weights = torch.softmax(act_2d.norm(dim=1, keepdim=True) / 0.1, dim=0)
                        aggregated_activations.append((act_2d * weights).sum(dim=0))
                    case "last_only":
                        aggregated_activations.append(act_2d[-1])

            # Compute mean activation for this class
            class_means[label] = torch.stack(aggregated_activations).mean(dim=0)

        return class_means

    @torch.enable_grad()
    def fit(
        self,
        activation_paths,
        trajectories,
        labels,
        num_epochs=1,
        **kwargs,
    ) -> float:
        if self.accelerator.is_main_process:
            print(f"Fitting mean difference probe...")

        if self.params is None:
            self.init_params()

        # Load activations
        activations = [torch.load(path, map_location=self.device) for path in activation_paths]

        # Compute class means locally on each process
        local_class_means = self.compute_class_means(activations, labels)

        # Convert to tensors for gathering
        class_keys = list(local_class_means.keys())
        if len(class_keys) != 2:
            raise ValueError(f"Expected 2 classes for mean diff probe, got {len(class_keys)}")

        # Prepare local means for gathering
        local_means_0 = local_class_means[class_keys[0]]
        local_means_1 = local_class_means[class_keys[1]]

        # Gather class means from all processes
        all_means_0 = self.accelerator.gather(local_means_0.unsqueeze(0)).cpu()
        all_means_1 = self.accelerator.gather(local_means_1.unsqueeze(0)).cpu()

        # Make sure all processes have calculated their part before proceeding
        if dist.is_initialized():
            dist.barrier()

        # Compute the global means
        global_mean_0 = torch.mean(all_means_0, dim=0).to(self.device)
        global_mean_1 = torch.mean(all_means_1, dim=0).to(self.device)

        # Update class means dictionary (do this on all processes to avoid broadcasts)
        self.class_means = {class_keys[0]: global_mean_0, class_keys[1]: global_mean_1}

        # Compute direction vector as the difference between global class means
        direction_vector = (global_mean_1 - global_mean_0).to(self.dtype)

        # Normalize the direction vector
        direction_vector = direction_vector / direction_vector.norm()

        # Store the direction vector in our parameter holder
        self.params = ParameterHolder(tensor=direction_vector, device=self.device, dtype=self.dtype)

        if self.accelerator.is_main_process:
            print(f"Direction vector computed with norm: {direction_vector.norm()}")

        return 0.0  # No loss to return for mean diff probe

    def get_activations(
        self, trajectories: List[List[Any]], layers: List[int] = []
    ) -> List[Tensor]:
        result = super().get_activations(trajectories, layers=[self.layer])
        return result


tasks = {
    "code_rule_violation",
    "code_sabotage",
    "code_deception",
    "adversarial_code_deception",
    "adversarial_code_sabotage",
    "adversarial_code_rule_violation",
    "all",
    "adversarial_all",
}


short_model_names = {
    "meta-llama/Llama-3.1-70B-Instruct": "70b",
    "meta-llama/Llama-3.2-1B-Instruct": "1b",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a difference in means probe.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--train_task", type=str)
    parser.add_argument("--eval_task", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--eval_path", type=str)
    parser.add_argument("--out_path", type=str, default="./out.pkl")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--num_train_trajectories", type=int, default=100)
    parser.add_argument("--num_eval_trajectories", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=1)  # Only need 1 epoch for mean diff
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--on_policy_model_organism", choices=[True, False], type=bool, default=False
    )
    parser.add_argument(
        "--eval_on_trainset", action="store_true", help="Whether to evaluate on training set"
    )
    args = parser.parse_args()
    args.num_epochs = 1  # hard code num epochs for mean diff probe, since it doesn't need training

    model = args.model
    assert args.train_task in tasks, f"Invalid train task: {args.train_task}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get the directory from out_path
    monitor_dir = os.path.dirname(args.out_path)

    start_time = time.time()
    mean_diff_probe = MeanDiffProbe(
        model_name=model,
        monitor_dir=monitor_dir,
        inference_batch_size_per_device=4,
        dtype=torch.bfloat16,
        layer=args.layer,
        on_policy_model_organism=args.on_policy_model_organism,
    )
    model_init_time = time.time() - start_time
    if mean_diff_probe.accelerator.is_main_process:
        print(f"Time to initialize model: {model_init_time:.2f} seconds")

    # Start timing training
    start_time = time.time()
    mean_diff_probe.train(
        args.train_path,
        clear_activation_cache=True,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )
    training_time = time.time() - start_time
    if mean_diff_probe.accelerator.is_main_process:
        print(f"Time for training: {training_time:.2f} seconds")
        mean_diff_probe.save_model_params(os.path.join(monitor_dir, "model_params.pt"))

    # Evaluate on the training set
    if args.eval_on_trainset and mean_diff_probe.accelerator.is_main_process:
        print(f"Evaluating on the training set: {args.train_path} ...")
        start_time = time.time()
        train_metrics = mean_diff_probe.evaluate(
            args.train_path,
            out_path=os.path.join(monitor_dir, "train_metrics.json"),
            clear_activation_cache=True,
        )
        train_eval_time = time.time() - start_time
        print(
            "Training set evaluation metrics:",
            train_metrics.accuracy,
            train_metrics.tpr_at_fpr,
            train_metrics.auc,
        )
        print(f"Time for training set evaluation: {train_eval_time:.2f} seconds")

    if mean_diff_probe.accelerator.is_main_process:
        print(f"Evaluating on {args.eval_path} ...")
        print(f"args.eval_path => {args.eval_path}")
        print(f"args.out_path => {args.out_path}")

    # Start timing evaluation
    start_time = time.time()
    monitor_predictions = mean_diff_probe.evaluate(
        args.eval_path,
        out_path=args.out_path,
        clear_activation_cache=True,
    )
    eval_time = time.time() - start_time

    if mean_diff_probe.accelerator.is_main_process:
        print(f"Time for evaluation: {eval_time:.2f} seconds")
        # Print summary of timing information
        print("\n===== Timing Summary =====")
        print(f"Model initialization: {model_init_time:.2f} seconds")
        print(f"Training: {training_time:.2f} seconds")
        if args.eval_on_trainset:
            print(f"Training set evaluation: {train_eval_time:.2f} seconds")
        print(f"Test set evaluation: {eval_time:.2f} seconds")
        print(f"Total time: {model_init_time + training_time + eval_time:.2f} seconds")
        if args.eval_on_trainset:
            print(
                f"Total time (including train eval): {model_init_time + training_time + train_eval_time + eval_time:.2f} seconds"
            )

    dist.destroy_process_group()

# %%
