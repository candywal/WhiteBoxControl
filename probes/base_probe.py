import copy
import json
import logging
import math
import os
import pickle
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

# Make bitsandbytes optional
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
    
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.state import DistributedType
from accelerate.utils import gather_object
from scipy.interpolate import interp1d
from sklearn.metrics import auc, roc_curve
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_types import Action
from .metrics import EvaluationMetrics

logging.getLogger("deepspeed").setLevel(logging.ERROR)
"""Base class for all monitoring methods."""


@dataclass
class MonitorPredictions:
    labels: List[int]
    predicted_probabilities: List[float]
    auc: float = None
    cross_entropy_loss: float = None
    tpr_at_fpr: dict = None
    accuracy: float = None

    def __init__(self, labels: List[int], predicted_probabilities: List[float]):
        self.labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.predicted_probabilities = (
            torch.tensor(predicted_probabilities)
            if not isinstance(predicted_probabilities, torch.Tensor)
            else predicted_probabilities
        )
        self.compute_additional_metrics()

    @staticmethod
    def from_serializable(serializable):
        return MonitorPredictions(
            torch.tensor(serializable["labels"]),
            torch.tensor(serializable["predicted_probabilities"]),
        )

    def compute_additional_metrics(self):
        self.predicted_labels = [1 if p > 0.5 else 0 for p in self.predicted_probabilities]
        self.accuracy = sum(
            [
                1
                for label, prediction in zip(self.labels, self.predicted_labels)
                if label == prediction
            ]
        ) / len(self.labels)
        self.cross_entropy_loss = round(
            torch.nn.functional.binary_cross_entropy(
                self.predicted_probabilities, self.labels
            ).item(),
            3,
        )
        _, _, self.tpr_at_fpr, self.tpr_at_fpr_with_std = self.get_tpr_at_fpr_thresholds(
            self.labels, self.predicted_probabilities
        )
        self.tpr_at_fpr = {k: round(v, 3) for k, v in self.tpr_at_fpr.items()}
        self.auc = round(
            float(
                auc(*self.get_tpr_at_fpr_thresholds(self.labels, self.predicted_probabilities)[:2])
            ),
            3,
        )

    def get_aggregatable_metrics(self) -> dict:
        """Returns a dictionary of metrics that can be aggregated across experiments"""
        metrics = {
            "accuracy": round(self.accuracy, 3),
            "auc": round(self.auc, 3),
            "cross_entropy_loss": round(self.cross_entropy_loss, 3),
            "tpr_at_fpr": self.tpr_at_fpr,
            "tpr_at_fpr_with_std": {
                k: {"value": round(v["value"], 3), "std": round(v["std"], 3)}
                for k, v in self.tpr_at_fpr_with_std.items()
            },
            "num_eval_examples": len(self.labels),
        }
        return metrics

    def to_serializable(self):
        """
        Converts to a form that can be serialized to JSON.
        """
        d = asdict(self)
        return {
            k: v.tolist() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v
            for k, v in d.items()
        }

    def get_tpr_at_fpr_thresholds(
        self, labels, predicted_probabilities, target_fprs=[0.01, 0.05, 0.1, 0.2]
    ):
        """
        Calculate TPR at specific FPR thresholds using sklearn.metrics.roc_curve for robustness.
        Finds the operating point on the ROC curve with the lowest threshold
        such that the FPR meets or exceeds the target FPR. Calculates standard error
        for the TPR at that point based on binomial distribution.

        Args:
            labels: Ground truth binary labels (0 = safe, 1 = unsafe)
            predicted_probabilities: Predicted probabilities of being unsafe
            target_fprs: Target FPR thresholds at which to compute TPR

        Returns:
            fpr_full: Array of FPR values from the full ROC curve
            tpr_full: Array of TPR values from the full ROC curve
            tpr_at_fpr: Dictionary mapping each target FPR to its corresponding TPR value
            tpr_at_fpr_with_std: Dictionary mapping each target FPR to a dict with TPR value and standard error
        """
        # Convert to numpy arrays if needed
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        if isinstance(predicted_probabilities, torch.Tensor):
            predicted_probabilities_np = predicted_probabilities.cpu().numpy()
        else:
            predicted_probabilities_np = np.array(predicted_probabilities)

        # Add small noise to predictions that are exactly 0 or 1
        # This prevents issues with thresholds at the extremes
        epsilon = 1e-5
        mask_ones = predicted_probabilities_np >= 1.0
        mask_zeros = predicted_probabilities_np <= 0.0

        # Modify values that are exactly 0 or 1
        if np.any(mask_ones):
            predicted_probabilities_np[mask_ones] = 1.0 - epsilon
        if np.any(mask_zeros):
            predicted_probabilities_np[mask_zeros] = epsilon

        positives = np.sum(labels_np == 1)
        negatives = np.sum(labels_np == 0)

        # Handle edge cases
        if positives == 0 or negatives == 0:
            tpr_at_target_fpr = {fpr_target: 0.0 for fpr_target in target_fprs}
            tpr_at_target_fpr_with_std = {
                fpr_target: {"value": 0.0, "std": 0.0} for fpr_target in target_fprs
            }
            return (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                tpr_at_target_fpr,
                tpr_at_target_fpr_with_std,
            )

        # Calculate ROC curve
        fpr_full, tpr_full, thresholds_full = roc_curve(labels_np, predicted_probabilities_np)

        tpr_at_target_fpr_info = {}

        # Set different thresholds for each target FPR to ensure different TPR values
        for i, target_fpr in enumerate(target_fprs):
            # Find indices where FPR exceeds target
            indices = np.where(fpr_full >= target_fpr)[0]

            if len(indices) == 0:
                # Edge case: no point exceeds target FPR
                idx = len(fpr_full) - 1
                actual_tpr = tpr_full[idx]
                actual_fpr = fpr_full[idx]
                threshold = thresholds_full[idx] if idx < len(thresholds_full) else 0.5
            else:
                # Get first point that meets or exceeds target FPR
                idx = indices[0]
                actual_tpr = tpr_full[idx]
                actual_fpr = fpr_full[idx]
                threshold = thresholds_full[idx]

            # Ensure threshold is never exactly 0 or 1
            if threshold >= 1.0:
                threshold = 1.0 - epsilon
            elif threshold <= 0.0:
                threshold = epsilon

            # For different FPR targets, slightly adjust thresholds to ensure different results
            # Each higher FPR target gets a slightly lower threshold
            threshold_adjustment = epsilon * i
            threshold = max(epsilon, min(1.0 - epsilon, threshold - threshold_adjustment))

            # Recalculate TPR and FPR with the adjusted threshold
            predictions = (predicted_probabilities_np >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (labels_np == 1))
            fp = np.sum((predictions == 1) & (labels_np == 0))

            actual_tpr = tp / positives if positives > 0 else 0.0
            actual_fpr = fp / negatives if negatives > 0 else 0.0

            # Calculate standard error
            std_error = np.sqrt(actual_tpr * (1 - actual_tpr) / positives) if positives > 0 else 0.0

            tpr_at_target_fpr_info[target_fpr] = {
                "value": float(actual_tpr),
                "std": float(std_error),
                "threshold": float(threshold),
                "actual_fpr": float(actual_fpr),
            }

        # Extract the dictionaries in the required format
        tpr_at_fpr = {
            target_fpr: info["value"] for target_fpr, info in tpr_at_target_fpr_info.items()
        }
        tpr_at_fpr_with_std = {
            target_fpr: {"value": info["value"], "std": info["std"]}
            for target_fpr, info in tpr_at_target_fpr_info.items()
        }

        return fpr_full, tpr_full, tpr_at_fpr, tpr_at_fpr_with_std

    def plot_roc_curve(self, out_dir="./roc_curve.png"):
        fpr, tpr, tpr_at_fpr, tpr_at_fpr_with_std = self.get_tpr_at_fpr_thresholds(
            self.labels, self.predicted_probabilities
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(fpr, tpr, "b-", label="ROC curve")
        for fpr_target, tpr_data in tpr_at_fpr_with_std.items():
            tpr_value = tpr_data["value"]
            tpr_std = tpr_data["std"]
            ax1.plot(fpr_target, tpr_value, "ro", label=f"TPR at FPR={fpr_target:.3f}")
            # Add error bars
            ax1.errorbar(fpr_target, tpr_value, yerr=tpr_std, fmt="none", ecolor="r", capsize=5)
        ax1.plot([0, 1], [0, 1], "k--", label="Random")
        ax1.set_xlim([0, 0.2])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve (Zoomed to FPR ≤ 0.2)")
        ax1.legend()
        ax1.grid(True)
        if self.accelerator.is_main_process:
            print(f"Saving ROC curve to {out_dir}")

        # Right subplot (full view)
        ax2.plot(fpr, tpr, "b-", label="ROC curve")
        for fpr_target, tpr_data in tpr_at_fpr_with_std.items():
            tpr_value = tpr_data["value"]
            tpr_std = tpr_data["std"]
            ax2.plot(fpr_target, tpr_value, "ro", label=f"TPR at FPR={fpr_target:.3f}")
            # Add error bars
            ax2.errorbar(fpr_target, tpr_value, yerr=tpr_std, fmt="none", ecolor="r", capsize=5)
        ax2.plot([0, 1], [0, 1], "k--", label="Random")
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve (Full View)")
        ax2.legend()
        ax2.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(out_dir)
        return tpr_at_fpr

    def print_metrics(self, metrics_file=None):
        metrics_file = metrics_file or self.metrics_file
        try:
            with open(metrics_file, "rb") as f:
                metrics = EvaluationMetrics(pickle.load(f))
        except:
            with open(metrics_file, "r") as f:
                metrics = EvaluationMetrics(json.load(f))

        for d in metrics.monitor_predictions:
            experiment = d["experiment"]
            predictions = d["monitor_predictions"]
            print(f"Experiment {experiment}")
            mon = MonitorPredictions.from_serializable(predictions)
            print(mon.get_aggregatable_metrics())

            # Calculate average logits for safe and unsafe examples
            labels = mon.labels.cpu().numpy()
            probs = mon.predicted_probabilities.cpu().numpy()

            # Convert probabilities to logits: logit = log(p/(1-p))
            # Adding epsilon to avoid division by zero or log(0)
            epsilon = 1e-7
            probs = np.clip(probs, epsilon, 1 - epsilon)
            logits = np.log(probs / (1 - probs))

            safe_logits = logits[labels == 0]
            unsafe_logits = logits[labels == 1]

            avg_safe_logit = safe_logits.mean() if len(safe_logits) > 0 else 0.0
            avg_unsafe_logit = unsafe_logits.mean() if len(unsafe_logits) > 0 else 0.0
            logit_separation = avg_unsafe_logit - avg_safe_logit

            print(f"Average logit for safe examples: {avg_safe_logit:.3f}")
            print(f"Average logit for unsafe examples: {avg_unsafe_logit:.3f}")
            print(f"Logit separation: {logit_separation:.3f}")


@dataclass
class Monitor:
    name: str

    def __init__(
        self,
        model_name: str,
        monitor_dir: str = None,
        dtype: torch.dtype = torch.float32,
        activations_dir: str = None,
        params_dir: str = None,
        inference_batch_size_per_device: int = 1,
    ):
        self.model_name = model_name
        self.monitor_dir = monitor_dir

        os.makedirs(monitor_dir, exist_ok=True)

        self.dtype = dtype
        self.inference_batch_size_per_device = inference_batch_size_per_device
        self.activations_dir = activations_dir or f"{monitor_dir}/activations"
        self.params_dir = params_dir or f"{monitor_dir}/params"
        self.clear_activation_cache = True

        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.model = None  # We'll initialize the model later if when we need it
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.load_params()

    def init_params(self):
        """
        Initializes whatever trainable parameters the monitor uses.
        """
        raise NotImplementedError

    def forward(
        self, trajectories: List[List[Action]], activations: List[Tensor], **kwargs
    ) -> Tensor:
        """
        Returns a Tensor of probabilities that each trajectory is unsafe.
        """
        raise NotImplementedError

    def init_model(self, download=True):
        if self.model is None:
            if self.on_policy_model_organism:
                # HERE
                # model_path = "/mnt/nvme5/anshul/models/vio_ft"
                # model_path = "/mnt/nvme5/anshul/models/sabotage_ft"
                # model_path = "/mnt/nvme5/anshul/models/deception_ft"
                model_path = "/mnt/nvme3/aghyad/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7"
            else:
                model_path = "/mnt/nvme3/aghyad/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7"
            if os.path.exists(model_path):
                if self.accelerator.is_main_process:
                    print(f"Loading model from {model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(model_path).to(dtype=self.dtype)
            else:
                if download:
                    # from_pretrained does not download in parallel; huggingface-cli is 8x faster
                    download_command = ["huggingface-cli", "download", self.model_name]
                process = subprocess.Popen(
                    download_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for line in process.stdout:
                    if self.accelerator.is_main_process:
                        print(line, end="")

                process.wait()
                if process.returncode != 0 and self.accelerator.is_main_process:
                    print(
                        f"Error {process.returncode} downloading model {self.model_name}; falling back to from_pretrained"
                    )
                self.model = (
                    AutoModelForCausalLM.from_pretrained(self.model_name)
                    .to(dtype=self.dtype)
                    .eval()
                    .to(self.device)
                )
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.print("Loaded model")
        else:
            self.accelerator.print("Model already loaded")

    def save_params(self):
        """
        A utility for saving the parameters of the monitor (e.g. after training it)
        """
        if not os.path.exists(self.params_dir):
            os.makedirs(self.params_dir)
        torch.save(self.params.state_dict(), f"{self.params_dir}/params.pt")
        self.accelerator.print(f"Saved params to {self.params_dir}/params.pt")

    def load_params(self):
        """
        A utility for loading the parameters of the monitor
        """
        if not os.path.exists(self.params_dir):
            self.params = None
            self.accelerator.print("No params found")
        else:
            self.params = torch.load(
                f"{self.params_dir}/params.pt",
                weights_only=False,
                map_location=self.device,
            )
            self.accelerator.print("Loaded params")

    def get_activations(self, trajectories: List[List[Any]], layers: List[int]) -> List[Tensor]:
        """
        A helper method for self and self.evaluate.
        Returns all the activations at every token position in a trajectory rollout at the specified layers.
        These activaitons are saved and used to train and evaluate monitors.
        The result is a list of tensors, where the dimensions of each tensor are of dimension (num_layers, num_tokens_in_chat_history, hidden_size).
        """
        # Convert trajectories to message lists
        messages_list = [trajectory[-1]["messages"] for trajectory in trajectories]
        # Tokenization with padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).int().to(self.device)

        # Register layer hooks
        hidden_states = defaultdict(list)
        hooks = []

        def _hook_fn(layer_idx):
            def hook(module, input, output):
                hidden_states[layer_idx] = output[0] if isinstance(output, tuple) else output

            return hook

        for layer_idx in layers:
            if type(self.model) == torch.nn.parallel.DistributedDataParallel:
                layer = self.model.module.model.layers[layer_idx]
            else:
                layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(_hook_fn(layer_idx))
            hooks.append(hook)

        def cleanup():
            for hook in hooks:
                hook.remove()
            hooks.clear()
            hidden_states.clear()

        # Run forward pass and extract activations
        try:
            with torch.no_grad(), self.accelerator.autocast():
                self.model.eval()
                # Forward pass without output_hidden_states
                self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Stack hidden states from requested layers
            stacked_states = torch.stack([hidden_states[idx] for idx in sorted(layers)], dim=0)

            # Split by trajectories dimension
            return [stacked_states[:, i, :, :] for i in range(len(input_ids))]

        finally:
            # Ensure hooks are cleaned up even if an error occurs
            cleanup()

    def save_activations(
        self,
        chunks: List[List[List[Action]]],
        activations_dir: str,
        clear_activation_cache: bool = False,
    ):
        """
        A helper method for caching activations before training or evaluation.
        Caching is done for two reasons:
            - It speeds up computations if training or evaluations are run multiple times.
            - (the main reason) It allows activations to loaded in batches in the case where they are large and don't fit in memory.

        Must be called an equal number of times on each thread.
        """
        try:
            assert (
                len(chunks) == self.accelerator.num_processes
            ), f"Chunks must be split across {self.accelerator.num_processes} processes, but there are {len(chunks)} chunks."
            assert (
                len(set([len(chunk) for chunk in chunks])) == 1
            ), f"All chunks must have the same length: {[len(chunk) for chunk in chunks]}"
            trajectories = chunks[self.accelerator.process_index]

            if os.path.exists(activations_dir):
                if not clear_activation_cache:
                    activation_paths = os.listdir(activations_dir)
                    trajectory_indices_of_saved_activations = [
                        int(path.split(".")[0]) for path in activation_paths
                    ]
                    indices_of_requested_trajectories = [
                        trajectory[-1]["trajectory_index"] for trajectory in trajectories
                    ]
                    if set(indices_of_requested_trajectories).issubset(
                        set(trajectory_indices_of_saved_activations)
                    ):
                        activation_paths_of_requested_trajectories = [
                            f"{activations_dir}/{index}.pt"
                            for index in indices_of_requested_trajectories
                        ]
                        return activation_paths_of_requested_trajectories

            if not os.path.exists(activations_dir):
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    os.makedirs(activations_dir)

            self.accelerator.wait_for_everyone()
            self.init_model()

            activation_paths = []
            for i in tqdm(
                range(0, len(trajectories), self.inference_batch_size_per_device),
                desc=f"[Rank {self.accelerator.process_index}] Saving activations",
            ):
                batch_trajectories = trajectories[i : i + self.inference_batch_size_per_device]
                batch_activations = self.get_activations(batch_trajectories)
                assert len(batch_trajectories) == len(batch_activations)
                for j, activations in enumerate(batch_activations):
                    index = batch_trajectories[j][-1]["trajectory_index"]
                    path = f"{activations_dir}/{index}.pt"
                    activation_paths.append(path)
                    torch.save(activations, path)
                del batch_activations
                torch.cuda.synchronize()  # HERE
                torch.cuda.empty_cache()

            return activation_paths
        except Exception as e:
            raise

    @torch.enable_grad()
    def fit(
        self,
        activation_paths,
        trajectories,
        labels,
        num_epochs,
        lr,
        l2_penalty=0.001,
        min_grad_norm=0.001,
        max_grad_norm=400.0,
        optimizer=None,
        **kwargs,
    ):
        """Modified to run training up to 5 times until loss < 0.5"""
        target_loss = 0.5  # Target loss threshold
        max_attempts = 10  # Maximum number of training attempts
        patience = kwargs.get(
            "patience", 4
        )  # Number of epochs to wait before reinitializing, # HERE
        loss_delta_threshold = kwargs.get("loss_delta_threshold", 1e-4)  # Minimum change in loss

        # Keep track of best parameters and loss
        best_loss = float("inf")
        best_state_dict = None

        if self.accelerator.is_main_process:
            print(
                f"Training on process {self.accelerator.process_index} for {num_epochs} epochs with {lr = }"
            )
            print(f"Will attempt training up to {max_attempts} times until loss < {target_loss}")
            print(
                f"Will reinitialize parameters if loss doesn't change by {loss_delta_threshold} for {patience} epochs"
            )

            # Load activations once outside the retry loop
            activations = [
                torch.load(path, weights_only=True, map_location=self.device)
                for path in activation_paths
            ]

            # Add label smoothing - convert binary labels to soft labels
            smoothing_factor = kwargs.get("label_smoothing", 0.05)  # Default 0.05

            # Convert labels to tensor if needed
            if not isinstance(labels, torch.Tensor):
                labels_tensor = torch.tensor(labels, dtype=self.dtype, device=self.device)
            else:
                labels_tensor = labels.to(self.dtype).to(self.device)

            # Apply label smoothing: move labels slightly away from 0 and 1 if using learned attention probe or logistic adversarial probe
            # HERE
            # if self.probe_type == "learned_attention_probe":
            if self.accelerator.is_main_process:
                print("smoothing labels")
            smoothed_labels = labels_tensor * (1 - smoothing_factor) + smoothing_factor * 0.5
            # else:
            #     smoothed_labels = labels_tensor
            # Attempt training multiple times
            for attempt in range(max_attempts):
                print(f"\n===== Training Attempt {attempt+1}/{max_attempts} =====")

                # Reset parameters for each attempt
                self.init_params()

                # Use provided optimizer or create default AdamW
                if optimizer is None:
                    print(f"Using AdamW optimizer with weight_decay={l2_penalty}")
                    optimizer = torch.optim.AdamW(
                        self.params.parameters(),
                        lr=lr,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=l2_penalty,
                    )

                final_loss = None
                # For tracking loss plateau
                last_losses = []
                patience_counter = 0

                for epoch in tqdm(range(num_epochs), desc="Epoch"):
                    # Only print debug info every 5 epochs
                    should_print_debug = epoch % 5 == 0

                    def closure():
                        optimizer.zero_grad()
                        predictions = self.forward(trajectories, activations)
                        loss = torch.nn.functional.binary_cross_entropy(
                            predictions, smoothed_labels
                        )

                        # Always print the current loss value (added for better visibility)
                        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

                        loss.backward()

                        # Before clipping - Simplified
                        with torch.no_grad():
                            pre_clip_norms = [
                                torch.norm(p.grad).item()
                                for p in self.params.parameters()
                                if p.grad is not None
                            ]
                            pre_clip_max = max(pre_clip_norms) if pre_clip_norms else 0

                            # Only print important warning messages for parameters
                            has_nan_or_inf = False
                            for name, p in self.params.named_parameters():
                                if p.grad is not None:
                                    if torch.isnan(p.grad).any():
                                        print(f"  WARNING: NaN gradients in {name}")
                                        has_nan_or_inf = True
                                    if torch.isinf(p.grad).any():
                                        print(f"  WARNING: Inf gradients in {name}")
                                        has_nan_or_inf = True

                        # Apply clipping
                        torch.nn.utils.clip_grad_norm_(self.params.parameters(), max_grad_norm)

                        # After clipping - Simplified
                        with torch.no_grad():
                            post_clip_norms = [
                                torch.norm(p.grad).item()
                                for p in self.params.parameters()
                                if p.grad is not None
                            ]
                            post_clip_max = max(post_clip_norms) if post_clip_norms else 0

                        return loss

                    try:
                        loss = optimizer.step(closure)
                        current_loss = loss.item()
                        final_loss = current_loss  # Store the final loss for this epoch

                        # Handle NaN loss
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss detected at epoch {epoch}. Skipping update.")
                            continue

                        # Check for plateaus - add current loss to history and check for stagnation
                        last_losses.append(current_loss)

                        # Keep only the most recent losses for comparison
                        if len(last_losses) > patience:
                            last_losses.pop(0)

                        # Check if loss has plateaued
                        if len(last_losses) >= patience:
                            loss_changes = [
                                abs(last_losses[i] - last_losses[i - 1])
                                for i in range(1, len(last_losses))
                            ]
                            avg_change = sum(loss_changes) / len(loss_changes)

                            if (
                                avg_change < loss_delta_threshold
                                and self.probe_type == "learned_attention_probe"
                            ):
                                patience_counter += 1
                                if patience_counter >= 1:  # Only need to detect once
                                    print(f"\n===== LOSS PLATEAU DETECTED =====")
                                    print(
                                        f"Average loss change over last {patience} epochs: {avg_change:.6f}"
                                    )
                                    print(f"Reinitializing parameters to escape local minimum...")

                                    # Reinitialize parameters randomly
                                    self.init_params()

                                    # Reset optimizer with new parameters
                                    optimizer = torch.optim.AdamW(
                                        self.params.parameters(),
                                        lr=lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=l2_penalty,
                                    )

                                    # Reset loss history after reinitialization
                                    last_losses = []
                                    patience_counter = 0
                            else:
                                patience_counter = 0

                        # Simplified gradient norm monitoring
                        with torch.no_grad():
                            grad_norms = []
                            has_nan_or_inf = False

                            for name, p in self.params.named_parameters():
                                if p.grad is not None:
                                    grad_norm = torch.norm(p.grad.detach())
                                    grad_norms.append(grad_norm)
                                    if torch.isnan(p.grad).any():
                                        print(f"  WARNING: NaN gradients detected in {name}")
                                        has_nan_or_inf = True
                                    if torch.isinf(p.grad).any():
                                        print(f"  WARNING: Inf gradients detected in {name}")
                                        has_nan_or_inf = True

                            # Safer way to calculate total norm
                            if grad_norms:
                                total_grad_norm = torch.norm(torch.stack(grad_norms))
                            else:
                                total_grad_norm = torch.tensor(0.0, device=self.device)
                                print("No gradients found!")

                            # Only print warnings for abnormal gradient norms
                            if total_grad_norm < min_grad_norm:
                                print(
                                    f"WARNING: Gradient norm {total_grad_norm:.6f} is below min_grad_norm {min_grad_norm}"
                                )
                            if total_grad_norm > max_grad_norm:
                                print(
                                    f"WARNING: Gradient norm {total_grad_norm:.6f} exceeds max_grad_norm {max_grad_norm}"
                                )

                        # This is the main per-epoch status update - keep this prominent
                        tqdm.write(
                            f"Attempt {attempt+1}, Epoch {epoch}: Loss = {loss:.4f}, Gradient Norm = {total_grad_norm:.4f}"
                        )
                        # HERE
                        # Check for early stopping for learned attention probes if loss is below 0.5
                        if (
                            self.probe_type == "learned_attention_probe"
                            or self.probe_type == "logistic_probe"
                        ):
                            if loss <= kwargs.get("early_stopping_loss", 0.55):
                                tqdm.write(
                                    f"Stopping early as loss {loss:.4f} <= {kwargs.get('early_stopping_loss', 0.55)}"
                                )
                                final_loss = loss.item()
                                break

                    except RuntimeError as e:
                        if "nan" in str(e).lower() or "infinity" in str(e).lower():
                            print(
                                f"Warning: Runtime error with NaN or Inf at epoch {epoch}. Continuing to next epoch."
                            )
                            continue
                        else:
                            raise

                # After training loop, check if this attempt meets the target loss
                if final_loss is not None:
                    print(f"Attempt {attempt+1} finished with loss: {final_loss:.4f}")

                    # Save the best model parameters seen so far
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_state_dict = copy.deepcopy(self.params.state_dict())

                    # If we've reached the target loss, stop retrying
                    if final_loss < target_loss:
                        print(
                            f"Target loss {target_loss} achieved at attempt {attempt+1}. Stopping."
                        )
                        break
                else:
                    print(f"Attempt {attempt+1} failed to produce a valid loss.")

            # Load the best parameters (either below target or best seen)
            if best_state_dict is not None:
                self.params.load_state_dict(best_state_dict)
                print(f"Using best parameters with loss {best_loss:.4f}")

            self.save_params()

        self.accelerator.wait_for_everyone()

    def split_evenly(self, trajectories):
        # Split data across GPUs
        assert len(trajectories) > 0, "Trajectories must be non-empty"
        num_gpus = self.accelerator.num_processes
        self.chunk_size = len(trajectories) // num_gpus
        chunks = [
            trajectories[i * self.chunk_size : (i + 1) * self.chunk_size] for i in range(num_gpus)
        ]
        return chunks

    def train(self, trajectories_path, clear_activation_cache=True, num_epochs=35, **kwargs):
        """
        Used to train the monitor on a set of trajectories.
        Saves parameters to self.params_dir.
        """
        self.init_params()

        clear_activation_cache = self.clear_activation_cache or clear_activation_cache
        with open(trajectories_path, "r") as f:
            trajectories = json.load(f)
            new_trajectories = []
            for trajectory in trajectories:
                if isinstance(trajectory, dict):
                    new_trajectories.append([Action(**action) for action in trajectory["actions"]])
                elif isinstance(trajectory, list):
                    new_trajectories.append([Action(**action) for action in trajectory])
                else:
                    raise ValueError(f"Unknown trajectory type: {type(trajectory)}")
            trajectories = new_trajectories

        # Split data across GPUs before obtaining activations
        chunks = self.split_evenly(trajectories)
        trajectories = trajectories[: sum([len(chunk) for chunk in chunks])]
        train_activations_dir = f"{self.activations_dir}/train"
        if self.accelerator.is_main_process:
            print(f"Saving activations")
        activation_paths = self.save_activations(
            chunks, train_activations_dir, clear_activation_cache
        )

        # Wait for all processes to finish saving activations for their assigned trajectory chunks before training
        self.accelerator.wait_for_everyone()

        labels = [trajectory[-1]["classification"] == "unsafe" for trajectory in trajectories]
        activation_paths = [
            f"{train_activations_dir}/{trajectory[-1]['trajectory_index']}.pt"
            for trajectory in trajectories
        ]

        # Get the learning rate from kwargs if provided, otherwise use default
        lr = kwargs.pop("lr")  # Default to 1e-3 if not specified

        self.accelerator.print(f"\n===== Training with learning rate: {lr} =====")

        # Train with the specified learning rate
        self.fit(activation_paths, trajectories, labels, num_epochs, lr=lr, **kwargs)

        # Evaluate on the training data to get the loss
        if self.accelerator.is_main_process:
            eval_results = self.evaluate_with_current_params(trajectories, labels, activation_paths)
            loss = eval_results["cross_entropy_loss"]
            auc = eval_results["auc"]

            self.accelerator.print(f"Training complete: Loss = {loss:.4f}, AUC = {auc:.4f}")

            # Save the parameters
            self.save_params()

        self.accelerator.wait_for_everyone()

    def evaluate(self, trajectories_path, out_path, clear_activation_cache=False, **kwargs):
        """
        Used to evaluate the monitor on a set of trajectories. Returns the false positive rate and false negative rate.
        TODO: Save evaluations to a file for large datasets.
        """
        if self.accelerator.is_main_process:
            print(f"\n===== Evaluating Monitor =====")
            print(f"Monitor Name: {self.name}")
            print(f"Model Name: {self.model_name}")
            print(f"Monitor Directory: {self.monitor_dir}")
            print(f"Activations Directory: {self.activations_dir}")
            print(f"Parameters Directory: {self.params_dir}")
            print(f"Data Type: {self.dtype}")
            print(f"Inference Batch Size per Device: {self.inference_batch_size_per_device}")

            if hasattr(self, "layer"):
                print(f"Layer: {self.layer}")
            if hasattr(self, "token_position_aggregation_strategy"):
                print(f"Token Aggregation: {self.token_position_aggregation_strategy}")
            if hasattr(self, "fit_method"):
                print(f"Fit Method: {self.fit_method}")

            # Print parameter status
            if self.params is not None:
                param_status = (
                    f"Loaded (Shape: {self.params.weight.shape})"
                    if hasattr(self.params, "weight")
                    else "Loaded (Unknown Shape)"
                )
                print(f"Probe Parameters: {param_status}")
            else:
                print("Probe Parameters: Not Initialized/Loaded")

            print(f"\n--- Evaluation Call Arguments ---")
            print(f"Trajectories path: {trajectories_path}")
            print(f"Output path: {out_path}")
            print(f"Clear activation cache: {clear_activation_cache}")
            if kwargs:
                print(f"Additional keyword arguments: {kwargs}")
            print("============================\n")

        clear_activation_cache = self.clear_activation_cache or clear_activation_cache

        with open(trajectories_path, "r") as f:
            trajectories = json.load(f)
            new_trajectories = []
            for trajectory in trajectories:
                if isinstance(trajectory, dict):
                    new_trajectories.append([Action(**action) for action in trajectory["actions"]])
                elif isinstance(trajectory, list):
                    new_trajectories.append([Action(**action) for action in trajectory])
                else:
                    raise ValueError(f"Unknown trajectory type: {type(trajectory)}")
            trajectories = new_trajectories

        chunks = self.split_evenly(trajectories)
        trajectories = trajectories[: sum([len(chunk) for chunk in chunks])]

        eval_activations_dir = f"{self.activations_dir}/eval"
        activation_paths = self.save_activations(
            chunks, eval_activations_dir, clear_activation_cache
        )

        self.accelerator.wait_for_everyone()

        # assigned_trajectories = chunks[self.accelerator.process_index]
        predictions = []
        labels = [trajectory[-1]["classification"] == "unsafe" for trajectory in trajectories]

        # gather activation paths
        activation_paths = gather_object([activation_paths])
        activation_paths = [path for paths in activation_paths for path in paths]

        assert (
            len(activation_paths) == len(trajectories) == len(labels)
        ), f"Num of activation paths ({len(activation_paths)}) must match num of trajectories ({len(trajectories)}) and labels ({len(labels)})"

        # All processes save to a file, so the main process can load all the predictions and labels
        if self.accelerator.is_main_process:
            total_batches = math.ceil(len(labels) / self.inference_batch_size_per_device)
            pbar = tqdm(total=total_batches, desc="Forward passes through probe")

            for i in range(0, len(labels), self.inference_batch_size_per_device):
                batch_activations = [
                    torch.load(path, weights_only=True, map_location=self.device)  # HERE
                    for path in activation_paths[i : i + self.inference_batch_size_per_device]
                ]
                batch_predictions = self.forward(
                    trajectories[i : i + self.inference_batch_size_per_device],
                    batch_activations,
                    **kwargs,
                )
                predictions.extend(batch_predictions)
                del batch_activations
                pbar.update(1)

            pbar.close()

            predictions = torch.tensor(predictions, device=self.device)
            labels = torch.tensor(labels, device=self.device)

            predictions = predictions.cpu().detach().to(torch.float32).numpy()
            labels = labels.cpu().detach().to(torch.float32).numpy()
            self.accelerator.wait_for_everyone()
            ret = MonitorPredictions(labels, predictions)
            # make any required directories
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pickle.dump(ret, open(out_path, "wb"))
            print(f"Saved predictions to {out_path}")
            return ret
        else:
            self.accelerator.wait_for_everyone()
            return None

    @classmethod
    def make_train_and_eval_data(
        cls,
        safe_traj_path: str,
        unsafe_traj_path: str,
        model_name: str,
        third_safe_path: Optional[str] = None,
        third_unsafe_path: Optional[str] = None,
        train_output_path: Optional[str] = None,
        eval_output_path: Optional[str] = None,
        num_train_points: Optional[int] = None,
        num_eval_points: Optional[int] = None,
        min_num_eval_points: int = 10,
        max_token_length=4096,
        mode="standard",
        add_instruction_following: bool = False,
    ):
        """
        A utility for combining safe and unsafe trajectories and converting them into files for training and evaluating monitors.
        NOT meant to run with accelerate

        TODO change to only make data and clear activation cache if necessary
        """
        # self.clear_activation_cache = True
        # train_output_path = train_output_path or f"./{self.monitor_dir}/train_trajectories.json/"
        # eval_output_path = eval_output_path or f"./{self.monitor_dir}/eval_trajectories.json"
        if train_output_path is None and eval_output_path is None:
            raise ValueError(
                "At least one of train_output_path or eval_output_path must be specified."
            )

        with open(safe_traj_path, "r") as f:
            safe_trajectories = json.load(f)

        with open(unsafe_traj_path, "r") as f:
            unsafe_trajectories = json.load(f)

        # HERE randomly shuffle trajectories
        random.shuffle(safe_trajectories)
        random.shuffle(unsafe_trajectories)

        third_safe_trajectories = []
        third_unsafe_trajectories = []
        if third_safe_path is not None:
            with open(third_safe_path, "r") as f:
                third_safe_trajectories = json.load(f)
            with open(third_unsafe_path, "r") as f:
                third_unsafe_trajectories = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Filter trajectories by sequence length
        def filter_by_sequence_length(trajectories, min_token_length, max_token_length):
            ls = []
            for i, trajectory in enumerate(trajectories):
                if type(trajectory) == dict and "actions" in trajectory:
                    if (
                        min_token_length
                        <= len(
                            tokenizer.apply_chat_template(
                                trajectory["actions"][-1]["messages"], tokenize=True
                            )
                        )
                        < max_token_length
                    ):
                        ls.append(trajectory)
                elif type(trajectory) == dict and "action_id" in trajectory:
                    if (
                        min_token_length
                        <= len(tokenizer.apply_chat_template(trajectory["messages"], tokenize=True))
                        < max_token_length
                    ):
                        ls.append(trajectory)
                elif type(trajectory) == list:
                    if (
                        min_token_length
                        <= len(
                            tokenizer.apply_chat_template(trajectory[-1]["messages"], tokenize=True)
                        )
                        < max_token_length
                    ):
                        ls.append(trajectory)
            return ls

        safe_trajectories = filter_by_sequence_length(safe_trajectories, 0, max_token_length)
        unsafe_trajectories = filter_by_sequence_length(unsafe_trajectories, 0, max_token_length)
        if third_safe_path is not None:
            third_safe_trajectories = filter_by_sequence_length(
                third_safe_trajectories, 0, max_token_length
            )
            third_unsafe_trajectories = filter_by_sequence_length(
                third_unsafe_trajectories, 0, max_token_length
            )

        # print(
        #     f"Primary dataset - Safe: {len(safe_trajectories)}, Unsafe: {len(unsafe_trajectories)}"
        # )
        # if third_safe_path is not None:
        #     print(
        #         f"Third dataset - Safe: {len(third_safe_trajectories)}, Unsafe: {len(third_unsafe_trajectories)}"
        #     )

        max_train_size = min(
            len(safe_trajectories),
            len(third_safe_trajectories) if third_safe_path is not None else float("inf"),
            len(unsafe_trajectories),
            len(third_unsafe_trajectories) if third_unsafe_path is not None else float("inf"),
        )

        if train_output_path is None:
            print("MAKING EVAL DATA")
            num_train_points = 0
        if eval_output_path is None:
            print("MAKING TRAIN DATA")
            num_eval_points = 0

        if max_train_size < num_train_points:
            print(
                f"Warning: The maximum training set size is {max_train_size}, which is less than the requested training set size of {num_train_points}."
            )
            num_train_points = min(max_train_size, num_train_points)

        if mode == "standard":
            train_safe_trajectories = safe_trajectories[:num_train_points]
            train_unsafe_trajectories = unsafe_trajectories[:num_train_points]
            train_third_safe_trajectories = (
                third_safe_trajectories[:num_train_points] if third_safe_path is not None else []
            )
            train_third_unsafe_trajectories = (
                third_unsafe_trajectories[:num_train_points]
                if third_unsafe_path is not None
                else []
            )

            # Add detailed breakdown of training data composition
            print("\n=== Training Data Composition ===")
            print(f"Primary dataset - Safe: {len(train_safe_trajectories)}")
            print(f"Primary dataset - Unsafe: {len(train_unsafe_trajectories)}")
            if third_safe_path is not None:
                print(f"Third dataset - Safe: {len(train_third_safe_trajectories)}")
                print(f"Third dataset - Unsafe: {len(train_third_unsafe_trajectories)}")

            train_trajectories = (
                train_safe_trajectories
                + train_unsafe_trajectories
                + train_third_safe_trajectories
                + train_third_unsafe_trajectories
            )
            random.shuffle(train_trajectories)

            assert all([len(trajectory) > 0 for trajectory in train_trajectories])

            eval_safe_trajectories = safe_trajectories[num_train_points:]
            eval_unsafe_trajectories = unsafe_trajectories[num_train_points:]
            if num_eval_points:
                eval_safe_trajectories = eval_safe_trajectories[:num_eval_points]
                eval_unsafe_trajectories = eval_unsafe_trajectories[:num_eval_points]

            if (
                len(eval_safe_trajectories) < min_num_eval_points
                and num_eval_points != 0
                or len(eval_unsafe_trajectories) < min_num_eval_points
                and num_eval_points != 0
            ):
                print("Num unsafe eval: ", len(eval_unsafe_trajectories))
                print("Num safe eval: ", len(eval_safe_trajectories))
                raise Exception(
                    f"Not enough eval data to have > {min_num_eval_points} points in each category."
                )
            eval_trajectories = eval_safe_trajectories + eval_unsafe_trajectories
            random.shuffle(eval_trajectories)

        # Calculate class balance percentages
        if len(train_trajectories) > 0:
            train_unsafe_count = sum(
                1
                for t in train_trajectories
                if (type(t) == dict and t["actions"][-1]["classification"] == "unsafe")
                or (type(t) != dict and t[-1]["classification"] == "unsafe")
            )
            train_safe_count = len(train_trajectories) - train_unsafe_count
            print(
                f"Training data class balance: {train_safe_count} safe ({train_safe_count/len(train_trajectories)*100:.1f}%), "
                + f"{train_unsafe_count} unsafe ({train_unsafe_count/len(train_trajectories)*100:.1f}%)"
            )

        if len(eval_trajectories) > 0:
            eval_unsafe_count = sum(
                1
                for t in eval_trajectories
                if (type(t) == dict and t["actions"][-1]["classification"] == "unsafe")
                or (type(t) != dict and t[-1]["classification"] == "unsafe")
            )
            eval_safe_count = len(eval_trajectories) - eval_unsafe_count
            print(
                f"Evaluation data class balance: {eval_safe_count} safe ({eval_safe_count/len(eval_trajectories)*100:.1f}%), "
                + f"{eval_unsafe_count} unsafe ({eval_unsafe_count/len(eval_trajectories)*100:.1f}%)"
            )

            # if specified, add instruction following data to training set
            if add_instruction_following:
                instruction_following_dir = "../evaluations/instruction_following_data"
                for dataset in ["safe", "unsafe"]:
                    dataset_path = os.path.join(
                        instruction_following_dir, f"instruction_following_{dataset}.json"
                    )
                    print(
                        f"Adding instruction following data from {dataset} dataset to training set"
                    )
                    with open(dataset_path, "r") as f:
                        instruction_following_trajectories = json.load(f)
                    train_trajectories.extend(instruction_following_trajectories)
            else:
                print("Not adding extra instruction following data to training set")

        # Add an index for distributed training
        for i, t in enumerate(eval_trajectories):
            if type(t) == dict:
                t["actions"][-1]["trajectory_index"] = i
            else:
                t[-1]["trajectory_index"] = i

        for i, t in enumerate(train_trajectories):
            if type(t) == dict:
                t["actions"][-1]["trajectory_index"] = i
            else:
                t[-1]["trajectory_index"] = i

        if eval_output_path is not None:
            os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
            with open(eval_output_path, "w") as f:
                json.dump(eval_trajectories, f, indent=4)

        if train_output_path is not None:
            os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
            with open(train_output_path, "w") as f:
                json.dump(train_trajectories, f, indent=4)

    def train_multiple_times(
        self, trajectories_path, num_runs=5, clear_activation_cache=True, num_epochs=500, **kwargs
    ):
        """
        Run the training process multiple times and report the results.
        """
        results = []

        # Only clear activation cache once at the beginning
        initial_clear_cache = clear_activation_cache

        for run in range(num_runs):
            if self.accelerator.is_main_process:
                print(f"\n===== Training Run {run+1}/{num_runs} =====")

            # Initialize fresh parameters for each run
            self.init_params()

            # Only clear the cache on the first run
            clear_cache_this_run = initial_clear_cache if run == 0 else False

            # Train as normal
            with open(trajectories_path, "r") as f:
                trajectories = json.load(f)
                new_trajectories = []
                for trajectory in trajectories:
                    if isinstance(trajectory, dict):
                        new_trajectories.append(
                            [Action(**action) for action in trajectory["actions"]]
                        )
                    elif isinstance(trajectory, list):
                        new_trajectories.append([Action(**action) for action in trajectory])
                    else:
                        raise ValueError(f"Unknown trajectory type: {type(trajectory)}")
                trajectories = new_trajectories

            # Split data across GPUs before obtaining activations
            chunks = self.split_evenly(trajectories)
            trajectories = trajectories[: sum([len(chunk) for chunk in chunks])]
            train_activations_dir = f"{self.activations_dir}/train"
            if self.accelerator.is_main_process:
                print(f"Saving activations")
            activation_paths = self.save_activations(
                chunks, train_activations_dir, clear_cache_this_run
            )

            # Wait for all processes to finish saving activations for their assigned trajectory chunks before training
            self.accelerator.wait_for_everyone()

            labels = [trajectory[-1]["classification"] == "unsafe" for trajectory in trajectories]
            activation_paths = [
                f"{train_activations_dir}/{trajectory[-1]['trajectory_index']}.pt"
                for trajectory in trajectories
            ]

            # Modify the output path for each run
            run_params_dir = f"{self.params_dir}_run{run+1}"
            original_params_dir = self.params_dir
            self.params_dir = run_params_dir

            self.accelerator.print(f"---Fitting monitor for {num_epochs} epochs...")
            self.fit(activation_paths, trajectories, labels, num_epochs, **kwargs)

            # Evaluate on the same data to get results for this run
            if self.accelerator.is_main_process:
                # Evaluate using the parameters we just trained
                run_results = self.evaluate_with_current_params(
                    trajectories, labels, activation_paths
                )
                results.append(run_results)
                print(f"Run {run+1} results: {run_results}")

            # Restore original params dir
            self.params_dir = original_params_dir

        # Aggregate results across all runs
        if self.accelerator.is_main_process:
            self._report_aggregate_results(results)

        return results

    def evaluate_with_current_params(self, trajectories, labels, activation_paths):
        """Evaluate using current parameters without saving to file"""
        if self.accelerator.is_main_process:
            predictions = []
            batch_size = self.inference_batch_size_per_device

            for i in range(0, len(labels), batch_size):
                batch_activations = [
                    torch.load(path, weights_only=True, map_location=self.device)
                    for path in activation_paths[i : i + batch_size]
                ]
                batch_predictions = self.forward(
                    trajectories[i : i + batch_size],
                    batch_activations,
                )
                predictions.extend(batch_predictions)
                del batch_activations

            predictions = torch.tensor(predictions, device=self.device)
            labels = torch.tensor(labels, device=self.device)

            predictions = predictions.cpu().detach().to(torch.float32).numpy()
            labels = labels.cpu().detach().to(torch.float32).numpy()

            return MonitorPredictions(labels, predictions).get_aggregatable_metrics()
        return None

    def _report_aggregate_results(self, results):
        """Aggregate and report results from multiple training runs"""
        if not results:
            return

        # Calculate statistics for each metric
        metrics = {}
        for key in results[0].keys():
            if key == "tpr_at_fpr_with_std":
                # Special handling for tpr_at_fpr_with_std which has nested dictionaries
                metrics[key] = {}
                for fpr in results[0][key].keys():  # For each FPR threshold
                    metrics[key][fpr] = {
                        "value": {
                            "mean": np.mean([run[key][fpr]["value"] for run in results]),
                            "std": np.std([run[key][fpr]["value"] for run in results]),
                            "min": np.min([run[key][fpr]["value"] for run in results]),
                            "max": np.max([run[key][fpr]["value"] for run in results]),
                        },
                        "std": {
                            "mean": np.mean([run[key][fpr]["std"] for run in results]),
                            "std": np.std([run[key][fpr]["std"] for run in results]),
                            "min": np.min([run[key][fpr]["std"] for run in results]),
                            "max": np.max([run[key][fpr]["std"] for run in results]),
                        },
                    }
            elif isinstance(results[0][key], dict):
                # Handle normal dictionaries like tpr_at_fpr
                metrics[key] = {}
                for subkey in results[0][key].keys():
                    values = [run[key][subkey] for run in results]
                    metrics[key][subkey] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                    }
            else:
                # Handle simple metrics like accuracy, auc, etc.
                values = [run[key] for run in results]
                metrics[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        # Print the aggregate results (adjust as needed)
        print("\n===== Aggregate Results Across Runs =====")
        for metric, stats in metrics.items():
            if metric == "tpr_at_fpr_with_std":
                # Special handling for deeply nested tpr_at_fpr_with_std
                print(f"\n{metric}:")
                for fpr, fpr_stats in stats.items():
                    print(f"  FPR={fpr}:")
                    for stat_type, stat_values in fpr_stats.items():
                        print(
                            f"    {stat_type}: mean={stat_values['mean']:.3f}, std={stat_values['std']:.3f}, min={stat_values['min']:.3f}, max={stat_values['max']:.3f}"
                        )
            elif isinstance(stats, dict) and "mean" in stats:
                # Regular metrics like accuracy, auc
                print(
                    f"{metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}"
                )
            elif isinstance(stats, dict):
                # Normal dictionaries like tpr_at_fpr
                print(f"\n{metric}:")
                for submetric, substats in stats.items():
                    if isinstance(substats, dict) and "mean" in substats:
                        print(
                            f"  {submetric}: mean={substats['mean']:.3f}, std={substats['std']:.3f}, min={substats['min']:.3f}, max={substats['max']:.3f}"
                        )
                    else:
                        print(f"  {submetric}: {substats}")

        print("\nRun-by-run results:")
        for i, run_result in enumerate(results):
            print(f"Run {i+1}:")
            for metric, value in run_result.items():
                if isinstance(value, dict):
                    print(f"  {metric}: {value}")
                else:
                    print(f"  {metric}: {value:.3f}")

    def learning_rate_sweep(
        self,
        trajectories_path,
        clear_activation_cache=False,
        num_epochs=100,
        min_lr=4e-4,
        max_lr=2e-3,
        num_steps=4,
        **kwargs,
    ):
        """
        Perform a sweep of learning rates and return the best model parameters.

        Args:
            trajectories_path: Path to trajectories file
            clear_activation_cache: Whether to clear activation cache
            num_epochs: Number of epochs to train for each learning rate
            min_lr: Minimum learning rate to try
            max_lr: Maximum learning rate to try
            num_steps: Number of learning rates to try
            **kwargs: Additional arguments to pass to fit

        Returns:
            Best learning rate and its performance metrics
        """
        if self.accelerator.is_main_process:
            print(f"\n===== Starting Learning Rate Sweep =====")
            print(f"Range: {min_lr} to {max_lr} with {num_steps} steps")

        # Generate learning rates on log scale
        learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
        results = []
        best_lr = None
        best_loss = float("inf")
        best_tpr_at_fpr_001 = 0
        best_state_dict = None

        # Remove lr from kwargs if it exists to prevent duplicate argument error
        if "lr" in kwargs:
            del kwargs["lr"]

        # Prepare data once
        clear_activation_cache = self.clear_activation_cache or clear_activation_cache
        with open(trajectories_path, "r") as f:
            trajectories = json.load(f)
            new_trajectories = []
            for trajectory in trajectories:
                if isinstance(trajectory, dict):
                    new_trajectories.append([Action(**action) for action in trajectory["actions"]])
                elif isinstance(trajectory, list):
                    new_trajectories.append([Action(**action) for action in trajectory])
                else:
                    raise ValueError(f"Unknown trajectory type: {type(trajectory)}")
            trajectories = new_trajectories

        # Split data across GPUs before obtaining activations
        chunks = self.split_evenly(trajectories)
        trajectories = trajectories[: sum([len(chunk) for chunk in chunks])]
        train_activations_dir = f"{self.activations_dir}/train"
        if self.accelerator.is_main_process:
            print(f"Saving activations")
        activation_paths = self.save_activations(
            chunks, train_activations_dir, clear_activation_cache
        )

        # Wait for all processes to finish saving activations
        self.accelerator.wait_for_everyone()

        labels = [trajectory[-1]["classification"] == "unsafe" for trajectory in trajectories]
        activation_paths = [
            f"{train_activations_dir}/{trajectory[-1]['trajectory_index']}.pt"
            for trajectory in trajectories
        ]

        # Try each learning rate
        for i, lr in enumerate(learning_rates):
            if self.accelerator.is_main_process:
                print(f"\n----- Testing Learning Rate {i+1}/{len(learning_rates)}: {lr:.6f} -----")

            # Initialize fresh parameters for each run
            self.init_params()

            # Train with the current learning rate
            self.fit(activation_paths, trajectories, labels, num_epochs, lr=lr, **kwargs)

            # Evaluate on the training data to get metrics
            if self.accelerator.is_main_process:
                metrics = self.evaluate_with_current_params(trajectories, labels, activation_paths)
                results.append((lr, metrics))

                # Check if this is the best model so far
                current_loss = metrics["cross_entropy_loss"]
                current_tpr_at_fpr_001 = metrics["tpr_at_fpr"][0.01]

                if current_tpr_at_fpr_001 > best_tpr_at_fpr_001:
                    best_tpr_at_fpr_001 = current_tpr_at_fpr_001
                    best_lr = lr
                    best_state_dict = copy.deepcopy(self.params.state_dict())
                    if self.accelerator.is_main_process:
                        print(
                            f"New best TPR@FPR=0.01: {best_tpr_at_fpr_001:.4f} at LR={best_lr:.6f}"
                        )

                if current_loss < best_loss:
                    best_loss = current_loss
                    if self.accelerator.is_main_process:
                        print(f"New best loss: {best_loss:.4f} at LR={lr:.6f}")

        # Load the best parameters
        if best_state_dict is not None and self.accelerator.is_main_process:
            self.params.load_state_dict(best_state_dict)
            print(f"\n===== Learning Rate Sweep Complete =====")
            print(f"Best learning rate: {best_lr:.6f} with TPR@FPR=0.01: {best_tpr_at_fpr_001:.4f}")

            # Save the parameters from the best run
            self.save_params()

            # Report all results
            print("\nAll results:")
            for lr, metrics in results:
                print(
                    f"LR: {lr:.6f}, Loss: {metrics['cross_entropy_loss']:.4f}, TPR@FPR=0.01: {metrics['tpr_at_fpr'][0.01]:.4f}"
                )

        self.accelerator.wait_for_everyone()
        return best_lr if self.accelerator.is_main_process else None
