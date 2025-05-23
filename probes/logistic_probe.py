# %%
"""The standard logistic regression linear probe."""

import os
import time

# Remember to set HF_HOME and HF_TOKEN_PATH variables before importing anything from transformers
import argparse
import logging
import torch
from torch import Tensor
from typing import List, Literal, Any
from torch import nn
import torch.distributed as dist
from .base_probe import Monitor
from .data_types import Action
from transformers import AutoConfig
from pathlib import Path
from accelerate.utils import broadcast_object_list
from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

from huggingface_hub import whoami

# print(f'{os.environ["HF_HOME"]=}')
# print(f'{os.environ["HF_TOKEN_PATH"]}')
# whoami()

import os

# %%


class LogisticProbe(Monitor):
    name = "logistic_probe"

    def __init__(
        self,
        model_name: str,
        layer: int,
        token_position_aggregation_strategy: Literal["mean", "soft_max", "last_only"] = "last_only",
        fit_method: Literal["lbfgs", "adam"] = "adam",
        on_policy_model_organism: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.layer = layer
        self.token_position_aggregation_strategy = token_position_aggregation_strategy
        self.fit_method = fit_method
        self.params = None
        self.probe_type = "logistic_probe"
        self.on_policy_model_organism = on_policy_model_organism

    def init_params(self):
        if self.accelerator.is_main_process:
            print("===== init_params called =====")
        config = AutoConfig.from_pretrained(self.model_name)
        hidden_size = config.hidden_size
        if self.accelerator.is_main_process:
            print(f"Creating linear layer = nn.Linear({hidden_size}, 1)")
        self.params = nn.Linear(hidden_size, 1).to(self.dtype).to(self.device)
        if self.accelerator.is_main_process:
            print(
                f"Parameters (Linear) have been created on device={self.device}, dtype={self.dtype}"
            )

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
            assert (
                activations.shape[0] == 1
            ), f"Activations should be shape (1, num_tokens, hidden_size), but got {activations.shape}"
            self.params = self.params.to(self.device)
            logits = self.params(activations[0, :, :])
            token_heatmap = torch.sigmoid(logits)

            match self.token_position_aggregation_strategy:
                case "mean":
                    prediction = token_heatmap.mean()
                case "soft_max":
                    prediction = self.soft_max(token_heatmap)
                case "last_only":
                    prediction = token_heatmap[-1]
                case _:
                    raise ValueError(
                        f"Unknown token_position_aggregation_strategy {self.token_position_aggregation_strategy}"
                    )

            predictions[i] = prediction

        return predictions

    def save_model_params(self, file_path: str):
        if self.accelerator.is_main_process:
            if self.params is not None:
                torch.save(self.params.state_dict(), file_path)
                print(f"Model parameters saved to {file_path}")
            else:
                print("Model parameters are not initialized.")

    @torch.enable_grad()
    def fit(
        self,
        activation_paths,
        trajectories,
        labels,
        num_epochs,
        l2_penalty=0.001,
        max_iter=1000,
        **kwargs,
    ) -> float:
        if self.accelerator.is_main_process:
            print(f"Fit method: {self.fit_method}")
            print(f"Number of epochs: {num_epochs}")

        if self.params is not None:
            self.params = self.params.to(self.device)

        if self.fit_method == "adam":
            super().fit(activation_paths, trajectories, labels, num_epochs, **kwargs)
        elif not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
        else:
            optimizer = torch.optim.LBFGS(
                self.params.parameters(), line_search_fn="strong_wolfe", max_iter=max_iter
            )
            criterion = nn.BCEWithLogitsLoss()
            loss = torch.inf

            def closure():
                nonlocal loss
                optimizer.zero_grad()

                activations = [
                    torch.load(path, map_location=self.device) for path in activation_paths
                ]
                predictions = self.forward(trajectories, activations)
                labels_tensor = torch.tensor(labels, dtype=self.dtype, device=self.device)
                loss = criterion(predictions, labels_tensor)
                if l2_penalty:
                    reg_loss = loss + l2_penalty * self.params.weight.square().sum()
                else:
                    reg_loss = loss

                reg_loss.backward()
                return float(reg_loss)

            optimizer.step(closure)
            if self.accelerator.is_main_process:
                print("Loss: ", loss)
            self.save_params()

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
    "code_sabotage_on_policy",
    "code_rule_violation_on_policy",
    "code_deception_on_policy",
    "redblue",
    "adversarial_all",
}


short_model_names = {
    "meta-llama/Llama-3.1-70B-Instruct": "70b",
    "meta-llama/Llama-3.2-1B-Instruct": "1b",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a logistic probe.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--train_task", type=str)
    parser.add_argument("--eval_task", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--eval_path", type=str)
    parser.add_argument("--out_path", type=str, default="./out.pkl")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--num_train_trajectories", type=int, default=100)
    parser.add_argument("--num_eval_trajectories", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--eval_on_trainset", action="store_true", help="Whether to evaluate on training set"
    )
    parser.add_argument(
        "--on_policy_model_organism", choices=[True, False], type=bool, default=False
    )
    args = parser.parse_args()

    model = args.model
    assert args.train_task in tasks, f"Invalid train task: {args.train_task}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get the directory from out_path
    monitor_dir = os.path.dirname(args.out_path)

    start_time = time.time()
    linear_probe = LogisticProbe(
        model_name=model,
        monitor_dir=monitor_dir,
        inference_batch_size_per_device=1,
        dtype=torch.bfloat16,
        layer=args.layer,
        on_policy_model_organism=args.on_policy_model_organism,
    )
    model_init_time = time.time() - start_time
    if linear_probe.accelerator.is_main_process:
        print(f"Time to initialize model: {model_init_time:.2f} seconds")

    # Start timing training
    start_time = time.time()
    linear_probe.train(
        args.train_path,
        clear_activation_cache=True,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )
    training_time = time.time() - start_time
    if linear_probe.accelerator.is_main_process:
        print(f"Time for training: {training_time:.2f} seconds")
        linear_probe.save_model_params(os.path.join(monitor_dir, "model_params.pt"))

    # Evaluate on the training set
    if args.eval_on_trainset and linear_probe.accelerator.is_main_process:
        print(f"Evaluating on the training set: {args.train_path} ...")
        start_time = time.time()
        train_metrics = linear_probe.evaluate(
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

    if linear_probe.accelerator.is_main_process:
        print(f"Evaluating on {args.eval_path} ...")
        print(f"args.eval_path => {args.eval_path}")
        print(f"args.out_path => {args.out_path}")

    # Start timing evaluation
    start_time = time.time()
    monitor_predictions = linear_probe.evaluate(
        args.eval_path,
        out_path=args.out_path,
        clear_activation_cache=True,
    )
    eval_time = time.time() - start_time

    if linear_probe.accelerator.is_main_process:
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
