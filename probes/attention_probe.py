# %%
"""A logistic probe with learned attention.

This probe learns to attend to different parts of the input sequence using two learned vectors:
- A score vector that determines attention weights for each token
- A value vector that determines how to aggregate token information

The final prediction is computed as a weighted sum of token values, where weights are determined by the attention mechanism.
# HERE reduced max grad norm from 50 to 10, reduced weight init from 0.5 to 0.01
"""
import os

# Remember to set HF_HOME and HF_TOKEN_PATH variables before importing anything from transformers
import argparse
import logging
import torch
from torch import Tensor
from typing import List, Literal, Any
from torch import nn
import torch.distributed as dist
from .base_probe import Monitor
from transformers import AutoConfig
import math

logging.getLogger().setLevel(logging.WARNING)

# %%


class LearnedAttentionProbe(Monitor):
    name = "learned_attention_probe"

    def __init__(
        self,
        model_name: str,
        layer: int,
        fit_method: Literal["lbfgs", "adam"] = "adam",
        use_normalization: bool = False,
        attention_scaling: bool = True,
        initialization_scale: float = 1.0,
        on_policy_model_organism: bool = False,
        **kwargs,
    ):
        # Minimal initialization prints
        super().__init__(model_name, **kwargs)
        self.layer = layer
        self.fit_method = fit_method
        self.use_normalization = use_normalization
        self.attention_scaling = attention_scaling
        self.initialization_scale = initialization_scale
        self.init_params()
        self.probe_type = "learned_attention_probe"
        self.on_policy_model_organism = on_policy_model_organism

    def init_params(self):
        config = AutoConfig.from_pretrained(self.model_name)
        hidden_size = config.hidden_size
        self.hidden_size = hidden_size

        # Create a single module that contains both vectors
        class CombinedModule(nn.Module):
            def __init__(self, hidden_size, dtype, device, use_normalization):
                super().__init__()
                # Add layer normalization for activation inputs
                self.use_normalization = use_normalization
                if use_normalization:
                    self.norm = nn.LayerNorm(hidden_size).to(dtype).to(device)

                # Change linear layers to include bias terms # HERE removed bias terms
                self.score_vector = nn.Linear(hidden_size, 1, bias=False).to(dtype).to(device)
                self.value_vector = nn.Linear(hidden_size, 1, bias=False).to(dtype).to(device)

                # Initialize with Xavier/Glorot initialization for weights
                nn.init.xavier_uniform_(self.score_vector.weight, gain=1.0)
                nn.init.xavier_uniform_(self.value_vector.weight, gain=1.0)

                # Initialize biases to zero
                if self.score_vector.bias is not None:
                    nn.init.zeros_(self.score_vector.bias)
                if self.value_vector.bias is not None:
                    nn.init.zeros_(self.value_vector.bias)

            def forward(self, x):
                # Forward pass is handled in the probe's forward method
                return x

        self.params = CombinedModule(hidden_size, self.dtype, self.device, self.use_normalization)

        # Then scale up the weights to avoid tiny initial gradients
        with torch.no_grad():
            for param in self.params.parameters():
                # Initialize with larger values to avoid tiny gradients
                param.data.mul_(self.initialization_scale)

    def forward(self, trajectories, trajectory_activations, extra_info=None):
        if self.params is None:
            raise ValueError(
                "The probe parameters have not been initialized. "
                "Please call train() or init_params() before forward()."
            )

        if extra_info is not None and self.accelerator.is_main_process:
            print(f"Extra info provided: {extra_info}")
            raise Exception("stop")

        predictions = torch.zeros(len(trajectories), device=self.device, dtype=self.dtype)

        # Debug info structure
        debug_info = {
            "score_stats": [],
            "attention_stats": [],
            "value_stats": [],
            "logit_stats": [],
        }

        # Static counter to only print every 5 updates
        if not hasattr(self, "update_counter"):
            self.update_counter = 0
        self.update_counter += 1
        should_print = self.update_counter % 5 == 0

        for i, (trajectory, activations) in enumerate(zip(trajectories, trajectory_activations)):
            activations = activations.to(self.device)
            assert activations.shape[0] == 1

            # Apply normalization if enabled (removed prints)
            if self.params.use_normalization:
                normalized_activations = self.params.norm(activations[0])
            else:
                normalized_activations = activations[0]

            # Compute attention scores and weights
            scores = self.params.score_vector(normalized_activations)

            # Scale scores by sqrt(hidden_dim) if enabled
            if self.attention_scaling:
                scaling_factor = math.sqrt(self.hidden_size)
                scores = scores / scaling_factor

            # Subtract max for numerical stability before softmax
            scores = scores - scores.max(dim=0, keepdim=True).values
            temperature = 2.0  # Higher values = softer attention
            attention_weights = torch.softmax(scores / temperature, dim=0)

            # Compute values (removed verbose prints)
            values = self.params.value_vector(normalized_activations)

            # Simple weighted sum
            logits = (attention_weights * values).sum()

            # Only show warnings for NaN/Inf values
            if i == 0 and self.accelerator.is_main_process:
                if torch.isnan(logits):
                    print("WARNING: NaN value in logit")
                if torch.isinf(logits):
                    print("WARNING: Inf value in logit")

            # Standard sigmoid and clamping
            prediction = torch.sigmoid(logits)
            prediction = torch.clamp(prediction, min=1e-4, max=1 - 1e-4)
            predictions[i] = prediction

        return predictions

    def save_model_params(self, file_path: str):
        if self.params is not None:
            state_dict = {
                "score_vector": self.params.score_vector.state_dict(),
                "value_vector": self.params.value_vector.state_dict(),
            }
            # Also save normalization layer if it exists
            if self.params.use_normalization:
                state_dict["norm"] = self.params.norm.state_dict()

            torch.save(state_dict, file_path)
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
        lr=0.01,
        l2_penalty=0.001,
        **kwargs,
    ) -> float:
        """Always use super().fit() for consistency"""
        print(f"[Rank {self.accelerator.process_index}] Starting fit with {self.fit_method}")

        # Print detailed info about the training data
        if self.accelerator.is_main_process:
            print(f"\n===== TRAINING DATA INFO =====")
            print(f"Number of trajectories: {len(trajectories)}")
            print(f"Number of labels: {len(labels)}")
            print(
                f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative"
            )
            print(f"Positive class ratio: {sum(labels)/len(labels):.2f}")
            print(f"Learning rate: {lr}")
            print(f"L2 penalty: {l2_penalty}")
            print(f"Number of epochs: {num_epochs}")
            print(f"Additional kwargs: {kwargs}")

        # Always use super().fit(), but modify the optimizer setup if using LBFGS
        if self.fit_method == "lbfgs":
            kwargs["optimizer"] = torch.optim.LBFGS(
                self.params.parameters(),
                line_search_fn="strong_wolfe",
                max_iter=kwargs.get("max_iter", 1000),
            )

        return super().fit(
            activation_paths,
            trajectories,
            labels,
            num_epochs,
            lr=lr,
            l2_penalty=l2_penalty,
            **kwargs,
        )

    def get_activations(
        self, trajectories: List[List[Any]], layers: List[int] = []
    ) -> List[Tensor]:
        "First trajectory in get_activations =>", trajectories[0]
        result = super().get_activations(trajectories, layers=[self.layer])
        return result


tasks = {
    "code_rule_violation",
    "code_sabotage",
    "code_deception",
    "adversarial_code_deception",
    "adversarial_code_sabotage",
    "adversarial_code_rule_violation",
    "adversarial_only",
    "adversarial_all",
    "code_deception_on_policy",
    "code_rule_violation_on_policy",
    "code_sabotage_on_policy",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a learned attention probe.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
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
        "--use_normalization", action="store_true", help="Whether to use layer normalization"
    )
    parser.add_argument(
        "--attention_scaling",
        action="store_true",
        help="Whether to scale attention scores by sqrt(hidden_dim)",
        default=True,
    )
    parser.add_argument(
        "--initialization_scale",
        type=float,
        help="Scale factor for initial weights (higher values = stronger initialization)",
        default=2.0,
    )
    parser.add_argument(
        "--min_lr", type=float, help="Minimum learning rate for sweep", default=8e-4
    )
    parser.add_argument(
        "--max_lr", type=float, help="Maximum learning rate for sweep", default=3e-3
    )
    parser.add_argument(
        "--num_lr_steps", type=int, help="Number of learning rates to try in sweep", default=4
    )
    parser.add_argument(
        "--on_policy_model_organism", choices=[True, False], type=bool, default=False
    )
    args = parser.parse_args()
    out_path = f"./{args.train_task}__{args.eval_task}/out.pkl"

    model = args.model
    assert args.train_task in tasks, f"Invalid train task: {args.train_task}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    monitor_dir = os.path.dirname(out_path)

    attention_probe = LearnedAttentionProbe(
        model_name=model,
        monitor_dir=monitor_dir,
        inference_batch_size_per_device=1,
        dtype=torch.bfloat16,
        layer=args.layer,
        use_normalization=args.use_normalization,
        attention_scaling=args.attention_scaling,
        initialization_scale=args.initialization_scale,
        on_policy_model_organism=args.on_policy_model_organism,
    )

    # Replace train() with learning_rate_sweep()
    best_lr = attention_probe.learning_rate_sweep(
        args.train_path,
        clear_activation_cache=True,
        num_epochs=args.num_epochs,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_lr_steps,  # replace these three with args.lr to call train()
        # lr=args.lr,
        optimizer_type="AdamW",
        max_grad_norm=10.0,
        min_grad_norm=1e-4,
    )

    if attention_probe.accelerator.is_main_process:
        print(f"Best learning rate from sweep: {best_lr}")
        attention_probe.save_model_params(os.path.join(monitor_dir, "model_params.pt"))

    # Evaluate on the training set
    if args.eval_on_trainset and attention_probe.accelerator.is_main_process:
        print(f"Evaluating on the training set: {args.train_path} ...")
        train_metrics = attention_probe.evaluate(
            args.train_path,
            out_path=os.path.join(monitor_dir, "train_metrics.json"),
            clear_activation_cache=True,
        )
        print(
            "Training set evaluation metrics:",
            train_metrics.accuracy,
            train_metrics.tpr_at_fpr,
            train_metrics.auc,
        )
    monitor_predictions = attention_probe.evaluate(
        args.eval_path,
        out_path=args.out_path,
        clear_activation_cache=True,
    )

    dist.destroy_process_group()
