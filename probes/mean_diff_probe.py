import torch
from torch import nn, Tensor
from typing import List, Dict
from .base_probe import BaseProbe # Assuming BaseProbe is in the same directory or accessible
from Utils import ProbeDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ParameterHolder for MeanDiffProbe's direction vector
class MeanDiffDirectionVector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=False) # Direction vector

    def forward(self, x): # Not used for direct forward pass, but to hold the parameter
        return x @ self.weight # Project activations

class MeanDiffProbe(BaseProbe):
    name = "mean_diff_probe"

    def __init__(
        self,
        base_model_name: str,
        layer: int,
        token_position_aggregation_strategy: str = "last_only",
        **kwargs,
    ):
        current_probe_specific_kwargs = {
            "token_position_aggregation_strategy": token_position_aggregation_strategy,
        }
        super().__init__(
            base_model_name=base_model_name,
            layer=layer,
            probe_type=self.name,
            probe_specific_kwargs=current_probe_specific_kwargs,
            **kwargs
        )
        self.token_position_aggregation_strategy = token_position_aggregation_strategy
        self.class_means_for_direction: Dict[int, Tensor] = {} # Stores computed means for 0 and 1
        self.init_params()

    def init_params(self):
        # self.params will hold the direction vector as a non-trainable parameter
        self.params = MeanDiffDirectionVector(self.hidden_size)
        # self.params.to(self.device) # Accelerator will prepare it

    def _aggregate_activations(self, batch_activations: torch.Tensor) -> torch.Tensor:
        # batch_activations shape: (batch_size, seq_len, hidden_size)
        if self.token_position_aggregation_strategy == "last_only":
            return batch_activations[:, -1, :]
        elif self.token_position_aggregation_strategy == "mean":
            return batch_activations.mean(dim=1)
        elif self.token_position_aggregation_strategy == "soft_max":
             weights = torch.softmax(batch_activations.norm(dim=2, keepdim=True) / 0.1, dim=1)
             return (batch_activations * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown strategy: {self.token_position_aggregation_strategy}")

    # Override train_on_data for non-gradient "fitting"
    def train_on_data(
        self,
        train_messages: List[List[Dict[str, str]]],
        train_labels: List[int],
        batch_size: int = 8, # For activation computation
        max_token_length: int = 512,
        **kwargs # Consumes other args from BaseProbe.train_on_data signature
    ):
        if self.params is None: self.init_params()
        self.params = self.accelerator.prepare(self.params)

        if self.accelerator.is_local_main_process:
            print(f"[{self.name}] 'Training' by computing class means...")

        train_dataset = ProbeDataset(train_messages, train_labels, self.tokenizer, max_token_length)
        # No shuffle needed for mean computation
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
        train_dataloader = self.accelerator.prepare(train_dataloader)

        all_aggregated_activations = []
        all_true_labels = []

        self.params.eval() # Ensure no grads, though params are already non-trainable
        with torch.no_grad():
            progress_bar = tqdm(train_dataloader, desc=f"[{self.name}] Computing Aggregated Activations", disable=not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                
                batch_activations_raw = self.get_activations(input_ids, attention_mask)
                aggregated_batch_acts = self._aggregate_activations(batch_activations_raw) # (batch_size, hidden_size)
                
                # Gather from all processes
                gathered_acts = self.accelerator.gather(aggregated_batch_acts)
                gathered_labels = self.accelerator.gather(labels)
                
                all_aggregated_activations.append(gathered_acts.cpu())
                all_true_labels.append(gathered_labels.cpu())

        if self.accelerator.is_local_main_process:
            final_aggregated_activations = torch.cat(all_aggregated_activations)
            final_labels = torch.cat(all_true_labels)

            safe_activations = final_aggregated_activations[final_labels == 0]
            unsafe_activations = final_aggregated_activations[final_labels == 1]

            if len(safe_activations) == 0 or len(unsafe_activations) == 0:
                print("Warning: One class has no samples. Mean difference probe may not be effective.")
                # Set a zero direction vector or handle as error
                self.params.weight.data.zero_()
                self.class_means_for_direction[0] = torch.zeros(self.hidden_size, device="cpu")
                self.class_means_for_direction[1] = torch.zeros(self.hidden_size, device="cpu")
            else:
                mean_safe = safe_activations.mean(dim=0)
                mean_unsafe = unsafe_activations.mean(dim=0)
                
                self.class_means_for_direction[0] = mean_safe
                self.class_means_for_direction[1] = mean_unsafe

                direction_vector = mean_unsafe - mean_safe
                if direction_vector.norm() > 1e-9:
                    direction_vector = direction_vector / direction_vector.norm()
                
                # Update the non-trainable parameter on the main process
                self.params.weight.data = direction_vector.to(self.params.weight.device, self.params.weight.dtype)
                print(f"[{self.name}] Direction vector computed with norm: {self.params.weight.data.norm().item()}")
        
        # Broadcast the computed direction vector from main process to all others
        self.accelerator.wait_for_everyone() # Ensure main process has computed it
        # self.params is already prepared by accelerator, so DDP should sync it.
        # If not using DDP, or for safety, explicitly broadcast the parameter:
        if self.accelerator.num_processes > 1:
             # Convert to a buffer that can be broadcasted by accelerate or torch.distributed
            param_tensor_list = [self.params.weight.data.clone()] # List for broadcast_object_list
            self.accelerator.broadcast_object_list(param_tensor_list, from_process=0)
            if not self.accelerator.is_main_process:
                self.params.weight.data = param_tensor_list[0].to(self.params.weight.device, self.params.weight.dtype)

        if self.accelerator.is_local_main_process:
            print(f"[{self.name}] 'Training' complete.")


    def forward(self, batch_activations: torch.Tensor) -> torch.Tensor:
        # batch_activations shape: (batch_size, seq_len, hidden_size)
        aggregated_acts = self._aggregate_activations(batch_activations) # (batch_size, hidden_size)
        
        # Project onto the direction vector (self.params.weight)
        # .weight is the Parameter in MeanDiffDirectionVector
        projections = aggregated_acts @ self.params.weight 
        return projections # These are logits/scores, sigmoid will be applied in evaluate_on_data

    # Need to customize save/load for class_means_for_direction if we want to save it.
    # The BaseProbe saves self.params.state_dict(), which for MeanDiffDirectionVector is just the weight.
    # If class_means_for_direction is needed for anything else, it should be saved/loaded separately.
    # For now, only the direction vector (self.params.weight) is crucial for forward.