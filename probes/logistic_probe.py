import torch
from torch import nn, Tensor

from .base_probe import BaseProbe

class LogisticProbe(BaseProbe):
    name = "logistic_probe"

    def __init__(
        self,
        # Args for BaseProbe constructor
        base_model_name: str,
        layer: int,
        # Probe-specific args for its own __init__ and for probe_specific_kwargs in BaseProbe
        token_position_aggregation_strategy: str = "last_only", # mean, soft_max, last_only
        # Pass-through args for BaseProbe
        **kwargs, # Catches device_str, dtype_str, output_dir_for_saving etc.
    ):
        # Prepare probe_specific_kwargs for BaseProbe
        # These are the args unique to this probe's __init__ signature
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
        self.init_params() # Called after super().__init__ sets up hidden_size

    def init_params(self):
        # self.hidden_size is available from BaseProbe after super().__init__
        self.params = nn.Linear(self.hidden_size, 1)
        # No need to move to device here, BaseProbe.train_on_data or load_probe will handle it via Accelerator

    def forward(self, batch_activations: torch.Tensor) -> torch.Tensor:
        # batch_activations shape: (batch_size, seq_len, hidden_size)
        # self.params is the nn.Linear layer
        
        # Aggregate activations along sequence length dimension
        if self.token_position_aggregation_strategy == "last_only":
            # This needs careful handling if sequences have padding.
            # Assuming activations are for non-padded tokens or caller handles masking.
            # For now, simple last token. A more robust way involves attention_mask.
            # If using max_length padding in ProbeDataset, actual sequence lengths vary.
            # A common approach is to average non-padded token activations or use CLS token if applicable.
            # Let's assume for now the "last" meaningful token.
            # If `tokenizer.apply_chat_template` gives proper attention masks, we can find the last non-pad token.
            # For simplicity, taking the last token from the fixed sequence length used in get_activations.
            aggregated_activations = batch_activations[:, -1, :] # (batch_size, hidden_size)
        elif self.token_position_aggregation_strategy == "mean":
            # This should ideally mask out padding tokens.
            # (batch_size, hidden_size)
            aggregated_activations = batch_activations.mean(dim=1) 
        elif self.token_position_aggregation_strategy == "soft_max":
            # Softmax over sequence length for each feature, then weighted sum?
            # Or softmax over token scores if we had them.
            # A simple interpretation: weighted average based on token norms or a simple linear layer.
            # For now, let's treat it like mean for simplicity or implement a more specific version.
            # Original used softmax on sigmoid(logits), this is different.
            # Let's do mean pooling, and user can extend if a more complex softmax agg is needed.
            weights = torch.softmax(batch_activations.norm(dim=2, keepdim=True) / 0.1, dim=1) # (batch_size, seq_len, 1)
            aggregated_activations = (batch_activations * weights).sum(dim=1) # (batch_size, hidden_size)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.token_position_aggregation_strategy}")

        logits = self.params(aggregated_activations) # (batch_size, 1)
        return logits.squeeze(-1) # (batch_size,)