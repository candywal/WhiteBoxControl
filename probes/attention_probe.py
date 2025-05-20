import torch
from torch import nn, Tensor
import math

from .base_probe import BaseProbe


class AttentionProbeParameters(nn.Module):
    def __init__(self, hidden_size: int, use_normalization: bool = False):
        super().__init__()
        self.use_normalization = use_normalization
        if use_normalization:
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.norm = nn.Identity()

        self.score_vector_module = nn.Linear(hidden_size, 1, bias=False)
        self.value_vector_module = nn.Linear(hidden_size, 1, bias=False)

        nn.init.xavier_uniform_(self.score_vector_module.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value_vector_module.weight, gain=1.0)

class LearnedAttentionProbe(BaseProbe):
    name = "learned_attention_probe"

    def __init__(
        self,
        base_model_name: str,
        layer: int,
        use_normalization: bool = False,
        attention_scaling: bool = True,
        initialization_scale: float = 1.0, # Applied after Xavier
        temperature: float = 2.0, # For attention softmax
        **kwargs,
    ):
        current_probe_specific_kwargs = {
            "use_normalization": use_normalization,
            "attention_scaling": attention_scaling,
            "initialization_scale": initialization_scale,
            "temperature": temperature,
        }
        super().__init__(
            base_model_name=base_model_name,
            layer=layer,
            probe_type=self.name,
            probe_specific_kwargs=current_probe_specific_kwargs,
            **kwargs
        )
        self.use_normalization = use_normalization
        self.attention_scaling = attention_scaling
        self.initialization_scale = initialization_scale
        self.temperature = temperature
        self.init_params()

    def init_params(self):
        self.params = AttentionProbeParameters(self.hidden_size, self.use_normalization)
        if self.initialization_scale != 1.0:
            with torch.no_grad():
                for param_set in [self.params.score_vector_module.parameters(), self.params.value_vector_module.parameters()]:
                    for param in param_set:
                        param.data.mul_(self.initialization_scale)
    
    def forward(self, batch_activations: torch.Tensor) -> torch.Tensor:
        # batch_activations: (batch_size, seq_len, hidden_size)
        
        normalized_activations = self.params.norm(batch_activations) # (batch_size, seq_len, hidden_size)
        
        scores = self.params.score_vector_module(normalized_activations) # (batch_size, seq_len, 1)
        
        if self.attention_scaling:
            scaling_factor = math.sqrt(self.hidden_size)
            scores = scores / scaling_factor
            
        # Stability for softmax: (batch_size, seq_len, 1)
        scores = scores - scores.max(dim=1, keepdim=True).values 
        
        # Attention weights: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(scores / self.temperature, dim=1) 
        
        values = self.params.value_vector_module(normalized_activations) # (batch_size, seq_len, 1)
        
        # Weighted sum of values: (batch_size, 1)
        # (attention_weights * values) is element-wise product, then sum over seq_len
        final_logits = (attention_weights * values).sum(dim=1) 
        
        return final_logits.squeeze(-1) # (batch_size,)