import torch
import torch.nn as nn
import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from Utils import ProbeDataset # Assuming Utils.py is accessible

class BaseProbe(nn.Module, ABC):
    name: str = "base_probe" # Subclasses should override this

    def __init__(
        self,
        base_model_name: str,
        layer: int,
        probe_type: str, # Will be cls.name from the specific probe class
        probe_specific_kwargs: Dict = None, # For probe-specific __init__ args
        # Common params
        device_str: Optional[str] = None, # e.g. "cuda", "cpu"
        dtype_str: str = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        # For saving/loading
        output_dir_for_saving: Optional[str] = None, # Used when saving the probe
    ):
        super().__init__()
        self.accelerator = Accelerator()
        self._device_override = torch.device(device_str) if device_str else self.accelerator.device
        self.torch_dtype = getattr(torch, dtype_str)

        self.base_model_name = base_model_name
        self.layer = layer
        self.probe_type_name = probe_type
        self.probe_specific_kwargs = probe_specific_kwargs if probe_specific_kwargs else {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model_config = AutoConfig.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.hidden_size = self.base_model_config.hidden_size

        self.base_model = None # Loaded on demand
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self.params: nn.Module | None = None # Probe's learnable parameters

        # For saving/loading config
        self.probe_config_args_to_save = {
            "base_model_name": self.base_model_name,
            "layer": self.layer,
            "probe_type": self.probe_type_name,
            "dtype_str": dtype_str,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "probe_specific_kwargs": self.probe_specific_kwargs,
        }
        self.output_dir_for_saving = output_dir_for_saving


    @property
    def device(self):
        return self._device_override # Accelerator handles device placement for model and data

    def _init_base_model_if_needed(self):
        if self.base_model is None:
            if self.accelerator.is_main_process:
                print(f"Initializing base model: {self.base_model_name} on device: {self.device}")
            
            quantization_config = None
            if self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype, # e.g. torch.bfloat16
                    bnb_4bit_quant_type="nf4", # Recommended
                    bnb_4bit_use_double_quant=True, # Recommended
                )

            # Model loaded on CPU first if quantization is used, then moved by accelerator
            model_load_device = "auto" # Let transformers decide, or use accelerator.device
            if quantization_config:
                 # For BNB, often good to load on CPU then let Accelerator handle final placement
                 # Or use device_map="auto"
                 pass


            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=self.torch_dtype,
                quantization_config=quantization_config,
                device_map=model_load_device if not quantization_config else None, # device_map='auto' or specific device
                trust_remote_code=True
            )
            
            if not quantization_config: # If not quantized, manually move to device
                self.base_model.to(self.device)

            self.base_model = self.accelerator.prepare(self.base_model)
            self.base_model.eval() # Probes operate on frozen base model
            if self.accelerator.is_main_process:
                print(f"Base model {self.base_model_name} initialized and prepared with Accelerator.")


    @torch.no_grad()
    def get_activations(self, batch_input_ids: torch.Tensor, batch_attention_mask: torch.Tensor) -> torch.Tensor:
        self._init_base_model_if_needed()
        self.base_model.eval() # Ensure model is in eval mode

        # We need to get to the specific layer module. Path might vary.
        # Common paths: self.base_model.model.layers[idx] (Llama, Mistral, GPT-NeoX)
        # Or self.base_model.transformer.h[idx] (GPT2)
        # This needs to be robust or configurable if supporting many model types.
        # For now, assuming a common structure like Llama/Mistral:
        try:
            target_layer_module = self.base_model.module.model.layers[self.layer] if self.accelerator.num_processes > 1 else self.base_model.model.layers[self.layer]
        except AttributeError:
            try:
                target_layer_module = self.base_model.module.transformer.h[self.layer] if self.accelerator.num_processes > 1 else self.base_model.transformer.h[self.layer]
            except AttributeError as e:
                raise AttributeError(f"Could not access layer {self.layer}. Unsupported model architecture or incorrect layer index. Error: {e}")


        activations_output = None
        def hook_fn(module, input, output):
            nonlocal activations_output
            # Output can be a tuple (hidden_states, optional_other_stuff)
            activations_output = output[0].detach()

        hook = target_layer_module.register_forward_hook(hook_fn)
        
        # Forward pass
        # The model might be wrapped by Accelerator (e.g. DDP, FSDP)
        # Accelerator's prepared model handles the forward pass correctly.
        _ = self.base_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        hook.remove()

        if activations_output is None:
            raise RuntimeError(f"Failed to capture activations from layer {self.layer}.")
        
        # activations_output shape: (batch_size, seq_len, hidden_size)
        return activations_output #.to(self.device) # Ensure on correct device, though hook should capture it on model's device

    @abstractmethod
    def init_params(self):
        """Initialize probe-specific parameters (self.params). Called in __init__ of subclass."""
        pass

    @abstractmethod
    def forward(self, batch_activations: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of activations and return probe predictions
        (logits or probabilities).
        Input shape: (batch_size, seq_len, hidden_size)
        Output shape: (batch_size,) for logits/probabilities
        """
        pass

    def train_on_data(
        self,
        train_messages: List[List[Dict[str, str]]],
        train_labels: List[int],
        val_messages: Optional[List[List[Dict[str, str]]]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 8,
        l2_penalty: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer_type: str = "AdamW", # AdamW or SGD
        max_token_length: int = 512,
        label_smoothing: float = 0.0,
    ):
        if self.params is None:
            self.init_params() # Ensure probe params are initialized
        self.params = self.accelerator.prepare(self.params) # Prepare probe params with accelerator

        if optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.params.parameters(), lr=lr, weight_decay=l2_penalty)
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(self.params.parameters(), lr=lr, weight_decay=l2_penalty)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
        
        optimizer = self.accelerator.prepare(optimizer)

        train_dataset = ProbeDataset(train_messages, train_labels, self.tokenizer, max_token_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloader = self.accelerator.prepare(train_dataloader)

        val_dataloader = None
        if val_messages and val_labels:
            val_dataset = ProbeDataset(val_messages, val_labels, self.tokenizer, max_token_length)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            val_dataloader = self.accelerator.prepare(val_dataloader)

        criterion = nn.BCEWithLogitsLoss() # Assumes self.forward() returns logits

        for epoch in range(epochs):
            self.params.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", disable=not self.accelerator.is_local_main_process)
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"] #.to(self.device) -> Accelerator handles this
                attention_mask = batch["attention_mask"] #.to(self.device)
                labels = batch["labels"].unsqueeze(1) #.to(self.device) # Shape (batch_size, 1) for BCEWithLogitsLoss

                batch_activations = self.get_activations(input_ids, attention_mask)
                
                # Pass activations to the probe's forward method
                # Ensure self.params (the probe itself) is on the correct device
                # If self.params is a simple nn.Linear, it should be fine.
                # If it's more complex, ensure it's moved. Accelerator prepares it.
                logits = self.forward(batch_activations) # Expected (batch_size, 1) or (batch_size,)

                if logits.ndim == 1: # Ensure (batch_size, 1) for BCEWithLogitsLoss
                    logits = logits.unsqueeze(1)

                # Apply label smoothing
                if label_smoothing > 0:
                    smoothed_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                else:
                    smoothed_labels = labels
                
                loss = criterion(logits, smoothed_labels)
                
                optimizer.zero_grad()
                self.accelerator.backward(loss)
                if max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.params.parameters(), max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(train_dataloader)
            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            if val_dataloader:
                self.params.eval()
                total_val_loss = 0
                all_val_preds = []
                all_val_labels = []
                val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", disable=not self.accelerator.is_local_main_process)
                with torch.no_grad():
                    for batch in val_progress_bar:
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        labels = batch["labels"].unsqueeze(1)

                        batch_activations = self.get_activations(input_ids, attention_mask)
                        logits = self.forward(batch_activations)
                        if logits.ndim == 1: logits = logits.unsqueeze(1)
                        
                        loss = criterion(logits, labels) # Use original labels for val loss
                        total_val_loss += loss.item()
                        
                        # Gather predictions and labels for metrics
                        # Accelerator.gather works on tensors from all processes
                        gathered_preds = self.accelerator.gather(torch.sigmoid(logits)) # Probabilities
                        gathered_labels = self.accelerator.gather(labels)
                        all_val_preds.append(gathered_preds.cpu())
                        all_val_labels.append(gathered_labels.cpu())
                
                avg_val_loss = total_val_loss / len(val_dataloader)
                if self.accelerator.is_local_main_process:
                    all_val_preds_cat = torch.cat(all_val_preds).numpy().flatten()
                    all_val_labels_cat = torch.cat(all_val_labels).numpy().flatten()
                    val_auc = roc_auc_score(all_val_labels_cat, all_val_preds_cat)
                    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation AUC: {val_auc:.4f}")

    @torch.no_grad()
    def evaluate_on_data(self, eval_messages: List[List[Dict[str, str]]], eval_labels: List[int], batch_size: int = 8, max_token_length: int = 512):
        if self.params is None:
            raise RuntimeError("Probe parameters not loaded or initialized. Call load_probe() or train_on_data() first.")
        self.params.eval() # Ensure probe is in eval mode
        self.params = self.accelerator.prepare(self.params) # Ensure consistency if not already done

        eval_dataset = ProbeDataset(eval_messages, eval_labels, self.tokenizer, max_token_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        all_preds_probs = []
        all_true_labels = []

        progress_bar = tqdm(eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process)
        for batch in progress_bar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"] # Already 1D

            batch_activations = self.get_activations(input_ids, attention_mask)
            logits = self.forward(batch_activations) # Expected (batch_size,) or (batch_size,1)
            
            probabilities = torch.sigmoid(logits).squeeze()

            # Gather predictions and labels from all processes
            gathered_probs = self.accelerator.gather(probabilities)
            gathered_labels = self.accelerator.gather(labels)

            all_preds_probs.append(gathered_probs.cpu())
            all_true_labels.append(gathered_labels.cpu())

        all_preds_probs_cat = torch.cat(all_preds_probs).numpy()
        all_true_labels_cat = torch.cat(all_true_labels).numpy()
        
        metrics = {}
        if self.accelerator.is_local_main_process: # Calculate metrics only on main process with all data
            metrics["auc"] = roc_auc_score(all_true_labels_cat, all_preds_probs_cat)
            predicted_classes = (all_preds_probs_cat > 0.5).astype(int)
            metrics["accuracy"] = accuracy_score(all_true_labels_cat, predicted_classes)
            
            # TPR at FPR (simplified from MonitorPredictions)
            fpr_full, tpr_full, thresholds = roc_curve(all_true_labels_cat, all_preds_probs_cat)
            metrics["tpr_at_fpr"] = {}
            for target_fpr_val in [0.01, 0.05, 0.1]:
                try:
                    idx = np.where(fpr_full >= target_fpr_val)[0][0]
                    metrics["tpr_at_fpr"][f"tpr_at_{target_fpr_val*100:.0f}_fpr"] = tpr_full[idx]
                except IndexError: # Target FPR not reached
                    metrics["tpr_at_fpr"][f"tpr_at_{target_fpr_val*100:.0f}_fpr"] = tpr_full[-1] if len(tpr_full)>0 else 0.0
            
            metrics_obj = {
                "labels": all_true_labels_cat.tolist(),
                "predicted_probabilities": all_preds_probs_cat.tolist(),
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "tpr_at_fpr": metrics["tpr_at_fpr"]
            }
            return metrics_obj
        return None # Other processes return None

    def save_probe(self, output_dir: Optional[str] = None):
        output_dir_to_use = output_dir or self.output_dir_for_saving
        if not output_dir_to_use:
            raise ValueError("output_dir must be provided either at init or during save_probe call.")
        
        self.accelerator.wait_for_everyone() # Ensure all processes are synchronized
        if self.accelerator.is_main_process:
            os.makedirs(output_dir_to_use, exist_ok=True)
            weights_path = os.path.join(output_dir_to_use, "probe_weights.pt")
            config_path = os.path.join(output_dir_to_use, "config.json")

            # Save unwrapped model parameters
            unwrapped_params = self.accelerator.unwrap_model(self.params)
            torch.save(unwrapped_params.state_dict(), weights_path)
            
            # Update config with actual output_dir if it was passed to save_probe
            config_to_save = self.probe_config_args_to_save.copy()
            # config_to_save["output_dir_for_saving"] = output_dir_to_use # Not needed, implied by where it's saved

            with open(config_path, "w") as f:
                json.dump(config_to_save, f, indent=4)
            print(f"Probe saved to {output_dir_to_use}")

    @classmethod
    def load_probe(cls, probe_dir: str, device_str: Optional[str] = None, **cli_override_kwargs):
        config_path = os.path.join(probe_dir, "config.json")
        weights_path = os.path.join(probe_dir, "probe_weights.pt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Probe config file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Probe weights file not found: {weights_path}")

        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        # Override saved config with any CLI args if necessary
        saved_config.update(cli_override_kwargs)
        
        # probe_specific_kwargs should be extracted correctly
        probe_specific_kwargs = saved_config.pop("probe_specific_kwargs", {})

        # Instantiate the probe
        instance = cls(
            base_model_name=saved_config["base_model_name"],
            layer=saved_config["layer"],
            probe_type=saved_config["probe_type"], # This is cls.name
            probe_specific_kwargs=probe_specific_kwargs,
            device_str=device_str or saved_config.get("device_str"), # Prioritize call-time device
            dtype_str=saved_config["dtype_str"],
            load_in_8bit=saved_config.get("load_in_8bit", False),
            load_in_4bit=saved_config.get("load_in_4bit", False),
            output_dir_for_saving=probe_dir # For potential re-saving
        )
        
        instance.init_params() # Initialize self.params structure
        
        # Load state dict
        # Important: self.params needs to be on the CPU before loading state_dict if device_map was used for base model
        # However, for the probe itself, it's usually small enough.
        # Load weights onto the instance's designated device.
        state_dict = torch.load(weights_path, map_location=instance.device)
        instance.params.load_state_dict(state_dict)
        instance.params.to(instance.device) # Ensure it's on the correct device
        instance.params.eval()

        if instance.accelerator.is_main_process:
            print(f"Probe {cls.name} loaded from {probe_dir} on device {instance.device}")
        return instance