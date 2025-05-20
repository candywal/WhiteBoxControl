import json
import os
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Dict, Tuple

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON to {path}")

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded JSON from {path}")
    return data

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_hf_datasets(
    safe_dataset_name: str,
    unsafe_dataset_name: str,
    safe_config_name: str = None,
    unsafe_config_name: str = None,
    split: str = "train",
    cache_dir: str = None
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    """
    Loads safe and unsafe datasets from Hugging Face, combines them,
    and extracts messages and labels.

    Assumes datasets have 'messages' (list of dicts with 'role', 'content')
    and 'classification' (string like "safe" or "unsafe", or 0/1 int) fields.
    """
    print(f"Loading safe dataset: {safe_dataset_name} (config: {safe_config_name}, split: {split})")
    safe_ds = load_dataset(safe_dataset_name, name=safe_config_name, split=split, cache_dir=cache_dir)
    print(f"Loading unsafe dataset: {unsafe_dataset_name} (config: {unsafe_config_name}, split: {split})")
    unsafe_ds = load_dataset(unsafe_dataset_name, name=unsafe_config_name, split=split, cache_dir=cache_dir)

    all_messages = []
    all_labels = []

    def process_dataset(dataset, label_value, label_str):
        count = 0
        for item in dataset:
            if "messages" not in item or "classification" not in item:
                print(f"Warning: Item missing 'messages' or 'classification' field in {label_str} dataset. Skipping.")
                continue
            
            # Ensure messages is a list of dicts
            if not isinstance(item["messages"], list) or \
               not all(isinstance(turn, dict) and "role" in turn and "content" in turn for turn in item["messages"]):
                print(f"Warning: Item in {label_str} dataset has malformed 'messages' field. Skipping: {item['messages'][:50]}")
                continue

            all_messages.append(item["messages"])
            
            # Standardize classification to 0 or 1
            if isinstance(item["classification"], str):
                current_label = 1 if item["classification"].lower() == "unsafe" else 0
            elif isinstance(item["classification"], (int, float)):
                current_label = 1 if int(item["classification"]) == 1 else 0 # Assuming 1 is unsafe
            else:
                print(f"Warning: Unknown classification format {item['classification']}. Defaulting to safe (0).")
                current_label = 0

            if current_label != label_value:
                # This can happen if a dataset labeled "safe_dataset_name" contains items marked "unsafe"
                # Or if the dataset uses different label conventions.
                # For now, we trust the dataset name implies the majority label if not explicit.
                pass # Or print a warning
            all_labels.append(label_value)
            count +=1
        print(f"Processed {count} items from {label_str} dataset.")

    process_dataset(safe_ds, 0, safe_dataset_name)
    process_dataset(unsafe_ds, 1, unsafe_dataset_name)
    
    if not all_messages:
        raise ValueError("No valid data found after processing datasets. Check dataset formats and field names.")

    return all_messages, all_labels

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, messages: List[List[Dict[str, str]]], labels: List[int], tokenizer, max_length: int = 512):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenizer.apply_chat_template handles list of messages directly
        # For a single conversation (item in the batch)
        tokenized_inputs = self.tokenizer.apply_chat_template(
            self.messages[idx],
            tokenize=True,
            add_generation_prompt=False, # Usually False for classification on existing text
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length" # Pad to max_length for consistent tensor shapes in batch
        )
        # Squeeze to remove batch dimension added by return_tensors="pt" for single item
        input_ids = tokenized_inputs["input_ids"].squeeze(0)
        attention_mask = tokenized_inputs["attention_mask"].squeeze(0)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(self.labels[idx], dtype=torch.float)}