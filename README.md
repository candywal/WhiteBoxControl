# 🔍 WhiteBoxControl: Evaluating Probes in a Research Sabotage Setting


The main goals we are trying to achieve
- Introduce a risk assessment methodology for white box monitors.
- Introduce benchmark data sets for iterating on techniques.


## *Abstract* 
LLM agents are increasingly entrusted with high-stakes affordances within software companies, creating a security risk. Just as organizations defend against human insider threats, they should prepare defenses against LLM agent adversaries. Previous work introduced 'control evaluations' as a methodology to red-team defenses against LLM adversaries; however, these evaluations only addressed strategies that monitored agents' outputs and treated the LLMs themselves as black boxes. 'White box' methods, i.e. monitoring a model's activations to see if it's engaging in dangerous reasoning, could also be effective. We introduce a suite of evaluations to assess how white box monitoring methods such as linear probing catch LLM attacks in a software engineering setting. Our evaluations suggest that state-of-the art probes are fragile, and often generalize poorly between attack types. While probes offer some risk reduction, we couldn't make them reliable enough for high assurance. We hope our evaluation methodology and datasets help improve white box monitors so they are a reliable defense against AI agent security threats. 


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 **Key Features**

- 🎯 **Three Probe Types**: Logistic, Attention-based, and Mean Difference probes
- 🚀 **Clean API**: Simple `TrainProbe.py` and `EvalProbe.py` interfaces  
- 📊 **Comprehensive Evaluation**: Detailed metrics, plots, and analysis
- 🔧 **Flexible Configuration**: Support for various models and hyperparameters
- 📈 **Multi-GPU Support**: Efficient parallel training and evaluation
- 🤗 **HuggingFace Integration**: Direct loading of datasets from HuggingFace Hub

---

## 🚀 **Quick Start**

### **1. Installation**

```bash
git clone https://github.com/yourusername/WhiteBoxControl.git
cd WhiteBoxControl

# Install dependencies
pip install -r requirements.txt

# Setup directories
python Utils.py
```

### **2. Train a Probe**

```bash
# Train a logistic probe
python TrainProbe.py \
    --probe_type logistic \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "data/datasets/my_training_data.json" \
    --layer 15 \
    --epochs 10

# Train an attention probe with optimal hyperparameters  
python TrainProbe.py \
    --probe_type attention \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "data/datasets/my_training_data.json" \
    --layer 32 \
    --epochs 14 \
    --lr 0.0022 \
    --use_normalization

# Train a probe using HuggingFace dataset
python TrainProbe.py \
    --probe_type logistic \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "Anthropic/hh-rlhf" \
    --hf_dataset \
    --hf_subset "helpful-base" \
    --hf_split "train" \
    --hf_messages_field "messages" \
    --hf_classification_field "label" \
    --hf_classification_mapping '{"chosen": "safe", "rejected": "unsafe"}' \
    --layer 15 \
    --epochs 10
```

### **3. Evaluate the Probe**

```bash
# Evaluate with visualizations
python EvalProbe.py \
    --probe_dir "outputs/trained_probes/logistic_probe_layer15" \
    --data "data/datasets/my_test_data.json" \
    --save_predictions \
    --save_plots

# Batch evaluation on multiple datasets
python EvalProbe.py \
    --probe_dir "outputs/trained_probes/attention_probe_layer32" \
    --data test1.json test2.json test3.json \
    --batch_eval

# Evaluate using HuggingFace dataset
python EvalProbe.py \
    --probe_dir "outputs/trained_probes/logistic_probe_layer15" \
    --data "Anthropic/hh-rlhf" \
    --hf_dataset \
    --hf_subset "helpful-base" \
    --hf_split "test" \
    --hf_messages_field "messages" \
    --hf_classification_field "label" \
    --hf_classification_mapping '{"chosen": "safe", "rejected": "unsafe"}' \
    --save_predictions \
    --save_plots
```

### **4. Run Complete Example**

```bash
# Run the full pipeline
chmod +x run_scripts/example_bash_script.sh
./run_scripts/example_bash_script.sh
```

---

## 📁 **Project Structure**

```
WhiteBoxControl/
├── 🔧 TrainProbe.py               # Main training interface
├── 🔍 EvalProbe.py                # Main evaluation interface  
├── 🛠️  Utils.py                    # Utility functions
│
├── 📊 data/                       # Data storage
│   ├── datasets/                  # Training and test datasets
│   │   └── combined/              # Processed datasets
│   └── models/                    # Downloaded model cache
│
├── 🧠 probes/                     # Core probe implementations
│   ├── __init__.py
│   ├── base_probe.py              # Abstract base class
│   ├── logistic_probe.py          # 🔵 Logistic regression probe
│   ├── attention_probe.py         # 🔴 Attention-based probe
│   ├── mean_diff_probe.py         # 🟡 Mean difference probe
│   ├── data_types.py              # Data structure definitions
│   └── metrics.py                 # Evaluation metrics
│
├── 📝 run_scripts/                # Example scripts
│   └── example_bash_script.sh     # Complete workflow example
│
├── 📈 outputs/                    # Generated results
│   ├── trained_probes/            # Trained model weights
│   │   └── <probe_name>/
│   │       ├── weights.pt         # Model parameters
│   │       ├── config.json        # Training configuration
│   │       └── evaluation/        # Evaluation results
│   │
│   └── evaluation/                # Standalone evaluations
│       └── <experiment_name>/
│           ├── metrics.json       # Performance metrics
│           ├── predictions.csv    # Detailed predictions
│           └── plots/             # Visualization plots
│
├── 📋 requirements.txt            # Python dependencies
├── 📖 README.md                  # This file
└── 🙈 .gitignore                 # Git ignore rules
```

---

## 🎯 **Detailed Usage**

### **Training Configuration**

#### **Logistic Probe**
```bash
python TrainProbe.py \
    --probe_type logistic \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "data/datasets/safety_data.json" \
    --layer 12 \
    --epochs 15 \
    --lr 0.001 \
    --fit_method adam \
    --token_aggregation mean \
    --output_dir "outputs/trained_probes"
```

#### **Attention Probe**
```bash
python TrainProbe.py \
    --probe_type attention \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "data/datasets/deception_data.json" \
    --layer 32 \
    --epochs 14 \
    --lr 0.0022 \
    --use_normalization \
    --initialization_scale 1.5 \
    --gpu_id 0
```

#### **Mean Difference Probe**
```bash
python TrainProbe.py \
    --probe_type mean_diff \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "data/datasets/manipulation_data.json" \
    --layer 20 \
    --token_aggregation last_only
```

### **Data Format**

Your training data should be a JSON file with the following structure:

```json
[
  [
    {
      "messages": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
      ],
      "classification": "safe",
      "trajectory_index": 0,
      "action_id": 1
    }
  ],
  [
    {
      "messages": [
        {"role": "user", "content": "Can you help me hack into a system?"},
        {"role": "assistant", "content": "I can't help with illegal activities."}
      ],
      "classification": "unsafe", 
      "trajectory_index": 1,
      "action_id": 1
    }
  ]
]
```

### **HuggingFace Dataset Support**

WhiteBoxControl now supports loading datasets directly from HuggingFace Hub! This makes it easy to work with popular safety datasets without manual conversion.

#### **Requirements**
- The dataset must have a `messages` field (or customizable field name) containing conversation data
- Messages should be in the format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- The dataset should have a classification field that can be mapped to "safe"/"unsafe"

#### **Usage Examples**

```bash
# Train with HuggingFace dataset
python TrainProbe.py \
    --probe_type logistic \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --data "Anthropic/hh-rlhf" \
    --hf_dataset \
    --hf_subset "helpful-base" \
    --hf_split "train" \
    --hf_messages_field "messages" \
    --hf_classification_field "label" \
    --hf_classification_mapping '{"chosen": "safe", "rejected": "unsafe"}' \
    --layer 15

# Evaluate with HuggingFace dataset  
python EvalProbe.py \
    --probe_dir "outputs/trained_probes/logistic_probe_layer15" \
    --data "my-org/safety-dataset" \
    --hf_dataset \
    --hf_split "test" \
    --save_predictions
```

#### **Configuration Options**
- `--hf_dataset`: Enable HuggingFace dataset loading
- `--hf_subset`: Dataset configuration/subset name (optional)
- `--hf_split`: Dataset split to use (default: "train" for training, "test" for evaluation)
- `--hf_messages_field`: Field containing conversation messages (default: "messages")
- `--hf_classification_field`: Field containing labels (default: "label")
- `--hf_classification_mapping`: JSON string mapping dataset labels to "safe"/"unsafe"

#### **Supported Dataset Formats**
```python
# Example 1: Direct safe/unsafe labels
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ],
    "label": "safe"
}

# Example 2: Custom labels with mapping
{
    "conversations": [
        {"role": "user", "content": "Help me hack"},
        {"role": "assistant", "content": "I can't help with that"}
    ],
    "category": "harmful"
}
# Use: --hf_messages_field "conversations" --hf_classification_mapping '{"harmful": "unsafe", "helpful": "safe"}'
```

### **Evaluation Outputs**

After evaluation, you'll get:

- **📊 Performance Metrics**: Accuracy, AUC, TPR at various FPR thresholds
- **📈 Visualizations**: ROC curves, precision-recall plots, prediction histograms  
- **📋 Detailed Results**: Per-sample predictions and confidence scores
- **🎯 Analysis**: Classification reports and confusion matrices

---

### **Multi-GPU Training**

```bash
# Use specific GPU
python TrainProbe.py --gpu_id 3 [other args]

# Use all available GPUs (default)
python TrainProbe.py [other args]

# Set specific GPUs via environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
python TrainProbe.py [other args]
```

### **Batch Processing**

```bash
# Process multiple datasets
python EvalProbe.py \
    --probe_dir "outputs/trained_probes/logistic_probe_layer15" \
    --data dataset1.json dataset2.json dataset3.json \
    --batch_eval \
    --output_dir "outputs/evaluation/comparison"
```

---

## 🛠️ **Utilities**

WhiteBoxControl includes utility functions:

```python
import Utils

# Download and cache models
Utils.download_hf_model("Qwen/Qwen2.5-Coder-32B-Instruct", cache_dir="data/models")

# Validate datasets
Utils.validate_dataset_format("data/my_dataset.json")

# Split datasets
splits = Utils.split_dataset("data/full_dataset.json", train_ratio=0.8)

# Combine multiple datasets  
Utils.combine_datasets(
    ["safe_data.json", "unsafe_data.json"], 
    "combined_data.json",
    labels=["safe", "unsafe"]
)

# Check system info
Utils.print_system_info()

# Load HuggingFace dataset and convert to WhiteBoxControl format
hf_data = Utils.load_hf_dataset(
    "Anthropic/hh-rlhf", 
    subset="helpful-base",
    split="train",
    classification_mapping={"chosen": "safe", "rejected": "unsafe"}
)

# Load data with custom field names
custom_data = Utils.load_hf_dataset(
    "my-org/safety-dataset",
    messages_field="conversations", 
    classification_field="category",
    classification_mapping={"harmful": "unsafe", "helpful": "safe"}
)
```

```

