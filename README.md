# 🔍 WhiteBoxControl

> **Advanced Neural Probe Training and Evaluation Framework**  
> Train and evaluate interpretability probes on transformer model activations to understand model behavior and detect potentially unsafe outputs.

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
- 🎨 **Rich Visualizations**: ROC curves, precision-recall plots, and distributions

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
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --data "data/datasets/my_training_data.json" \
    --layer 15 \
    --epochs 10

# Train an attention probe with optimal hyperparameters  
python TrainProbe.py \
    --probe_type attention \
    --model "/path/to/local/model" \
    --data "data/datasets/my_training_data.json" \
    --layer 32 \
    --epochs 14 \
    --lr 0.0022 \
    --use_normalization
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
    --model "microsoft/DialoGPT-medium" \
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
    --model "meta-llama/Llama-3.1-70B-Instruct" \
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
    --model "anthropic/claude-3-sonnet" \
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

### **Evaluation Outputs**

After evaluation, you'll get:

- **📊 Performance Metrics**: Accuracy, AUC, TPR at various FPR thresholds
- **📈 Visualizations**: ROC curves, precision-recall plots, prediction histograms  
- **📋 Detailed Results**: Per-sample predictions and confidence scores
- **🎯 Analysis**: Classification reports and confusion matrices

---

## 🔬 **Advanced Features**

### **Hyperparameter Optimization**

Based on our experiments, here are the optimal configurations:

#### **🏆 Best Attention Probe Settings**
- **Learning Rate**: 0.0022  
- **Epochs**: 14
- **Normalization**: Optional (test both)
- **Initialization Scale**: 1.5
- **Layer**: 32 (for large models)

#### **🚀 Performance Benchmarks**
| Probe Type | Accuracy | AUC | TPR@FPR=0.01 | Training Time |
|------------|----------|-----|---------------|---------------|
| Logistic | **100.0%** | **1.000** | **100.0%** | ~2 min |
| Mean Diff | **100.0%** | **1.000** | **100.0%** | ~30 sec |
| Attention | 73.5% | 0.822 | 5.0% | ~5 min |

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

WhiteBoxControl includes powerful utility functions:

```python
import Utils

# Download and cache models
Utils.download_hf_model("meta-llama/Llama-3.1-8B-Instruct", cache_dir="data/models")

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
```

---

## 📊 **Experimental Results**

### **Attention Probe Hyperparameter Study**

We conducted systematic experiments to find optimal attention probe configurations:

| Configuration | LR | Epochs | Normalization | AUC | Accuracy |
|---------------|----|---------| ------------- |-----|----------|
| **🏆 Best Overall** | 0.0022 | 14 | No | **0.8220** | 63.0% |
| **🎯 Best Accuracy** | 0.002 | 15 | Yes | 0.8180 | **73.5%** |
| **⚡ Fast & Good** | 0.0035 | 8 | Yes | 0.8020 | 58.5% |

### **Key Insights**
- **Sweet Spot**: Learning rates between 0.002-0.0035 work best
- **Normalization**: Helps on average but not always necessary
- **Training Duration**: 8-18 epochs sufficient, more isn't always better
- **Layer Choice**: Layer 32 optimal for large models (70B+)

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **❌ CUDA Out of Memory**
```bash
# Reduce batch size
python TrainProbe.py --batch_size 1 [other args]

# Use specific GPU
python TrainProbe.py --gpu_id 0 [other args]

# Clear cache
python TrainProbe.py --clear_cache [other args]
```

#### **❌ Model Not Found**
```bash
# Check model path
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('your-model-name')"

# Use local path
python TrainProbe.py --model "/absolute/path/to/model" [other args]
```

#### **❌ Data Format Error**
```bash
# Validate your dataset
python -c "import Utils; Utils.validate_dataset_format('your-data.json')"

# Add trajectory indices if missing
python -c "import Utils; Utils.add_trajectory_indices('your-data.json')"
```

---

## 🤝 **Contributing**

We welcome contributions! Please:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💚 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

### **Development Setup**

```bash
# Clone for development
git clone https://github.com/yourusername/WhiteBoxControl.git
cd WhiteBoxControl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Check code style
black . && flake8 .
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **🤗 Hugging Face** for transformer models and tokenizers
- **🔥 PyTorch** for deep learning framework  
- **📊 scikit-learn** for evaluation metrics
- **🎨 matplotlib** for visualization capabilities
- **⚡ Accelerate** for multi-GPU support

---

## 📞 **Support**

- 📧 **Email**: your.email@domain.com
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/WhiteBoxControl/issues)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/WhiteBoxControl/wiki)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/yourusername/WhiteBoxControl/discussions)

---

<div align="center">

**🌟 Star this repository if you find it useful! 🌟**

Made with ❤️ for AI Safety and Interpretability

</div>
