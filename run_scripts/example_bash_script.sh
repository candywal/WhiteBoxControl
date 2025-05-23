#!/bin/bash
# example_bash_script.sh - Example usage of WhiteBoxControl API

echo "🚀 WhiteBoxControl - Example Training and Evaluation Script"
echo "============================================================"

# Configuration
MODEL="/mnt/nvme5/anshul/models/Qwen2.5-Coder-32B-Instruct"
TRAIN_DATA="data/datasets/combined/code_sabotage_train.json"
TEST_DATA="data/datasets/combined/code_sabotage_test.json"
OUTPUT_DIR="outputs/trained_probes"
LAYER=32
GPU_ID=0

echo "📋 Configuration:"
echo "  Model: $MODEL"
echo "  Training Data: $TRAIN_DATA"
echo "  Test Data: $TEST_DATA"
echo "  Layer: $LAYER"
echo "  GPU: $GPU_ID"
echo ""

# Train different probe types
echo "🎯 Training Probes..."
echo "====================="

# 1. Train Logistic Probe
echo "🔵 Training Logistic Probe..."
python TrainProbe.py \
    --probe_type logistic \
    --model "$MODEL" \
    --data "$TRAIN_DATA" \
    --layer $LAYER \
    --epochs 10 \
    --lr 0.001 \
    --gpu_id $GPU_ID \
    --output_dir "$OUTPUT_DIR" \
    --clear_cache

# 2. Train Attention Probe (with best hyperparameters from experiments)
echo "🔴 Training Attention Probe..."
python TrainProbe.py \
    --probe_type attention \
    --model "$MODEL" \
    --data "$TRAIN_DATA" \
    --layer $LAYER \
    --epochs 14 \
    --lr 0.0022 \
    --gpu_id $GPU_ID \
    --output_dir "$OUTPUT_DIR" \
    --initialization_scale 1.5 \
    --clear_cache

# 3. Train Mean Difference Probe  
echo "🟡 Training Mean Difference Probe..."
python TrainProbe.py \
    --probe_type mean_diff \
    --model "$MODEL" \
    --data "$TRAIN_DATA" \
    --layer $LAYER \
    --gpu_id $GPU_ID \
    --output_dir "$OUTPUT_DIR" \
    --token_aggregation last_only \
    --clear_cache

echo ""
echo "✅ Training Complete!"
echo ""

# Evaluate all trained probes
echo "🔍 Evaluating Probes..."
echo "======================="

# Evaluate Logistic Probe
echo "🔵 Evaluating Logistic Probe..."
python EvalProbe.py \
    --probe_dir "$OUTPUT_DIR/logistic_probe_layer$LAYER" \
    --data "$TEST_DATA" \
    --gpu_id $GPU_ID \
    --save_predictions \
    --save_plots \
    --clear_cache

# Evaluate Attention Probe  
echo "🔴 Evaluating Attention Probe..."
python EvalProbe.py \
    --probe_dir "$OUTPUT_DIR/attention_probe_layer$LAYER" \
    --data "$TEST_DATA" \
    --gpu_id $GPU_ID \
    --save_predictions \
    --save_plots \
    --clear_cache

# Evaluate Mean Difference Probe
echo "🟡 Evaluating Mean Difference Probe..."
python EvalProbe.py \
    --probe_dir "$OUTPUT_DIR/mean_diff_probe_layer$LAYER" \
    --data "$TEST_DATA" \
    --gpu_id $GPU_ID \
    --save_predictions \
    --save_plots \
    --clear_cache

echo ""
echo "✅ Evaluation Complete!"
echo ""

# Batch evaluation comparison
echo "📊 Batch Evaluation Comparison..."
echo "================================="

python EvalProbe.py \
    --probe_dir "$OUTPUT_DIR/logistic_probe_layer$LAYER" \
    --data "$TEST_DATA" "data/datasets/combined/code_sabotage_train.json" \
    --batch_eval \
    --output_dir "outputs/evaluation/batch_comparison_logistic" \
    --gpu_id $GPU_ID

echo ""
echo "🎉 All Done!"
echo "📁 Results saved in:"
echo "  - Trained probes: $OUTPUT_DIR"
echo "  - Evaluations: outputs/evaluation/"
echo ""
echo "🔍 Quick Results Summary:"
echo "Run: ls -la $OUTPUT_DIR/*/evaluation/" 