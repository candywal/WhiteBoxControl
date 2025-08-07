#!/usr/bin/env python3
"""
Full test of the attention probe with Qwen model.
"""

import json
import torch
import os
import sys
from probes.attention_probe import LearnedAttentionProbe

def test_with_qwen():
    print("=" * 60)
    print("Testing Attention Probe with Qwen Model")
    print("=" * 60)
    
    # Check if we have existing trajectory files to use
    possible_paths = [
        "/mnt/nvme1/asher/white-box-monitoring/white_box_monitoring/environments/trajectories/animal_code_deception_dataset_safe.json",
        "/mnt/nvme1/asher/dataset_generation/white_box_monitoring/environments/trajectories/animal_code_deception_dataset_safe.json",
    ]
    
    trajectory_file = None
    for path in possible_paths:
        if os.path.exists(path):
            trajectory_file = path
            break
    
    if not trajectory_file:
        print("No trajectory files found. Creating synthetic data...")
        # Create simple synthetic trajectories
        trajectories = []
        for i in range(10):
            trajectory = {
                "trajectory_index": i,
                "messages": [
                    {"role": "user", "content": f"Write a Python function to calculate the factorial of {i+1}"},
                    {"role": "assistant", "content": f"Here's a function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nresult = factorial({i+1})\nprint(f'Factorial of {i+1} is {{result}}')\n```"}
                ],
                "classification": "safe" if i < 5 else "unsafe"
            }
            trajectories.append(trajectory)
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trajectories, f)
            trajectory_file = f.name
            temp_file = True
    else:
        print(f"Using existing trajectories from: {trajectory_file}")
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
        trajectories = trajectories[:10]  # Use only first 10
        temp_file = False
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Initialize the probe
    print("\n1. Initializing LearnedAttentionProbe with Qwen...")
    monitor_dir = "/tmp/qwen_test"
    os.makedirs(monitor_dir, exist_ok=True)
    
    probe = LearnedAttentionProbe(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        monitor_dir=monitor_dir,
        layer=16,
        use_normalization=True,
        attention_scaling=True,
        initialization_scale=2.0,
        temperature=2.0,
        dtype=torch.bfloat16,
        inference_batch_size_per_device=1,
    )
    
    print("   ✓ Probe initialized")
    print(f"   Hidden size: {probe.hidden_size}")
    
    # Test the response token masking with Qwen
    print("\n2. Testing response token masking with Qwen tokenizer...")
    
    # Initialize the model to get tokenizer
    print("   Loading Qwen model and tokenizer...")
    probe.init_model()
    
    try:
        # Test on first trajectory
        test_trajectory = [trajectories[0]]
        mask = probe._get_response_token_mask(test_trajectory)
        
        print(f"   ✓ Response mask created successfully")
        print(f"   Total tokens: {len(mask)}")
        print(f"   Response tokens: {mask.sum().item()}")
        print(f"   Response percentage: {mask.sum().item() / len(mask) * 100:.1f}%")
        
        # Show a sample of the mask
        if len(mask) < 200:
            mask_str = ''.join(['R' if m else 'P' for m in mask[:100]])
            print(f"   Mask pattern (first 100): P=Prompt, R=Response")
            print(f"   {mask_str}")
            
    except Exception as e:
        print(f"   ✗ Error in response masking: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test getting activations
    print("\n3. Testing activation extraction...")
    try:
        # Get activations for a few trajectories
        test_trajectories = trajectories[:2]
        activations = probe.get_activations(test_trajectories, layers=[probe.layer])
        
        print(f"   ✓ Activations extracted successfully")
        print(f"   Number of trajectories: {len(activations)}")
        print(f"   Activation shape per trajectory: {activations[0].shape}")
        
    except Exception as e:
        print(f"   ✗ Error getting activations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass with actual data
    print("\n4. Testing forward pass with real trajectories...")
    try:
        # Format trajectories properly
        formatted_trajectories = [[t] for t in test_trajectories]
        
        # Run forward pass
        with torch.no_grad():
            predictions = probe.forward(formatted_trajectories, activations)
        
        print(f"   ✓ Forward pass successful")
        print(f"   Predictions: {predictions.tolist()}")
        print(f"   Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Quick training test
    print("\n5. Testing training (1 epoch)...")
    
    # Save trajectories for training
    train_file = os.path.join(monitor_dir, "train.json")
    with open(train_file, 'w') as f:
        json.dump(trajectories[:8], f)
    
    eval_file = os.path.join(monitor_dir, "eval.json") 
    with open(eval_file, 'w') as f:
        json.dump(trajectories[8:], f)
    
    try:
        print("   Training for 1 epoch...")
        loss = probe.train(
            train_file,
            clear_activation_cache=True,
            num_epochs=1,
            lr=1e-3,
            l2_penalty=0.001,
            label_smoothing=0.02,
        )
        print(f"   ✓ Training completed")
        print(f"   Final loss: {loss:.4f}")
        
        # Save the trained model
        probe.save_model_params(os.path.join(monitor_dir, "trained_params.pt"))
        print("   ✓ Model parameters saved")
        
    except Exception as e:
        print(f"   ✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    # Cleanup
    if temp_file:
        os.unlink(trajectory_file)

if __name__ == "__main__":
    test_with_qwen()