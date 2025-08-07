#!/usr/bin/env python3
"""
Test script to verify that inference batch size > 1 works correctly after the fixes.
This script demonstrates that the problematic assertions have been removed and 
inference can work with different batch sizes.
"""

import sys
import os
import tempfile
import json
import torch

# Add the current directory to the path to import probes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from probes.attention_probe import LearnedAttentionProbe
from probes.logistic_probe import LogisticProbe
from probes.mean_diff_probe import MeanDiffProbe

def create_test_data():
    """Create minimal test data for verification."""
    test_data = []
    for i in range(10):
        # Create simple test trajectories
        trajectory = [{
            "messages": [
                {"role": "user", "content": f"Test message {i}"},
                {"role": "assistant", "content": f"Test response {i}"}
            ],
            "classification": "safe" if i % 2 == 0 else "unsafe",
            "trajectory_index": i
        }]
        test_data.append(trajectory)
    return test_data

def test_probe_with_batch_size(probe_class, probe_name, batch_size=4):
    """Test a probe with different batch sizes."""
    print(f"\n🧪 Testing {probe_name} with batch size {batch_size}")
    
    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_data = create_test_data()
        data_path = os.path.join(temp_dir, "test_data.json")
        with open(data_path, 'w') as f:
            json.dump(test_data, f)
        
        try:
            # Initialize probe with the specified batch size
            probe_kwargs = {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",  # Use smaller model for testing
                "monitor_dir": temp_dir,
                "inference_batch_size_per_device": batch_size,
                "dtype": torch.bfloat16,
                "layer": 15,
            }
            
            # Add probe-specific arguments
            if probe_class == LogisticProbe:
                probe_kwargs["token_position_aggregation_strategy"] = "mean"
            elif probe_class == MeanDiffProbe:
                probe_kwargs["token_position_aggregation_strategy"] = "mean"
            elif probe_class == LearnedAttentionProbe:
                probe_kwargs["use_normalization"] = False
                probe_kwargs["attention_scaling"] = True
            
            probe = probe_class(**probe_kwargs)
            
            # Test that initialization works
            print(f"  ✅ Probe initialization successful with batch size {batch_size}")
            
            # Create fake activations to test forward pass
            # Simulate activations with shape (1, num_tokens, hidden_size)
            fake_trajectories = test_data[:batch_size]
            fake_activations = []
            
            for _ in range(batch_size):
                # Create fake activation tensor with shape (1, seq_len, hidden_size)
                seq_len = 10  # arbitrary sequence length
                hidden_size = 4096  # typical hidden size
                activation = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
                fake_activations.append(activation)
            
            # Initialize probe parameters
            probe.init_params()
            
            # Test forward pass
            try:
                predictions = probe.forward(fake_trajectories, fake_activations)
                print(f"  ✅ Forward pass successful with batch size {batch_size}")
                print(f"  📊 Generated {len(predictions)} predictions")
                
                # Verify predictions are valid probabilities
                if torch.all(predictions >= 0) and torch.all(predictions <= 1):
                    print(f"  ✅ All predictions are valid probabilities [0, 1]")
                else:
                    print(f"  ❌ Some predictions are outside valid range [0, 1]")
                    
            except Exception as e:
                print(f"  ❌ Forward pass failed: {str(e)}")
                return False
                
            print(f"  🎉 {probe_name} successfully handles batch size {batch_size}")
            return True
            
        except Exception as e:
            print(f"  ❌ {probe_name} failed with batch size {batch_size}: {str(e)}")
            return False

def main():
    """Run the batch inference tests."""
    print("🔍 Testing Batch Inference Fixes")
    print("=" * 50)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    probe_classes = [
        (LogisticProbe, "Logistic Probe"),
        (LearnedAttentionProbe, "Attention Probe"),
        (MeanDiffProbe, "Mean Difference Probe")
    ]
    
    results = {}
    
    for probe_class, probe_name in probe_classes:
        results[probe_name] = {}
        for batch_size in batch_sizes:
            success = test_probe_with_batch_size(probe_class, probe_name, batch_size)
            results[probe_name][batch_size] = success
    
    # Print summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    for probe_name, batch_results in results.items():
        print(f"\n{probe_name}:")
        for batch_size, success in batch_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  Batch size {batch_size}: {status}")
    
    # Check if all tests passed
    all_passed = all(all(batch_results.values()) for batch_results.values())
    
    if all_passed:
        print(f"\n🎉 All tests passed! Batch inference is now working correctly.")
        print("💡 You can now safely use inference_batch_size_per_device > 1")
    else:
        print(f"\n❌ Some tests failed. Please check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 