#!/usr/bin/env python3
"""
Test script for the updated attention probe with HuggingFace datasets.
Tests the response token masking functionality with a simple sentiment analysis task.
"""

import json
import torch
from datasets import load_dataset
from probes.attention_probe import LearnedAttentionProbe
import os
import tempfile

def create_test_trajectories_from_hf():
    """Create test trajectories from HuggingFace dataset (IMDB sentiment)."""
    
    # Load a small subset of IMDB dataset for testing
    dataset = load_dataset("imdb", split="test[:20]")
    
    trajectories = []
    
    for idx, example in enumerate(dataset):
        # Create a simple conversation format
        messages = [
            {"role": "user", "content": f"Please analyze the sentiment of this review: {example['text'][:500]}"},
            {"role": "assistant", "content": f"This review is {'positive' if example['label'] == 1 else 'negative'}."}
        ]
        
        # Create trajectory in the expected format
        trajectory = {
            "trajectory_index": idx,
            "messages": messages,
            "classification": "unsafe" if example['label'] == 1 else "safe"  # Arbitrary mapping for testing
        }
        
        trajectories.append([trajectory])
    
    return trajectories

def test_response_masking():
    """Test that the response token masking works correctly."""
    
    print("=" * 60)
    print("Testing Attention Probe with HuggingFace Dataset")
    print("=" * 60)
    
    # Create test trajectories
    print("\n1. Creating test trajectories from IMDB dataset...")
    trajectories = create_test_trajectories_from_hf()
    print(f"   Created {len(trajectories)} trajectories")
    
    # Save trajectories to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([t[0] for t in trajectories], f)
        train_path = f.name
    
    # Split into train and eval
    train_trajectories = trajectories[:15]
    eval_trajectories = trajectories[15:]
    
    # Save train and eval files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([t[0] for t in train_trajectories], f)
        train_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([t[0] for t in eval_trajectories], f)
        eval_file = f.name
    
    print(f"   Train samples: {len(train_trajectories)}")
    print(f"   Eval samples: {len(eval_trajectories)}")
    
    # Initialize the probe
    print("\n2. Initializing LearnedAttentionProbe...")
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Use Qwen which is already downloaded
    
    with tempfile.TemporaryDirectory() as monitor_dir:
        probe = LearnedAttentionProbe(
            model_name=model_name,
            monitor_dir=monitor_dir,
            layer=16,  # Middle layer for Qwen 32B (which has ~64 layers)
            use_normalization=True,
            attention_scaling=True,
            initialization_scale=2.0,
            temperature=1.0,
            dtype=torch.bfloat16,  # Use bfloat16 for efficiency
            inference_batch_size_per_device=1,
        )
        
        print(f"   Model: {model_name}")
        print(f"   Layer: 16")
        print(f"   Use normalization: True")
        print(f"   Attention scaling: True")
        
        # Test the response token masking
        print("\n3. Testing response token masking...")
        test_trajectory = train_trajectories[0]
        
        try:
            mask = probe._get_response_token_mask(test_trajectory)
            print(f"   ✓ Response mask created successfully")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Response tokens: {mask.sum().item()} / {len(mask)}")
            
            # Show which parts are masked
            if len(mask) < 50:  # Only show for short sequences
                print(f"   Mask pattern: {mask.to(torch.int).tolist()}")
        except Exception as e:
            print(f"   ✗ Error creating response mask: {e}")
            return
        
        # Test training (very brief, just to ensure it runs)
        print("\n4. Testing training pipeline...")
        try:
            # Train for just 1 epoch with a fixed learning rate
            print("   Training for 1 epoch...")
            loss = probe.train(
                train_file,
                clear_activation_cache=True,
                num_epochs=1,
                lr=1e-3,
                l2_penalty=0.001,
                label_smoothing=0.02,
            )
            print(f"   ✓ Training completed successfully")
            print(f"   Final loss: {loss:.4f}")
            
        except Exception as e:
            print(f"   ✗ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test evaluation
        print("\n5. Testing evaluation...")
        try:
            metrics = probe.evaluate(
                eval_file,
                out_path=os.path.join(monitor_dir, "test_eval.json"),
                clear_activation_cache=True,
            )
            print(f"   ✓ Evaluation completed successfully")
            print(f"   Accuracy: {metrics.accuracy:.3f}")
            print(f"   AUC: {metrics.auc:.3f}")
            
        except Exception as e:
            print(f"   ✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up temp files
    for f in [train_file, eval_file]:
        try:
            os.unlink(f)
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_response_masking()