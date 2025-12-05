"""
test to verify LoRA functionality in F2LLM
"""
import torch
from arguments import Args
from model import F2LLM
import tempfile
import os

def test_lora_functionality():
    """Test that LoRA can be applied to the model correctly."""
    
    # Create a mock args object with LoRA enabled
    args = Args(
        model_path="microsoft/Phi-3-mini-4k-instruct",  # Using a smaller model for testing
        experiment_id="test_lora",
        output_dir="test_output",
        tb_dir="test_tb",
        cache_dir="test_cache",
        train_data_path="dummy_path",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="all-linear"
    )
    
    try:
        print("Testing LoRA functionality...")
        
        # Create model with LoRA
        model = F2LLM(
            model_path=args.model_path,
            max_seq_length=512,
            args=args
        )
        
        # Check that model has LoRA applied
        total_params = model.lm.num_parameters()
        trainable_params = model.lm.num_parameters(only_trainable=True)
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")
        
        # With LoRA, we expect significantly fewer trainable parameters
        assert trainable_params < total_params * 0.1, \
            f"Expected fewer trainable parameters with LoRA. Total: {total_params}, Trainable: {trainable_params}"
        
        print("LoRA functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"PEFT library not available: {e}")
        print("Please install PEFT: pip install peft")
        return False
    except Exception as e:
        print(f"Error during LoRA test: {e}")
        return False


def test_non_lora_functionality():
    """Test that the model still works without LoRA."""
    
    # Create a mock args object with LoRA disabled
    args = Args(
        model_path="microsoft/Phi-3-mini-4k-instruct",  # Using a smaller model for testing
        experiment_id="test_no_lora",
        output_dir="test_output",
        tb_dir="test_tb",
        cache_dir="test_cache",
        train_data_path="dummy_path",
        use_lora=False
    )
    
    try:
        print("Testing non-LoRA functionality...")
        
        # Create model without LoRA
        model = F2LLM(
            model_path=args.model_path,
            max_seq_length=512,
            args=args
        )
        
        # Check that model parameters are as expected (all trainable)
        total_params = model.lm.num_parameters()
        trainable_params = model.lm.num_parameters(only_trainable=True)
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        
        # Without LoRA, most parameters should be trainable
        assert abs(trainable_params - total_params) < 10, \
            f"Expected most parameters to be trainable without LoRA. Total: {total_params}, Trainable: {trainable_params}"
        
        print("Non-LoRA functionality test passed!")
        return True
        
    except Exception as e:
        print(f"Error during non-LoRA test: {e}")
        return False


if __name__ == "__main__":
    print("Running LoRA functionality tests...")
    
    # Test LoRA functionality
    lora_test_passed = test_lora_functionality()
    
    # Test non-LoRA functionality
    no_lora_test_passed = test_non_lora_functionality()
    
    if lora_test_passed and no_lora_test_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)
