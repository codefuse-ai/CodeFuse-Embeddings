"""
Utilities for LoRA (Low-Rank Adaptation) support in F2LLM.
This module provides functions for loading LoRA models and converting between full and LoRA models.
"""

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch


def load_model_with_lora(base_model_path, lora_adapter_path=None, **lora_kwargs):
    """
    Load a base model with optional LoRA adapter.
    
    Args:
        base_model_path (str): Path to the base model
        lora_adapter_path (str, optional): Path to the LoRA adapter
        **lora_kwargs: Additional LoRA configuration arguments
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Load the base model
    model = AutoModel.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Apply LoRA if adapter path is provided
    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print(f"Loaded LoRA adapter from {lora_adapter_path}")
    elif lora_kwargs:  # Apply new LoRA if configuration is provided
        target_modules = lora_kwargs.get("target_modules", "all-linear")
        if target_modules == "all-linear":
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ]
        elif isinstance(target_modules, str):
            target_modules = [module.strip() for module in target_modules.split(",")]
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_kwargs.get("lora_r", 8),
            lora_alpha=lora_kwargs.get("lora_alpha", 16),
            target_modules=target_modules,
            lora_dropout=lora_kwargs.get("lora_dropout", 0.05),
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA with config: {lora_config}")
    
    return model, tokenizer


def merge_lora_weights(model, save_path=None):
    """
    Merge LoRA weights with the base model.
    
    Args:
        model: PEFT model with LoRA
        save_path (str, optional): Path to save the merged model
    
    Returns:
        Merged model
    """
    if hasattr(model, 'merge_and_unload'):
        merged_model = model.merge_and_unload()
        if save_path:
            merged_model.save_pretrained(save_path)
        return merged_model
    else:
        raise ValueError("Model does not support merging. Make sure it's a PEFT model.")


def get_lora_model_info(model):
    """
    Get information about a LoRA model.
    
    Args:
        model: PEFT model with LoRA
    
    Returns:
        dict: Information about the model's LoRA configuration
    """
    if hasattr(model, 'peft_config'):
        info = {}
        for adapter_name, config in model.peft_config.items():
            info[adapter_name] = {
                'r': config.r,
                'alpha': config.lora_alpha,
                'dropout': config.lora_dropout,
                'target_modules': config.target_modules,
                'bias': config.bias,
            }
        return info
    else:
        return {"message": "Model does not have LoRA configuration"}


def count_parameters(model, only_trainable=False):
    """
    Count the number of parameters in the model.
    
    Args:
        model: PyTorch model
        only_trainable (bool): Whether to count only trainable parameters
    
    Returns:
        int: Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())