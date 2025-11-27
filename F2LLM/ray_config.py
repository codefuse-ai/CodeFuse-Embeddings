"""
Ray Train Configuration Module

This module defines the configuration dataclass for Ray distributed training,
extending the base Args class with Ray-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from arguments import Args
import yaml
import json


@dataclass
class RayTrainConfig(Args):
    """
    Ray Train configuration extending base Args

    Adds Ray-specific settings while maintaining compatibility with
    existing training configuration.
    """

    # Ray cluster settings
    ray_address: str = "auto"  # "auto" for local, "ray://head:10001" for remote
    num_workers: int = 8       # Number of training workers (typically = num GPUs)
    use_gpu: bool = True       # Whether to use GPU for training

    # Resource allocation per worker
    resources_per_worker: Optional[Dict[str, float]] = None  # {"CPU": 4, "GPU": 1}

    # Fault tolerance settings
    enable_fault_tolerance: bool = True
    max_retries: int = 3       # Maximum number of failure retries

    # DeepSpeed settings (preserved from Accelerate)
    use_deepspeed: bool = True
    zero_stage: int = 2        # ZeRO optimization stage (1, 2, or 3)

    # Checkpoint settings
    checkpoint_num_to_keep: int = 3
    checkpoint_score_attribute: str = "loss"
    checkpoint_score_order: str = "min"  # "min" or "max"

    # Communication backend
    backend: str = "nccl"      # "nccl" for GPU, "gloo" for CPU

    def __post_init__(self):
        """Post-initialization processing"""
        super().__post_init__() if hasattr(super(), '__post_init__') else None

        # Set default resources per worker if not specified
        if self.resources_per_worker is None:
            self.resources_per_worker = {
                "CPU": 4,
                "GPU": 1 if self.use_gpu else 0
            }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RayTrainConfig':
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            RayTrainConfig instance
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Merge configuration sections
        merged_config = {}

        # Merge cluster, scaling, training, fault_tolerance sections
        for section in ['cluster', 'scaling', 'training', 'fault_tolerance']:
            if section in config:
                section_data = config[section]
                if isinstance(section_data, dict):
                    merged_config.update(section_data)

        # Handle DeepSpeed configuration
        if 'deepspeed_config' in merged_config:
            ds_config = merged_config.pop('deepspeed_config')
            if isinstance(ds_config, dict):
                # Extract ZeRO stage
                zero_config = ds_config.get('zero_optimization', {})
                if isinstance(zero_config, dict):
                    merged_config['zero_stage'] = zero_config.get('stage', 2)

        # Add top-level parameters (model_path, experiment_id, etc.)
        for key in ['model_path', 'experiment_id', 'output_dir', 'tb_dir',
                    'cache_dir', 'train_data_path', 'train_batch_size',
                    'max_seq_length', 'learning_rate', 'min_lr', 'weight_decay',
                    'warmup_steps', 'num_hard_neg', 'train_steps', 'train_epochs',
                    'log_interval', 'checkpointing_steps', 'validation_steps']:
            if key in config:
                merged_config[key] = config[key]

        return cls(**merged_config)

    @classmethod
    def from_json(cls, json_path: str) -> 'RayTrainConfig':
        """
        Load configuration from JSON file (for compatibility with existing configs)

        Args:
            json_path: Path to JSON configuration file

        Returns:
            RayTrainConfig instance
        """
        with open(json_path, 'r') as f:
            config = json.load(f)

        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary

        Returns:
            Dictionary representation of configuration
        """
        return self.dict()

    def save_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file

        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = self.to_dict()

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def save_json(self, json_path: str):
        """
        Save configuration to JSON file

        Args:
            json_path: Path to save JSON configuration
        """
        config_dict = self.to_dict()

        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def create_ray_config_from_accelerate(
    accelerate_yaml: str,
    base_json: str,
    output_yaml: str
) -> RayTrainConfig:
    """
    Create Ray configuration from existing Accelerate config files

    This helper function converts Accelerate configuration to Ray format,
    preserving all training parameters.

    Args:
        accelerate_yaml: Path to accelerate_config.yaml
        base_json: Path to config.json (base training config)
        output_yaml: Path to save Ray configuration

    Returns:
        RayTrainConfig instance
    """
    # Load base training config
    with open(base_json, 'r') as f:
        base_config = json.load(f)

    # Load Accelerate config
    with open(accelerate_yaml, 'r') as f:
        acc_config = yaml.safe_load(f)

    # Map Accelerate settings to Ray settings
    ray_config = {
        **base_config,  # Include all base training parameters
        'num_workers': acc_config.get('num_processes', 8),
        'use_gpu': not acc_config.get('use_cpu', False),
        'backend': 'nccl' if not acc_config.get('use_cpu', False) else 'gloo',
    }

    # Map DeepSpeed settings if present
    if 'deepspeed_config' in acc_config:
        ds_config = acc_config['deepspeed_config']
        ray_config['use_deepspeed'] = True
        ray_config['zero_stage'] = ds_config.get('zero_stage', 2)

    # Create RayTrainConfig instance
    config = RayTrainConfig(**ray_config)

    # Save to YAML
    config.save_yaml(output_yaml)

    return config


if __name__ == "__main__":
    # Example usage
    print("Ray Train Configuration Module")
    print("=" * 60)

    # Example 1: Create config from scratch
    config = RayTrainConfig(
        model_path="/path/to/model",
        experiment_id="ray_test",
        output_dir="./outputs",
        tb_dir="./tensorboard",
        cache_dir="./cache",
        train_data_path="./data",
        num_workers=8,
        use_gpu=True,
    )

    print("\nExample 1: Config created from scratch")
    print(f"  Workers: {config.num_workers}")
    print(f"  GPU: {config.use_gpu}")
    print(f"  DeepSpeed: {config.use_deepspeed} (ZeRO-{config.zero_stage})")

    # Example 2: Load from YAML
    # config = RayTrainConfig.from_yaml("configs/ray_config.yaml")
    # print("\nExample 2: Config loaded from YAML")

    print("\n" + "=" * 60)
