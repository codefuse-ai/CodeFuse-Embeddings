#!/usr/bin/env python3
"""
Ray Train Launcher Script for F2LLM

This script initializes Ray and starts distributed training using Ray Train.

Usage:
    # Local single-node training
    python scripts/run_ray_train.py --config F2LLM/configs/ray_config.yaml

    # Connect to remote Ray cluster
    python scripts/run_ray_train.py --config F2LLM/configs/ray_config.yaml --ray-address ray://head:10001

    # Override configuration parameters
    python scripts/run_ray_train.py --config F2LLM/configs/ray_config.yaml --num-workers 16
"""

import argparse
import sys
import os

# Add F2LLM to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'F2LLM'))

import ray
from ray_train import RayF2LLMTrainer
from ray_config import RayTrainConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ray Train launcher for F2LLM distributed training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local training with 8 GPUs
  python scripts/run_ray_train.py --config F2LLM/configs/ray_config.yaml

  # Multi-node training (connect to existing cluster)
  python scripts/run_ray_train.py --config F2LLM/configs/ray_config.yaml \\
      --ray-address ray://10.0.0.1:10001 --num-workers 16

  # Override specific parameters
  python scripts/run_ray_train.py --config F2LLM/configs/ray_config.yaml \\
      --experiment-id my_experiment --train-epochs 10
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Ray configuration file (YAML)"
    )

    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (e.g., 'ray://head:10001'). If not specified, uses value from config."
    )

    # Configuration overrides
    parser.add_argument("--num-workers", type=int, help="Override number of workers")
    parser.add_argument("--experiment-id", type=str, help="Override experiment ID")
    parser.add_argument("--train-epochs", type=int, help="Override training epochs")
    parser.add_argument("--train-steps", type=int, help="Override training steps")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--model-path", type=str, help="Override model path")

    # Debugging options
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run Ray in local mode for debugging (single process)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage (CPU training)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    print("=" * 80)
    print("Ray Train Launcher for F2LLM")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = RayTrainConfig.from_yaml(args.config)

    # Apply command-line overrides
    if args.ray_address:
        config.ray_address = args.ray_address
    if args.num_workers:
        config.num_workers = args.num_workers
    if args.experiment_id:
        config.experiment_id = args.experiment_id
    if args.train_epochs:
        config.train_epochs = args.train_epochs
    if args.train_steps:
        config.train_steps = args.train_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.model_path:
        config.model_path = args.model_path
    if args.no_gpu:
        config.use_gpu = False

    # Display configuration
    print("\nTraining Configuration:")
    print(f"  Experiment ID: {config.experiment_id}")
    print(f"  Model: {config.model_path}")
    print(f"  Workers: {config.num_workers}")
    print(f"  GPUs: {'Yes' if config.use_gpu else 'No'}")
    print(f"  Epochs: {config.train_epochs}")
    print(f"  Batch size: {config.train_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output dir: {config.output_dir}")

    # Initialize Ray
    print("\nInitializing Ray...")
    ray_address = config.ray_address

    if args.local_mode:
        print("  Mode: Local (debugging)")
        ray.init(local_mode=True)
    elif ray_address == "auto":
        print("  Mode: Auto (local cluster)")
        ray.init()
    else:
        print(f"  Mode: Remote cluster at {ray_address}")
        ray.init(address=ray_address)

    # Display cluster information
    try:
        dashboard_url = ray.get_runtime_context().get_dashboard_url()
        print(f"  Dashboard: {dashboard_url}")
    except Exception:
        print("  Dashboard: Not available")

    resources = ray.available_resources()
    print(f"  Available CPUs: {resources.get('CPU', 0):.0f}")
    print(f"  Available GPUs: {resources.get('GPU', 0):.0f}")

    if config.use_gpu and resources.get('GPU', 0) == 0:
        print("\n⚠️  WARNING: GPU training requested but no GPUs available!")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            ray.shutdown()
            return

    # Create trainer
    print("\nCreating Ray trainer...")
    trainer = RayF2LLMTrainer(config)

    # Start training
    print("\n" + "=" * 80)
    print(f"Starting training: {config.experiment_id}")
    print("=" * 80)
    print()

    try:
        result = trainer.fit()

        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"\nResults: {result}")
        print(f"\nCheckpoints saved to: {config.output_dir}")
        print(f"TensorBoard logs: {config.tb_dir}")
        print("\nTo view training metrics:")
        print(f"  tensorboard --logdir {config.tb_dir}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("Training failed!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Shutdown Ray
        print("\nShutting down Ray...")
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
