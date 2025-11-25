from dataclasses import dataclass, asdict
import argparse, json


@dataclass
class Args:

    model_path: str
    experiment_id: str
    # save dir
    output_dir: str
    tb_dir: str
    cache_dir: str
    # training arguments
    train_data_path: str
    train_batch_size: int = 8
    max_seq_length: int = 2048
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    # embedding-related settings
    num_hard_neg: int = 7
    # train steps take precedence over epochs, set to -1 to disable
    train_steps: int = -1
    train_epochs: int = 5
    log_interval: int = 20
    checkpointing_steps: int = 100
    validation_steps: int = 100
    # model configuration
    model_type: str = "auto"  # auto, qwen, llama, baichuan, etc.
    attn_implementation: str = "flash_attention_2"  # flash_attention_2, sdpa, None
    use_flash_attention: bool = True
    # just placeholder, for logging purpose
    num_processes: int=0

    def dict(self):
        return asdict(self)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arg = parser.parse_args()
    with open(arg.config) as f:
        config = json.load(f)
    args = Args(**config)
    args.output_dir = f"{args.output_dir}/{args.experiment_id}"
    args.tb_dir = f"{args.tb_dir}/{args.experiment_id}"
    return args