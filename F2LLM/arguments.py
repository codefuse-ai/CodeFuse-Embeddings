from dataclasses import dataclass, asdict
import argparse, json


@dataclass
class Args:
    model_path: str = None
    experiment_id: str = None

    model_config: dict = None
    tokenizer_config: dict = None
    
    # save dir
    output_dir: str = None
    tb_dir: str = None
    cache_dir: str = None
    
    # training arguments
    train_data_path: str = None
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
    
    # just placeholder, for logging purpose
    num_processes: int = 0

    def dict(self):
        return asdict(self)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    arg = parser.parse_args()
    
    with open(arg.config) as f:
        config = json.load(f)
    
    args = Args(**config)
    
    # 确保model_path正确设置
    if not args.model_path and args.model_config:
        args.model_path = args.model_config["model_path"]
    
    args.output_dir = f"{args.output_dir}/{args.experiment_id}"
    args.tb_dir = f"{args.tb_dir}/{args.experiment_id}"
    
    return args