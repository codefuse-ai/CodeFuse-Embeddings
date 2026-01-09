from arguments import parse_args
from utils import accelerate_train, CLASSIFICATION_DATASETS
from transformers import (
    AutoTokenizer,
    AutoConfig,
    set_seed,
    get_scheduler
)
import os, json, random
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from model import F2LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = parse_args()
accelerator = Accelerator()
args.num_processes = accelerator.num_processes
accelerator.print(args)

# Load model config to determine if it's encoder-only
model_config = AutoConfig.from_pretrained(args.model_path)
is_encoder_only = any(arch in model_config.architectures for arch in ['BertModel', 'RobertaModel', 'DebertaModel', 'ElectraModel', 'AlbertModel', 'DistilBertModel'])

def _stack(input_ids, max_len):
    data = [ids[:max_len] for ids in input_ids]     # input_ids: list of lists
    data = [ids if ids[-1] == tokenizer.eos_token_id else ids[:-1] + [tokenizer.eos_token_id] for ids in data]
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))            # (total_tokens,)
    return tensor.split(lens)                       # list of 1-d tensors


def collate_fn(batch_raw):
    '''
        length of input_ids: bs * (2 + num_hard_neg)
        0 - bs-1: query input ids
        bs - 2*bs-1: passage input ids
        2*bs - 2*bs+num_hard_neg-1: hard neg for sample 1
        2*bs+num_hard_neg*(i-1) - 2*bs+num_hard_neg*i-1: hard neg for sample i (i from 1 to bs)
    '''
    num_hard_neg = 1 if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS else args.num_hard_neg
    # select args.num_hard_neg hard negatives from a total of 24
    hard_neg_indices = [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)
    input_ids = _stack(
        [s['query_input_ids'] for s in batch_raw]+\
        [s['passage_input_ids'] for s in batch_raw]+\
        [s[f'negative_{i+1}_input_ids'] for s in batch_raw for i in hard_neg_indices],
        args.max_seq_length
    )
    seqlens = torch.tensor([ids.size(0) for ids in input_ids])
    # pad input ids to [bs, max_len]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
    
    return {'input_ids': input_ids, 'seq_lens': seqlens, 'attention_mask': attention_masks, 'bs': len(batch_raw), 'dataset_name': batch_raw[0]['dataset_name']}


set_seed(0)
if accelerator.is_main_process:
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:   
        json.dump(args.dict(), f, indent=2)

train_datasets, valid_datasets = [], []
with accelerator.main_process_first():
    for f in sorted(os.listdir(args.train_data_path)):
        dataset_name = f.split('.parquet')[0]
        dataset = load_dataset("parquet", data_files=os.path.join(args.train_data_path, f), cache_dir=args.cache_dir)['train']
        dataset = dataset.add_column("dataset_name", [dataset_name]*len(dataset))
        dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
        train_datasets.append((dataset_name, dataset['train']))
        valid_datasets.append((dataset_name, dataset['test']))

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

train_loaders = {
    name: DataLoader(ds, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
    for name, ds in train_datasets
}
valid_loaders = {
    name: DataLoader(ds, shuffle=False, batch_size=args.train_batch_size, collate_fn=collate_fn)
    for name, ds in valid_datasets
}

class MultiLoader:
    """
    Iterates over a dict(name -> DataLoader) and returns complete batches.
    At every __iter__ a new random order is created;
    the epoch ends when every loader is exhausted once.
    """
    def __init__(self, loader_dict):
        self.loader_dict = loader_dict
        for k, v in self.loader_dict.items():
            self.loader_dict[k] = accelerator.prepare(v)

    def __len__(self):
        return sum(len(v) for v in self.loader_dict.values())
    
    def reset_epoch(self, epoch):
        self.rng = random.Random(epoch)
        self.iters = {k: iter(v) for k, v in self.loader_dict.items()}
        self.names = list(self.iters.keys())
        self.weights = [len(self.loader_dict[k]) for k in self.names]

    def __iter__(self):
        while self.names:                           # until every DataLoader is empty
            name = self.rng.choices(self.names, weights=self.weights)[0] # pick a data-source at random
            try:
                batch = next(self.iters[name])
                yield batch
            except StopIteration:
                idx = self.names.index(name)
                self.names.pop(idx)                 # this dataset has no batch left
                self.weights.pop(idx)


# determine training steps
override_train_step = False
if args.train_steps < 0:
    args.train_steps = sum(len(v) for v in train_loaders.values()) * args.train_epochs
    override_train_step = True

accelerator.print(f"******************************** Training step before prepare: {args.train_steps} ********************************")
model = F2LLM(args.model_path, args.max_seq_length, args=args)
model.lm.gradient_checkpointing_enable()
# set seed again to make sure that different models share the same seed
set_seed(0)

optimizer = AdamW(model.lm.parameters(),
                  weight_decay=args.weight_decay,
                  lr=args.learning_rate,
                  betas=(0.9, 0.98))

lr_scheduler = get_scheduler("cosine",
                            optimizer=optimizer,
                            num_warmup_steps=args.warmup_steps,
                            num_training_steps=args.train_steps)

AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size
model.lm, optimizer, lr_scheduler = accelerator.prepare(
    model.lm, optimizer, lr_scheduler
)
model.set_device()
train_dataloader = MultiLoader(train_loaders)
for k, v in valid_loaders.items():
    valid_loaders[k] = accelerator.prepare(v)

# if training on multiple GPUs, length of dataloader would have changed
if override_train_step:
    args.train_steps = len(train_dataloader) * args.train_epochs
accelerator.print(f"******************************** Training step after prepare: {args.train_steps} ********************************")


accelerate_train(args, accelerator, model, train_dataloader, valid_loaders,
                 optimizer, lr_scheduler, sum(len(d[1]) for d in train_datasets))
