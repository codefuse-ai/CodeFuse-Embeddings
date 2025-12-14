import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm

# Minimal tokenizer-like object
class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

# Dummy model implementing required interface
class DummyModel:
    def __init__(self, hidden_size=32, tokenizer=None, device="cpu"):
        self.tokenizer = tokenizer or DummyTokenizer()
        self.lm = nn.Sequential(
            nn.Embedding(30522, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self._device = torch.device(device)
        self.lm.to(self._device)

    def set_device(self):
        self._device = next(self.lm.parameters()).device

    @property
    def device(self):
        return self._device

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device)  # [bs_total, seq]
        attention_mask = batch['attention_mask'].to(self.device)
        bs = batch['bs']
        # Compute simple pooled features
        emb = self.lm[0](input_ids)  # [bs_total, seq, h]
        pooled = (emb * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        # split back into query/passages/negatives
        num_hard = 1  # keep simple
        q = pooled[:bs]
        p = pooled[bs:2*bs]
        negs = pooled[2*bs:2*bs+bs*num_hard].view(bs, num_hard, -1)
        return {
            'query_passage_features': q.unsqueeze(1),        # [bs,1,h]
            'passage_passage_features': p.unsqueeze(1),      # [bs,1,h]
            'negative_passage_features': negs                # [bs,num_hard,h]
        }

class SyntheticDataset(Dataset):
    def __init__(self, length=64, seq_len=16, vocab=100):
        self.length = length
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        def rand_ids():
            return [torch.randint(1, self.vocab, ()).item() for _ in range(self.seq_len)]
        return {
            'query_input_ids': rand_ids(),
            'passage_input_ids': rand_ids(),
            'negative_1_input_ids': rand_ids(),
            'dataset_name': 'msmarco'
        }

def _stack(input_ids, max_len, pad_id):
    data = [ids[:max_len] for ids in input_ids]
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))
    chunks = tensor.split(lens)
    return chunks

def collate_fn(batch_raw, max_seq_length=32, tokenizer=None):
    tokenizer = tokenizer or DummyTokenizer()
    num_hard_neg = 1
    input_ids = _stack(
        [s['query_input_ids'] for s in batch_raw]+
        [s['passage_input_ids'] for s in batch_raw]+
        [s[f'negative_1_input_ids'] for s in batch_raw],
        max_seq_length,
        tokenizer.pad_token_id
    )
    seqlens = torch.tensor([ids.size(0) for ids in input_ids])
    # pad to batch
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
    return {
        'input_ids': input_ids,
        'seq_lens': seqlens,
        'attention_mask': attention_masks,
        'bs': len(batch_raw),
        'dataset_name': batch_raw[0]['dataset_name']
    }

# Minimal loss helpers adapted from utils.py
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def inbatch_loss(q, c, criterion, accelerator, temperature=0.05):
    bs = q.size(0)
    a_norm = F.normalize(q, p=2, dim=-1)
    b_cross = accelerator.gather(c)
    b_norm = F.normalize(b_cross, p=2, dim=-1)
    logits = torch.matmul(a_norm, b_norm.t()) / temperature
    labels = torch.arange(bs, device=logits.device) + bs * accelerator.process_index
    loss_bs = criterion(logits, labels)
    return loss_bs.mean()

def hard_loss(q, c, negs, criterion, accelerator, temperature=0.05):
    if negs is None:
        return torch.tensor(0.0, device=q.device)
    bs = q.size(0)
    a = F.normalize(q, p=2, dim=-1)
    hard = torch.concat([c.unsqueeze(1), negs], dim=1)
    hard = F.normalize(hard, p=2, dim=-1)
    logits = (a.unsqueeze(1) * hard).sum(-1) / temperature
    return criterion(logits, torch.zeros((bs), dtype=torch.long, device=logits.device)).mean()


def main():
    accelerator = Accelerator()
    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer=tokenizer, device="cpu")
    model.set_device()

    ds = SyntheticDataset(length=32, seq_len=8, vocab=100)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate_fn(b, max_seq_length=16, tokenizer=tokenizer))
    loader = accelerator.prepare(loader)

    optimizer = torch.optim.SGD(model.lm.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    criterion = CrossEntropyLoss(reduction='none')

    accumulation_steps = 4
    total_micro = len(loader)
    expected_opt_steps = total_micro // accumulation_steps
    completed = 0
    local_accum = 0

    for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
        out = model.forward(batch)
        loss_h = hard_loss(out['query_passage_features'].squeeze(1), out['passage_passage_features'].squeeze(1), out['negative_passage_features'], criterion, accelerator)
        loss_ib = inbatch_loss(out['query_passage_features'].squeeze(1), out['passage_passage_features'].squeeze(1), criterion, accelerator)
        loss = (loss_h + loss_ib) / accumulation_steps
        accelerator.backward(loss)
        local_accum += 1
        if local_accum % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            completed += 1
        if completed >= expected_opt_steps:
            break

    print(f"Optimization steps: {completed} (expected {expected_opt_steps})")
    assert completed == expected_opt_steps, "Accumulation did not match expected steps"
    print("Smoke test passed.")

if __name__ == "__main__":
    main()
