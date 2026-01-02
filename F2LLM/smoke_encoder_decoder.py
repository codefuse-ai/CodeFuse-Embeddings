"""
Lightweight smoke checks for encoder/decoder pooling and tokenizer behaviors.
Run: python smoke_encoder_decoder.py
"""
import torch
from tokenize_data_general import process_sent
from model import F2LLM


class MockTokenizer:
    def __init__(self, eos_token_id=2, pad_token_id=0):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, sentence, max_length, truncation=True, add_special_tokens=False):
        # deterministic token ids based on length
        base = list(range(1, min(max_length, len(sentence.split())) + 1))
        if add_special_tokens:
            ids = [101] + base
            if len(ids) < max_length:
                ids.append(102)
        else:
            ids = base
        ids = ids[:max_length]

        class Output:
            def __init__(self, ids):
                self.input_ids = ids
        return Output(ids)


def test_process_sent_encoder_special_tokens():
    tok = MockTokenizer()
    arr = process_sent("hello world", tok, max_seq_length=5, is_encoder_only=True, append_eos_decoder=True)
    assert arr[0] == 101, "CLS should be first"
    assert arr[-1] == 102, "SEP should be last when room remains"


def test_process_sent_decoder_eos_appended():
    tok = MockTokenizer(eos_token_id=9)
    arr = process_sent("a b c", tok, max_seq_length=6, is_encoder_only=False, append_eos_decoder=True)
    assert arr[-1] == 9, "EOS should be appended for decoder when enabled"


def test_process_sent_decoder_skip_eos():
    tok = MockTokenizer(eos_token_id=9)
    arr = process_sent("a b c", tok, max_seq_length=6, is_encoder_only=False, append_eos_decoder=False)
    assert arr[-1] != 9, "EOS should not be appended when disabled"


def test_encoder_pooling_variants():
    class Args:
        pooling = "cls"
        model_arch = "encoder"
    args = Args()
    model = F2LLM.__new__(F2LLM)
    model.args = args
    model.is_encoder_only = True
    bs = 2
    num_hard_neg = 1
    seq_lens = torch.tensor([5, 6, 7, 8, 5, 6])
    hidden = torch.randn(bs * (2 + num_hard_neg), 10, 4)
    attn_mask = torch.ones(bs * (2 + num_hard_neg), 10, dtype=torch.long)
    batch = {
        'input_ids': torch.zeros_like(attn_mask),
        'attention_mask': attn_mask,
        'seq_lens': seq_lens,
        'bs': bs
    }
    class MockLM:
        def __call__(self, input_ids, attention_mask):
            class Output:
                last_hidden_state = hidden
            return Output()
    model.lm = MockLM()
    model.lm.device = hidden.device
    model.forward = F2LLM.forward.__get__(model, F2LLM)

    out_cls = model.forward(batch)
    assert out_cls['query_passage_features'].shape == (bs, 1, hidden.size(-1))

    model.args.pooling = "mean"
    out_mean = model.forward(batch)
    assert out_mean['query_passage_features'].shape == (bs, 1, hidden.size(-1))

    model.args.pooling = "cls_mean"
    out_cls_mean = model.forward(batch)
    assert out_cls_mean['query_passage_features'].shape == (bs, 1, hidden.size(-1))


def test_decoder_pooling_last_token():
    model = F2LLM.__new__(F2LLM)
    model.args = None
    model.is_encoder_only = False
    bs = 2
    num_hard_neg = 1
    seq_lens = torch.tensor([2, 3, 4, 5, 6, 7])
    hidden = torch.randn(bs * (2 + num_hard_neg), 8, 4)
    attn_mask = torch.ones(bs * (2 + num_hard_neg), 8, dtype=torch.long)
    batch = {
        'input_ids': torch.zeros_like(attn_mask),
        'attention_mask': attn_mask,
        'seq_lens': seq_lens,
        'bs': bs
    }
    class MockLM:
        def __call__(self, input_ids, attention_mask):
            class Output:
                last_hidden_state = hidden
            return Output()
    model.lm = MockLM()
    model.lm.device = hidden.device
    model.forward = F2LLM.forward.__get__(model, F2LLM)

    out = model.forward(batch)
    assert out['query_passage_features'].shape == (bs, 1, hidden.size(-1))
    assert out['negative_passage_features'].shape == (bs, num_hard_neg, hidden.size(-1))


def main():
    tests = [
        test_process_sent_encoder_special_tokens,
        test_process_sent_decoder_eos_appended,
        test_process_sent_decoder_skip_eos,
        test_encoder_pooling_variants,
        test_decoder_pooling_last_token,
    ]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
