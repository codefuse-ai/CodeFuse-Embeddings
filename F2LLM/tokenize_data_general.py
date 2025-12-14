from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
import argparse

def process_sent(sentence, tokenizer, max_seq_length, is_encoder_only, append_eos_decoder=True):
    if is_encoder_only:
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True)
    else:
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False)
        if append_eos_decoder and tokenizer.eos_token_id is not None:
            if tokenizer_outputs.input_ids and tokenizer_outputs.input_ids[-1] != tokenizer.eos_token_id:
                tokenizer_outputs.input_ids.append(tokenizer.eos_token_id)
    return np.array(tokenizer_outputs.input_ids)


def process_sent_batch(data, tokenizer, max_seq_length, is_encoder_only, append_eos_decoder):
    return data.apply(lambda x: process_sent(x, tokenizer, max_seq_length, is_encoder_only, append_eos_decoder))


def parallelize_apply(data, tokenizer, max_seq_length, is_encoder_only, append_eos_decoder, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    args_list = [(ds, tokenizer, max_seq_length, is_encoder_only, append_eos_decoder) for ds in data_split]
    with Pool(num_of_processes) as pool:
        results = pool.starmap(process_sent_batch, args_list)
    return pd.concat(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_dir", type=str, default='training_data', help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default='data_tokenized', help="Directory to save tokenized data")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for parallel tokenization")
    parser.add_argument("--arch", type=str, choices=["auto", "encoder", "decoder"], default="auto", help="Force encoder/decoder tokenization behavior")
    parser.add_argument("--no_append_eos_decoder", action="store_true", help="Skip appending EOS for decoder models")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    encoder_archs = ['BertModel', 'RobertaModel', 'DebertaModel', 'ElectraModel', 'AlbertModel', 'DistilBertModel']
    detected_encoder = any(arch in getattr(config, 'architectures', []) for arch in encoder_archs)
    if args.arch != "auto":
        is_encoder_only = args.arch == "encoder"
    else:
        is_encoder_only = detected_encoder

    append_eos_decoder = not args.no_append_eos_decoder

    if not is_encoder_only and append_eos_decoder and tokenizer.eos_token_id is None:
        if tokenizer.pad_token_id is not None:
            tokenizer.eos_token_id = tokenizer.pad_token_id
        elif getattr(tokenizer, 'unk_token_id', None) is not None:
            tokenizer.eos_token_id = tokenizer.unk_token_id
        else:
            tokenizer.eos_token_id = 0

    max_seq_length = args.max_seq_length - 2 if is_encoder_only else args.max_seq_length

    root_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for ds_name in tqdm(sorted(os.listdir(root_dir))):
        print(ds_name, flush=True)
        df = pd.read_parquet(f"{root_dir}/{ds_name}")
        df['query_input_ids'] = parallelize_apply(df['query'], tokenizer, max_seq_length, is_encoder_only, append_eos_decoder, args.num_processes)

        num_neg = 24 if 'negative_2' in df.keys() else 1
        texts = df.passage.to_list()
        for i in range(1, num_neg+1):
            texts += df[f'negative_{i}'].to_list()
        texts = list(set(texts))
        df_tmp = pd.DataFrame({'text': texts})
        df_tmp['input_ids'] = parallelize_apply(df_tmp['text'], tokenizer, max_seq_length, is_encoder_only, append_eos_decoder, args.num_processes)
        df_tmp = df_tmp.set_index('text')

        df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)
        for i in range(1, num_neg+1):
            df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)

        df.to_parquet(f'{output_dir}/{ds_name}', index=False)


if __name__ == "__main__":
    main()
