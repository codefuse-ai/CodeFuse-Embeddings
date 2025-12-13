from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
import argparse
import functools


def process_sent(sentence, tokenizer, max_seq_length, is_encoder_only):
    if is_encoder_only:
        # For encoder-only models, add special tokens automatically
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True)
    else:
        # For decoder-only models, manually add eos token
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False)
        # Add EOS token if not present and if available
        if tokenizer.eos_token_id is not None and tokenizer_outputs.input_ids and tokenizer_outputs.input_ids[-1] != tokenizer.eos_token_id:
            tokenizer_outputs.input_ids.append(tokenizer.eos_token_id)

    return np.array(tokenizer_outputs.input_ids)


def process_sent_batch(data, tokenizer, max_seq_length, is_encoder_only):
    """Process a batch of sentences"""
    return data.apply(lambda x: process_sent(x, tokenizer, max_seq_length, is_encoder_only))


def parallelize_apply(data, tokenizer, max_seq_length, is_encoder_only, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    
    args_list = [(ds, tokenizer, max_seq_length, is_encoder_only) for ds in data_split]
    
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
    
    args = parser.parse_args()
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Determine if model is encoder-only
    is_encoder_only = any(arch in config.architectures for arch in ['BertModel', 'RobertaModel', 'DebertaModel', 'ElectraModel', 'AlbertModel', 'DistilBertModel'])
    
    # Handle missing eos token for decoder models
    if not is_encoder_only and tokenizer.eos_token_id is None:
        if tokenizer.pad_token_id is not None:
            tokenizer.eos_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
            tokenizer.eos_token_id = tokenizer.unk_token_id
        else:
            # Create a default eos token id
            tokenizer.eos_token_id = 0  # Using 0 as default, though this isn't ideal
    
    max_seq_length = args.max_seq_length - 2 if is_encoder_only else args.max_seq_length  # Reserve space for [CLS] and [SEP] if needed

    root_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for ds_name in tqdm(sorted(os.listdir(root_dir))):
        print(ds_name, flush=True)

        df = pd.read_parquet(f"{root_dir}/{ds_name}")
        
        # Process query input IDs
        df['query_input_ids'] = parallelize_apply(df['query'], tokenizer, args.max_seq_length, is_encoder_only, args.num_processes)
        
        num_neg = 24 if 'negative_2' in df.keys() else 1

        # Get all unique passages and negatives for efficient tokenization
        ls = df.passage.to_list()
        for i in range(1, num_neg+1):
            ls += df[f'negative_{i}'].to_list()
        ls = list(set(ls))
        df_tmp = pd.DataFrame({'text': ls})
        df_tmp['input_ids'] = parallelize_apply(df_tmp['text'], tokenizer, args.max_seq_length, is_encoder_only, args.num_processes)
        df_tmp = df_tmp.set_index('text')

        df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)

        for i in range(1, num_neg+1):
            df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)

        df.to_parquet(f'{output_dir}/{ds_name}', index=False)


if __name__ == "__main__":
    main()