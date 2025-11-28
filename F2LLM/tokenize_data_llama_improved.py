from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def process_sent(sentence, tokenizer, max_seq_length):
    """Process a single sentence with the given tokenizer for LLaMA models"""
    # LLaMA tokenizer - similar to Qwen but with different max length
    tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    return np.array(tokenizer_outputs.input_ids + [tokenizer.eos_token_id])


def process_sent_batch_serial(s, tokenizer, max_seq_length):
    """Process a batch of sentences in serial mode"""
    return s.apply(lambda x: process_sent(x, tokenizer, max_seq_length))


def process_sent_batch_parallel(s, tokenizer, max_seq_length, num_processes=8):
    """Process a batch of sentences in parallel mode"""
    def _process_single(sentence):
        return process_sent(sentence, tokenizer, max_seq_length)
    
    def _process_batch(batch_s):
        return batch_s.apply(_process_single)
    
    indices = np.array_split(s.index, min(num_processes, len(s)))
    data_split = [s.iloc[idx] for idx in indices if len(idx) > 0]
    
    if len(data_split) == 0:
        return pd.Series([], dtype=object)
    
    with Pool(min(num_processes, len(data_split))) as pool:
        results = pool.map(_process_batch, data_split)
    
    if results:
        return pd.concat(results)
    else:
        return pd.Series([], dtype=object)


def tokenize_data(data, column, tokenizer, max_seq_length, use_parallel=True, num_processes=8):
    """Tokenize data with option for serial or parallel processing"""
    if use_parallel and num_processes > 1:
        return process_sent_batch_parallel(data[column], tokenizer, max_seq_length, num_processes)
    else:
        return process_sent_batch_serial(data[column], tokenizer, max_seq_length)


def main():
    parser = argparse.ArgumentParser(description='Tokenize datasets for LLaMA models')
    parser.add_argument('--model_path', type=str, default='models/llama-3-8b', help='Path to the LLaMA model')
    parser.add_argument('--input_dir', type=str, default='training_data', help='Input directory with parquet files')
    parser.add_argument('--output_dir', type=str, default='data_tokenized_llama', help='Output directory for tokenized data')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes for parallel processing (0 or 1 for serial)')
    parser.add_argument('--mode', type=str, choices=['auto', 'serial', 'parallel'], default='auto', 
                        help='Processing mode: auto (based on num_processes), serial, or parallel')
    
    args = parser.parse_args()
    
    # Determine processing mode
    if args.mode == 'serial':
        use_parallel = False
    elif args.mode == 'parallel':
        use_parallel = True
    else:  # auto mode
        use_parallel = args.num_processes > 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Add padding token if not present
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing mode: {'Parallel' if use_parallel else 'Serial'} with {args.num_processes if use_parallel else 1} processes")
    
    for ds_name in tqdm(sorted(os.listdir(args.input_dir))):
        if not ds_name.endswith('.parquet'):
            continue
            
        print(f"Processing {ds_name}", flush=True)
        
        df = pd.read_parquet(f"{args.input_dir}/{ds_name}")
        
        # Tokenize query column
        df['query_input_ids'] = tokenize_data(
            df, 'query', tokenizer, args.max_seq_length, use_parallel, args.num_processes
        )
        
        # Handle passage and negative samples
        num_neg = 24 if 'negative_2' in df.keys() else 1
        
        ls = df.passage.to_list()
        for i in range(1, num_neg+1):
            if f'negative_{i}' in df.columns:
                ls += df[f'negative_{i}'].to_list()
        ls = list(set(ls))
        
        df_tmp = pd.DataFrame({'text': ls})
        df_tmp['input_ids'] = tokenize_data(
            df_tmp, 'text', tokenizer, args.max_seq_length, use_parallel, args.num_processes
        )
        df_tmp = df_tmp.set_index('text')
        
        # Map tokenized passages back to original dataframe
        df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)
        
        for i in range(1, num_neg+1):
            col_name = f'negative_{i}'
            if col_name in df.columns:
                new_col_name = f'negative_{i}_input_ids'
                df[new_col_name] = df[col_name].map(df_tmp.input_ids)
        
        # Save tokenized data
        output_path = f'{args.output_dir}/{ds_name}'
        df.to_parquet(output_path, index=False)
        print(f"Saved tokenized data to {output_path}")


if __name__ == "__main__":
    main()

