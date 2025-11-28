from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F


def process_sent(sentence, tokenizer, max_seq_length, add_eos_token=False, device='cpu'):
    """Process a single sentence with the given tokenizer"""
    if add_eos_token:
        # For decoder-only models like Qwen, LLaMA, etc.
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False)
        return np.array(tokenizer_outputs.input_ids + [tokenizer.eos_token_id])
    else:
        # For encoder models like BERT, RoBERTa, etc.
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, padding=False)
        return np.array(tokenizer_outputs.input_ids)


def process_sent_batch_serial(s, tokenizer, max_seq_length, add_eos_token=False, device='cpu'):
    """Process a batch of sentences in serial mode"""
    return s.apply(lambda x: process_sent(x, tokenizer, max_seq_length, add_eos_token, device))


def process_sent_batch_parallel(s, tokenizer, max_seq_length, add_eos_token=False, num_processes=8, device='cpu'):
    """Process a batch of sentences in parallel mode"""
    def _process_single(sentence):
        return process_sent(sentence, tokenizer, max_seq_length, add_eos_token, device)
    
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


def process_sent_batch_gpu(s, tokenizer, max_seq_length, add_eos_token=False, device='cuda'):
    """Process a batch of sentences using GPU acceleration"""
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU processing")
            return process_sent_batch_serial(s, tokenizer, max_seq_length, add_eos_token, 'cpu')
        
        # Move tokenizer to GPU if possible
        if hasattr(tokenizer, 'to'):
            tokenizer = tokenizer.to(device)
        
        # Process in batches to avoid GPU memory issues
        batch_size = 1000  # Adjust based on GPU memory
        results = []
        
        for i in range(0, len(s), batch_size):
            batch_sentences = s.iloc[i:i+batch_size].tolist()
            
            # Tokenize batch
            if add_eos_token:
                # For decoder-only models
                tokenizer_outputs = tokenizer(
                    batch_sentences, 
                    max_length=max_seq_length, 
                    truncation=True, 
                    add_special_tokens=False,
                    padding=True,
                    return_tensors='pt'
                )
                
                # Add EOS token to each sequence
                input_ids = tokenizer_outputs.input_ids.to(device)
                eos_tokens = torch.full((input_ids.shape[0], 1), tokenizer.eos_token_id, device=device)
                input_ids = torch.cat([input_ids, eos_tokens], dim=1)
                
                # Convert to list of numpy arrays
                for ids in input_ids.cpu().numpy():
                    results.append(np.array(ids))
            else:
                # For encoder models
                tokenizer_outputs = tokenizer(
                    batch_sentences, 
                    max_length=max_seq_length, 
                    truncation=True, 
                    padding=True,
                    return_tensors='pt'
                )
                
                input_ids = tokenizer_outputs.input_ids.to(device)
                
                # Convert to list of numpy arrays
                for ids in input_ids.cpu().numpy():
                    # Remove padding tokens
                    non_pad_tokens = ids[ids != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else ids
                    results.append(np.array(non_pad_tokens))
        
        return pd.Series(results, index=s.index)
        
    except Exception as e:
        print(f"GPU processing failed: {e}, falling back to CPU processing")
        return process_sent_batch_serial(s, tokenizer, max_seq_length, add_eos_token, 'cpu')


def tokenize_data(data, column, tokenizer, max_seq_length, add_eos_token=False, use_parallel=True, num_processes=8, use_gpu=False, device='cpu'):
    """Tokenize data with option for serial, parallel, or GPU processing"""
    if use_gpu:
        return process_sent_batch_gpu(data[column], tokenizer, max_seq_length, add_eos_token, device)
    elif use_parallel and num_processes > 1:
        return process_sent_batch_parallel(data[column], tokenizer, max_seq_length, add_eos_token, num_processes, device)
    else:
        return process_sent_batch_serial(data[column], tokenizer, max_seq_length, add_eos_token, device)


def tokenize_dataset(model_path, max_seq_length, add_eos_token, input_dir, output_dir, use_parallel=True, num_processes=8, use_gpu=False, device='cpu'):
    """Tokenize datasets with the specified model"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add padding token if not present (for LLaMA, Qwen, etc.)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    mode_str = 'GPU' if use_gpu else ('Parallel' if use_parallel else 'Serial')
    process_str = f"{num_processes if use_parallel and not use_gpu else 1} processes" if not use_gpu else "GPU acceleration"
    print(f"Processing mode: {mode_str} with {process_str}")
    
    for ds_name in tqdm(sorted(os.listdir(input_dir))):
        if not ds_name.endswith('.parquet'):
            continue
            
        print(f"Processing {ds_name}", flush=True)

        df = pd.read_parquet(f"{input_dir}/{ds_name}")
        df['query_input_ids'] = tokenize_data(
            df, 'query', tokenizer, max_seq_length, add_eos_token, use_parallel, num_processes, use_gpu, device
        )

        num_neg = 24 if 'negative_2' in df.keys() else 1

        ls = df.passage.to_list()
        for i in range(1, num_neg+1):
            if f'negative_{i}' in df.columns:
                ls += df[f'negative_{i}'].to_list()
        ls = list(set(ls))
        df_tmp = pd.DataFrame({'text': ls})
        df_tmp['input_ids'] = tokenize_data(
            df_tmp, 'text', tokenizer, max_seq_length, add_eos_token, use_parallel, num_processes, use_gpu, device
        )
        df_tmp = df_tmp.set_index('text')

        df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)

        for i in range(1, num_neg+1):
            col_name = f'negative_{i}'
            if col_name in df.columns:
                new_col_name = f'negative_{i}_input_ids'
                df[new_col_name] = df[col_name].map(df_tmp.input_ids)

        df.to_parquet(f'{output_dir}/{ds_name}', index=False)


def main():
    parser = argparse.ArgumentParser(description='Tokenize datasets for different models')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--model_type', type=str, choices=['bert', 'llama', 'qwen'], 
                        help='Type of model (affects tokenization strategy)')
    parser.add_argument('--max_seq_length', type=int, help='Maximum sequence length')
    parser.add_argument('--input_dir', type=str, help='Input directory with parquet files')
    parser.add_argument('--output_dir', type=str, help='Output directory for tokenized data')
    parser.add_argument('--num_processes', type=int, help='Number of processes for parallel processing (0 or 1 for serial)')
    parser.add_argument('--mode', type=str, choices=['auto', 'serial', 'parallel', 'gpu'], 
                        help='Processing mode: auto (based on num_processes), serial, parallel, or gpu')
    parser.add_argument('--device', type=str, help='Device to use for processing (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    def get_param(param_name, default=None):
        arg_value = getattr(args, param_name)
        return arg_value if arg_value is not None else config.get(param_name, default)
    
    model_path = get_param('model_path')
    model_type = get_param('model_type')
    max_seq_length = get_param('max_seq_length', 512)
    input_dir = get_param('input_dir', 'training_data')
    output_dir = get_param('output_dir')
    num_processes = get_param('num_processes', 1)
    mode = get_param('mode', 'auto')
    device = get_param('device', 'cpu')
    
    # Validate required parameters
    if not model_path:
        raise ValueError("model_path is required")
    if not model_type:
        raise ValueError("model_type is required")
    if not output_dir:
        raise ValueError("output_dir is required")
    
    # Determine processing mode
    if mode == 'gpu':
        use_gpu = True
        use_parallel = False
    elif mode == 'serial':
        use_gpu = False
        use_parallel = False
    elif mode == 'parallel':
        use_gpu = False
        use_parallel = True
    else:  # auto mode
        use_gpu = False
        use_parallel = num_processes > 1
    
    # Determine if EOS token should be added based on model type
    add_eos_token = model_type in ['llama', 'qwen']
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available, falling back to CPU")
        device = 'cpu'
    
    # If mode is gpu but device is cpu, switch to parallel or serial
    if mode == 'gpu' and device == 'cpu':
        use_gpu = False
        use_parallel = num_processes > 1
        print("Switching from GPU to CPU processing")
    
    tokenize_dataset(
        model_path=model_path,
        max_seq_length=max_seq_length,
        add_eos_token=add_eos_token,
        input_dir=input_dir,
        output_dir=output_dir,
        use_parallel=use_parallel,
        num_processes=num_processes,
        use_gpu=use_gpu,
        device=device
    )


if __name__ == "__main__":
    main()

