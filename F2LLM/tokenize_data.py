from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import os
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def get_tokenizer_config(tokenizer_type):
    if tokenizer_type == 'bert':
        return {
            'model_path': 'models/bert_multilingual',
            'max_seq_length': 512,
            'add_special_tokens': True,
            'add_eos_token': False,
            'output_dir': 'data_tokenized_bert'
        }
    elif tokenizer_type == 'qwen':
        return {
            'model_path': 'models/qwen3-0.6b',
            'max_seq_length': 1023,
            'add_special_tokens': False,
            'add_eos_token': True,
            'output_dir': 'data_tokenized_qwen'
        }
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def initialize_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    max_seq_length = config['max_seq_length']
    return tokenizer, max_seq_length


def process_sent(sentence, tokenizer, max_seq_length, config):
    # Process a single sentence with the specified tokenizer and configuration
    tokenizer_outputs = tokenizer(
        sentence, 
        max_length=max_seq_length, 
        truncation=True, 
        add_special_tokens=config['add_special_tokens']
    )
    
    input_ids = tokenizer_outputs.input_ids
    
    # Add EOS token for Qwen if required
    if config['add_eos_token']:
        input_ids = input_ids + [tokenizer.eos_token_id]
    
    return np.array(input_ids)


def process_sent_batch(s, tokenizer, max_seq_length, config):
    return s.apply(lambda x: process_sent(x, tokenizer, max_seq_length, config))

def parallelize(data, tokenizer, max_seq_length, config, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(partial(process_sent_batch, tokenizer=tokenizer, max_seq_length=max_seq_length, config=config), data_split))
    return data


def tokenize_dataset(tokenizer_type, root_dir='training_data'):
    config = get_tokenizer_config(tokenizer_type)
    tokenizer, max_seq_length = initialize_tokenizer(config)
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    for ds_name in tqdm(sorted(os.listdir(root_dir))):
        print(ds_name, flush=True)
        
        df = pd.read_parquet(f"{root_dir}/{ds_name}")
        df['query_input_ids'] = parallelize(
            df['query'], 
            tokenizer, 
            max_seq_length, 
            config, 
            62
        )
        
        num_neg = 24 if 'negative_2' in df.keys() else 1
        
        ls = df.passage.to_list()
        for i in range(1, num_neg+1):
            ls += df[f'negative_{i}'].to_list()
        ls = list(set(ls))
        df_tmp = pd.DataFrame({'text': ls})
        df_tmp['input_ids'] = parallelize(
            df_tmp['text'], 
            tokenizer, 
            max_seq_length, 
            config, 
            62
        )
        df_tmp = df_tmp.set_index('text')
        
        df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)
        
        for i in range(1, num_neg+1):
            df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)
        
        df.to_parquet(f'{output_dir}/{ds_name}', index=False)


def main():
    parser = argparse.ArgumentParser(description='Tokenize datasets using BERT or Qwen tokenizer')
    parser.add_argument('--tokenizer', type=str, choices=['bert', 'qwen'], default='bert',
                        help='Tokenizer type to use (default: bert)')
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='Directory containing the training data')
    
    args = parser.parse_args()
    
    tokenize_dataset(args.tokenizer, args.data_dir)


if __name__ == '__main__':
    main()