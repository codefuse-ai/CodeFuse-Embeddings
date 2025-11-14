from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def load_tokenizer(model_path, model_type='auto'):
    """
    Load tokenizer based on model type
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # For encoder models, ensure [CLS] token is available
    if model_type == 'encoder':
        if tokenizer.cls_token is None:
            # Add [CLS] token if not present
            tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        if tokenizer.sep_token is None:
            # Add [SEP] token if not present
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    return tokenizer


def process_sent(sentence, tokenizer, model_type='auto', max_seq_length=1023):
    # For encoder models, add [CLS] and [SEP] tokens
    if model_type == 'encoder':
        # We make sure there's always a [CLS] token at the beginning and [SEP] at the end
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True)
    else:
        # For decoder models, keep original behavior
        tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    
    # For decoder models, ensure eos token at the end
    if model_type != 'encoder' and tokenizer.eos_token_id is not None:
        input_ids = tokenizer_outputs.input_ids
        if len(input_ids) == 0 or input_ids[-1] != tokenizer.eos_token_id:
            input_ids = input_ids + [tokenizer.eos_token_id]
        return np.array(input_ids)
    else:
        return np.array(tokenizer_outputs.input_ids)


def process_sent_batch(s, tokenizer, model_type='auto', max_seq_length=1023):
    return s.apply(lambda x: process_sent(x, tokenizer, model_type, max_seq_length))


def parallelize(data, func, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(func, data_split))
    return data


# Configuration - can be passed as arguments
model_path = 'models/qwen3-0.6b'
model_type = 'auto'  # 'encoder', 'decoder', or 'auto'
max_seq_length = 1023

# Load tokenizer based on model type
tokenizer = load_tokenizer(model_path, model_type)

root_dir = 'training_data'
output_dir = 'data_tokenized'

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

for ds_name in tqdm(sorted(os.listdir(root_dir))):
    print(ds_name, flush=True)

    df = pd.read_parquet(f"{root_dir}/{ds_name}")
    df['query_input_ids'] = parallelize(df['query'], lambda s: process_sent_batch(s, tokenizer, model_type, max_seq_length), 62)

    num_neg = 24 if 'negative_2' in df.keys() else 1

    ls = df.passage.to_list()
    for i in range(1, num_neg+1):
        ls += df[f'negative_{i}'].to_list()
    ls = list(set(ls))
    df_tmp = pd.DataFrame({'text': ls})
    df_tmp['input_ids'] = parallelize(df_tmp['text'], lambda s: process_sent_batch(s, tokenizer, model_type, max_seq_length), 62)
    df_tmp = df_tmp.set_index('text')

    df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)

    for i in range(1, num_neg+1):
        df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)

    df.to_parquet(f'{output_dir}/{ds_name}', index=False)