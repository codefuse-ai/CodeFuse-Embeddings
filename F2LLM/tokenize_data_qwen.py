from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from transformers import set_seed


tokenizer = AutoTokenizer.from_pretrained('models/qwen3-0.6b')
max_seq_length = 2047


def process_sent(sentence):

    # We make sure there's always an eos token at the end of each sequence
    tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True)

    return np.array(tokenizer_outputs.input_ids + [tokenizer.eos_token_id])


def process_sent_batch(s):
    return s.apply(process_sent)

def parallelize(data, func, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(func, data_split))
    return data


root_dir = 'raw_training_data'
tgt_dir = 'tokenized_training_data'
num_proc = 40
n = 80000

set_seed(42)
for ds_name in tqdm(sorted(os.listdir(root_dir))):

    ds_prefix = ds_name.split('.')[0]
    print(ds_prefix, flush=True)

    # if os.path.exists(f'{tgt_dir}/{ds_prefix}_query.parquet'):
    #     continue
    df = pd.read_parquet(f"{root_dir}/{ds_name}")
    if len(df) > n:
        df = df.sample(n=n).reset_index(drop=True)
    print(df.loc[0].query, flush=True)

    df['query_input_ids'] = parallelize(df['query'], process_sent_batch, num_proc)

    num_neg = 24 if 'negative_2' in df.keys() else 1

    ls = df.passage.to_list()
    for i in range(1, num_neg+1):
        ls += df[f'negative_{i}'].to_list()
    ls = list(set(ls))
    df_tmp = pd.DataFrame({'text': ls})
    df_tmp['input_ids'] = parallelize(df_tmp['text'], process_sent_batch, num_proc)

    df_tmp['doc_id'] = [f"{ds_prefix}_{i}" for i in range(len(df_tmp))]

    mapping_dict = df_tmp.set_index('text')['doc_id'].to_dict()


    df['passage_input_ids'] = df.passage.map(mapping_dict)

    for i in range(1, num_neg+1):
        df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(mapping_dict)

    df.drop(columns=[f'negative_{i}' for i in range(1, num_neg+1)]+['passage', 'query'], inplace=True)

    df.to_parquet(f'{tgt_dir}/{ds_prefix}_query.parquet', index=False)
    df_tmp = df_tmp.drop(columns=['text']).set_index('doc_id')
    df_tmp.to_parquet(f'{tgt_dir}/{ds_prefix}_corpus.parquet')
