import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# 使用本地模型路径
model_path = 'models/bert-base-uncased'
max_seq_length = 512
root_dir = 'training_data'
output_dir = 'data_tokenized_bert'
os.makedirs(output_dir, exist_ok=True)

# 首先尝试加载本地模型
try:
    print(f"Trying to load tokenizer from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("Successfully loaded tokenizer from local path")
except Exception as e:
    print(f"Failed to load from local path: {e}")
    try:
        # 尝试使用国内镜像
        print("Trying to load tokenizer from HuggingFace with mirror...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', mirror='tuna')
        print("Successfully loaded tokenizer from mirror")
    except Exception as e2:
        print(f"Failed to load from mirror: {e2}")
        print("Creating a basic tokenizer for testing purposes...")
        # 创建基础tokenizer用于测试
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False, trust_remote_code=True)
        print("Successfully created basic tokenizer")

# ensure CLS/SEP if needed
if tokenizer.cls_token is None:
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
if tokenizer.sep_token is None:
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})

def batch_tokenize_texts(texts, batch_size=2048):
    """返回 list of list[int]"""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        toks = tokenizer(chunk,
                         padding=False,
                         truncation=True,
                         max_length=max_seq_length,
                         add_special_tokens=True)
        out.extend([ids for ids in toks['input_ids']])
    return out

for fname in tqdm(sorted(os.listdir(root_dir))):
    if not fname.endswith('.parquet'):
        continue
    df = pd.read_parquet(os.path.join(root_dir, fname))
    # tokenization of queries (batch)
    queries = df['query'].astype(str).tolist()
    q_input_ids = batch_tokenize_texts(queries)
    df['query_input_ids'] = q_input_ids

    # collect passages + negatives
    keys = ['passage'] + [k for k in df.columns if k.startswith('negative_')]
    # create unique list preserving order
    all_texts = []
    for k in keys:
        if k in df.columns:
            all_texts.extend(df[k].astype(str).tolist())
    # dedupe preserving order
    seen = {}
    unique_texts = []
    for t in all_texts:
        if t not in seen:
            seen[t] = True
            unique_texts.append(t)

    unique_input_ids = batch_tokenize_texts(unique_texts)
    mapping = dict(zip(unique_texts, unique_input_ids))

    # map back
    df['passage_input_ids'] = df['passage'].map(mapping)
    for k in df.columns:
        if k.startswith('negative_') and not k.endswith('_input_ids'):
            df[f'{k}_input_ids'] = df[k].map(mapping)

    # ensure lists (not numpy arrays) for parquet compatibility
    df['query_input_ids'] = df['query_input_ids'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else x)
    df['passage_input_ids'] = df['passage_input_ids'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else x)
    for k in df.columns:
        if k.endswith('_input_ids'):
            df[k] = df[k].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else x)

    df.to_parquet(os.path.join(output_dir, fname), index=False, engine='pyarrow')
