#!/usr/bin/env python3
"""
通用的数据分词脚本，支持多种decoder-only模型
使用方法: python tokenize_data.py --model_path <模型路径> --max_seq_length <最大序列长度>
"""

import argparse
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Tokenize data for various decoder-only models')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the model or model name on HuggingFace')
    parser.add_argument('--max_seq_length', type=int, default=1023,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--data_dir', type=str, default='training_data',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='training_data',
                       help='Directory to save tokenized data')
    parser.add_argument('--num_processes', type=int, default=8,
                       help='Number of processes for parallel processing')
    return parser.parse_args()


def create_tokenizer(model_path, max_seq_length):
    """创建并配置分词器"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            # 添加新的pad_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Added new pad_token: {tokenizer.pad_token}")
    
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    return tokenizer, max_seq_length


def process_sent(sentence, tokenizer=None, max_seq_length=None):
    """处理单个句子，添加eos token并截断"""
    if tokenizer is None:
        raise ValueError("Tokenizer is required")
    
    # 分词，不添加特殊token，因为我们手动添加eos
    tokenizer_outputs = tokenizer(
        sentence, 
        max_length=max_seq_length, 
        truncation=True, 
        add_special_tokens=False
    )
    
    # 添加eos token
    input_ids = tokenizer_outputs.input_ids + [tokenizer.eos_token_id]
    
    return np.array(input_ids)


def process_sent_batch(s, tokenizer=None, max_seq_length=None):
    """批量处理句子"""
    return s.apply(lambda x: process_sent(x, tokenizer, max_seq_length))


def parallelize(data, func, num_of_processes=8, **kwargs):
    """并行处理数据"""
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    
    with Pool(num_of_processes) as pool:
        # 使用starmap传递多个参数
        data = pd.concat(pool.starmap(func, [(d, ) + tuple(kwargs.values()) for d in data_split]))
    return data


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建分词器
    tokenizer, max_seq_length = create_tokenizer(args.model_path, args.max_seq_length)
    
    # 获取模型名称用于输出目录
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_subdir = os.path.join(args.output_dir, f"data_tokenized_{model_name}")
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"Processing data from {args.data_dir}...")
    print(f"Output directory: {output_subdir}")
    
    # 处理所有数据集
    for ds_name in tqdm(sorted(os.listdir(args.data_dir)), desc="Processing datasets"):
        if not ds_name.endswith('.parquet'):
            continue
            
        print(f"\nProcessing {ds_name}...")
        
        # 读取数据
        df = pd.read_parquet(os.path.join(args.data_dir, ds_name))
        
        # 处理查询
        print("Processing queries...")
        df['query_input_ids'] = parallelize(
            df['query'], 
            process_sent_batch, 
            args.num_processes,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length
        )
        
        # 确定负样本数量
        num_neg = 24 if 'negative_2' in df.columns else 1
        print(f"Number of negative samples: {num_neg}")
        
        # 收集所有passage和负样本
        all_passages = df['passage'].to_list()
        for i in range(1, num_neg + 1):
            if f'negative_{i}' in df.columns:
                all_passages += df[f'negative_{i}'].to_list()
        
        # 去重
        all_passages = list(set(all_passages))
        
        # 创建临时DataFrame处理passage
        df_tmp = pd.DataFrame({'text': all_passages})
        print(f"Processing {len(all_passages)} unique passages...")
        
        df_tmp['input_ids'] = parallelize(
            df_tmp['text'], 
            process_sent_batch, 
            args.num_processes,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length
        )
        
        # 设置索引以便映射
        df_tmp = df_tmp.set_index('text')
        
        # 映射passage的input_ids
        print("Mapping passages...")
        df['passage_input_ids'] = df['passage'].map(df_tmp['input_ids'])
        
        # 映射负样本的input_ids
        for i in range(1, num_neg + 1):
            neg_col = f'negative_{i}'
            neg_input_col = f'negative_{i}_input_ids'
            if neg_col in df.columns:
                df[neg_input_col] = df[neg_col].map(df_tmp['input_ids'])
        
        # 保存结果
        output_path = os.path.join(output_subdir, ds_name)
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")
        
        # 打印统计信息
        print(f"Dataset size: {len(df)}")
        print(f"Query avg length: {df['query_input_ids'].apply(len).mean():.1f}")
        print(f"Passage avg length: {df['passage_input_ids'].apply(len).mean():.1f}")
    
    print(f"\nAll datasets processed successfully!")
    print(f"Tokenized data saved to: {output_subdir}")


if __name__ == "__main__":
    main()
