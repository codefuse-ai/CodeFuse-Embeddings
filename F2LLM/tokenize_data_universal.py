from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
import json
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from model_factory import ModelFactory


class UniversalTokenizer:
    """Universal tokenizer supporting multiple model types"""
    
    def __init__(self, model_config_path):
        """Initialize universal tokenizer
        
        Args:
            model_config_path: Path to model config file or model directory
        """
        self.model_config = self._load_model_config(model_config_path)
        self.model_type = self.model_config.get('model_type', 'unknown')
        self.model_path = self.model_config.get('model_path', model_config_path)
        self.max_seq_length = self.model_config.get('max_seq_length', 1023)
        
        # Create adapter and load tokenizer using model factory
        self.adapter = ModelFactory.create_adapter(self.model_path)
        self.tokenizer = self.adapter.load_tokenizer()
        
        # Apply tokenizer configuration
        self._apply_tokenizer_config()
    
    def _load_model_config(self, config_path):
        """Load model configuration"""
        if os.path.isfile(config_path) and config_path.endswith('.json'):
            with open(config_path) as f:
                return json.load(f)
        elif os.path.isdir(config_path):
            # If directory, try to load config.json
            config_file = os.path.join(config_path, 'config.json')
            if os.path.exists(config_file):
                with open(config_file) as f:
                    return json.load(f)
        return {'model_path': config_path}
    
    def _apply_tokenizer_config(self):
        """Apply tokenizer configuration"""
        tokenizer_config = self.model_config.get('tokenizer_config', {})
        
        if 'padding_side' in tokenizer_config:
            self.tokenizer.padding_side = tokenizer_config['padding_side']
        
        if 'pad_token' in tokenizer_config:
            if tokenizer_config['pad_token'] is not None:
                self.tokenizer.pad_token = tokenizer_config['pad_token']
            elif self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize(self, text):
        """Universal tokenize method"""
        add_special_tokens = self.model_config.get('tokenizer_config', {}).get('add_special_tokens', False)
        
        tokenizer_outputs = self.tokenizer(
            text, 
            max_length=self.max_seq_length, 
            truncation=True, 
            add_special_tokens=add_special_tokens
        )
        return np.array(tokenizer_outputs.input_ids + [self.tokenizer.eos_token_id])


def process_sent(sentence, tokenizer):
    """Process single sentence"""
    add_special_tokens = tokenizer.model_config.get('tokenizer_config', {}).get('add_special_tokens', False)
    
    tokenizer_outputs = tokenizer.tokenizer(
        sentence, 
        max_length=tokenizer.max_seq_length, 
        truncation=True, 
        add_special_tokens=add_special_tokens
    )
    return np.array(tokenizer_outputs.input_ids + [tokenizer.tokenizer.eos_token_id])


def process_sent_batch(s, tokenizer):
    """Process sentences in batch"""
    return s.apply(lambda x: process_sent(x, tokenizer))


def parallelize(data, func, num_of_processes=8, tokenizer=None):
    """Parallel processing of data"""
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.starmap(func, [(df, tokenizer) for df in data_split]))
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal data tokenization script')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config file or model directory')
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='data_tokenized',
                        help='Output directory for tokenized data')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes for parallel processing')
    
    args = parser.parse_args()
    
    # Initialize universal tokenizer
    tokenizer = UniversalTokenizer(args.model_config)
    
    print(f"Model type: {tokenizer.model_type}")
    print(f"Model path: {tokenizer.model_path}")
    print(f"Vocab size: {len(tokenizer.tokenizer)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process data - exactly same as tokenize_data_qwen.py
    root_dir = args.data_dir
    for ds_name in tqdm(sorted(os.listdir(root_dir))):
        if not ds_name.endswith('.parquet'):
            continue
            
        print(f"Processing {ds_name}...", flush=True)
        
        df = pd.read_parquet(os.path.join(root_dir, ds_name))
        
        # Process query - use parallel processing, no deduplication
        df['query_input_ids'] = parallelize(df['query'], process_sent_batch, args.num_processes, tokenizer)

        # Determine number of negative samples
        num_neg = 24 if 'negative_2' in df.columns else 1

        # Collect all passages and negatives (no deduplication for query)
        ls = df.passage.to_list()
        for i in range(1, num_neg+1):
            if f'negative_{i}' in df.columns:
                ls += df[f'negative_{i}'].to_list()
        
        # Deduplicate passages and negatives (exactly same as tokenize_data_qwen.py)
        ls = list(set(ls))
        df_tmp = pd.DataFrame({'text': ls})
        df_tmp['input_ids'] = parallelize(df_tmp['text'], process_sent_batch, args.num_processes, tokenizer)
        df_tmp = df_tmp.set_index('text')

        # Apply mappings
        df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)

        for i in range(1, num_neg+1):
            if f'negative_{i}' in df.columns:
                df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)

        # Save results
        output_path = os.path.join(args.output_dir, ds_name)
        df.to_parquet(output_path, index=False)
        print(f"Saved tokenized data to {output_path}")


if __name__ == "__main__":
    main()