"""
Generic tokenization module supporting multiple model families.

This module replaces the Qwen-specific tokenizer and provides
support for various tokenization strategies across different models.
"""

from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging
from typing import Optional, Callable

from model_registry import get_registry, TokenizerType
try:
    from huggingface_hub.errors import GatedRepoError
except Exception:  # huggingface_hub may not expose errors in older versions
    class GatedRepoError(Exception):
        pass

logger = logging.getLogger(__name__)


class GenericTokenizer:
    """Flexible tokenizer supporting multiple model families"""
    
    def __init__(
        self,
        model_path: str,
        model_id: Optional[str] = None,
        max_seq_length: int = 1023,
        num_processes: int = 8,
        add_eos_token: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize generic tokenizer.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            model_id: Optional model registry ID
            max_seq_length: Maximum sequence length
            num_processes: Number of processes for parallel tokenization
            add_eos_token: Whether to add EOS token at the end
        """
        self.model_path = model_path
        self.model_id = model_id
        self.max_seq_length = max_seq_length
        self.num_processes = num_processes
        self.add_eos_token = add_eos_token
        
        # Load tokenizer (support gated repos via token if provided or via CLI login)
        logger.info(f"Loading tokenizer from {model_path}")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        try:
            if self.hf_token:
                try:
                    # Newer API (huggingface_hub>=0.14)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        token=self.hf_token,
                    )
                except TypeError:
                    # Older transformers API
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        use_auth_token=self.hf_token,
                    )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
        except GatedRepoError as e:
            raise SystemExit(
                "Access to this model is gated.\n"
                "Please request/accept access on Hugging Face and authenticate:\n"
                "  1) Visit the model page and accept terms (e.g., https://huggingface.co/meta-llama/Llama-2-7b)\n"
                "  2) Login: `huggingface-cli login` (or set HF_TOKEN env var)\n"
                "  3) Re-run this command.\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise
        
        # Get model config if available
        self.model_config = None
        if model_id:
            registry = get_registry()
            if registry.supports_model(model_id):
                self.model_config = registry.get(model_id)
                logger.info(f"Using model configuration: {model_id}")
        
        # Get EOS token
        self.eos_token_id = self._get_eos_token_id()
        logger.info(f"Using EOS token ID: {self.eos_token_id}")
    
    def _get_eos_token_id(self) -> int:
        """Get appropriate EOS token ID"""
        if self.model_config and self.model_config.eos_token_id is not None:
            return self.model_config.eos_token_id
        
        # Try common EOS token IDs
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        
        # Fallback to common defaults
        common_eos = [2, 151643, 151645]  # Common across different models
        for token_id in common_eos:
            if token_id < self.tokenizer.vocab_size:
                logger.warning(f"Using fallback EOS token ID: {token_id}")
                return token_id
        
        raise ValueError("Cannot determine EOS token ID")
    
    def tokenize_sentence(self, sentence: str) -> np.ndarray:
        """
        Tokenize a single sentence.
        
        Returns:
            Numpy array of token IDs with EOS token appended
        """
        tokenizer_outputs = self.tokenizer(
            sentence,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=False
        )
        
        input_ids = tokenizer_outputs.input_ids
        
        if self.add_eos_token:
            input_ids = input_ids + [self.eos_token_id]
        
        return np.array(input_ids)
    
    def tokenize_batch(self, texts: pd.Series) -> pd.Series:
        """Tokenize a batch of texts"""
        return texts.apply(self.tokenize_sentence)
    
    def parallelize_tokenization(
        self,
        data: pd.DataFrame,
        text_column: str,
        output_column: str
    ) -> pd.DataFrame:
        """
        Tokenize a dataframe column in parallel.
        
        Args:
            data: Dataframe containing text to tokenize
            text_column: Column name with text data
            output_column: Column name for output tokens
        
        Returns:
            Dataframe with added tokenized column
        """
        logger.info(f"Tokenizing {len(data)} texts (sequential mode)")
        
        indices = np.array_split(data.index, max(1, self.num_processes))
        data_split = [data.loc[idx] for idx in indices]
        
        # Avoid multiprocessing pickling issues on macOS by processing sequentially
        parts = [self._tokenize_dataframe(df, text_column) for df in data_split]
        tokenized = pd.concat(parts)
        
        data[output_column] = tokenized
        return data
    
    def _tokenize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> pd.Series:
        """Helper for parallel tokenization"""
        return df[text_column].apply(self.tokenize_sentence)


def tokenize_dataset(
    root_dir: str,
    output_dir: str,
    model_path: str,
    model_id: Optional[str] = None,
    max_seq_length: int = 1023,
    num_processes: int = 8,
    add_eos_token: bool = True,
    hf_token: Optional[str] = None,
):
    """
    Tokenize all parquet files in a directory.
    
    Args:
        root_dir: Input directory with parquet files
        output_dir: Output directory for tokenized data
        model_path: Path to model for tokenizer
        model_id: Optional model registry ID
        max_seq_length: Maximum sequence length
        num_processes: Number of parallel processes
        add_eos_token: Whether to add EOS token
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = GenericTokenizer(
        model_path,
        model_id=model_id,
        max_seq_length=max_seq_length,
        num_processes=num_processes,
        add_eos_token=add_eos_token,
        hf_token=hf_token,
    )
    
    logger.info(f"Processing datasets from {root_dir} (recursive)")
    
    for dirpath, _, filenames in os.walk(root_dir):
        parquet_files = sorted([f for f in filenames if f.endswith('.parquet')])
        for ds_name in tqdm(parquet_files):
            input_path = os.path.join(dirpath, ds_name)
            rel_name = os.path.relpath(input_path, root_dir)
            logger.info(f"Processing: {rel_name}")
            
            df = pd.read_parquet(input_path)
        
        # Tokenize queries
        df = tokenizer.parallelize_tokenization(
            df, 'query', 'query_input_ids'
        )
        
        # Determine number of negatives
        num_neg = 24 if 'negative_2' in df.columns else 1
        
        # Tokenize passages (collect unique texts first)
        ls = df['passage'].tolist()
        for i in range(1, num_neg + 1):
            ls += df[f'negative_{i}'].tolist()
        
        ls = list(set(ls))
        df_tmp = pd.DataFrame({'text': ls})
        
        df_tmp = tokenizer.parallelize_tokenization(
            df_tmp, 'text', 'input_ids'
        )
        df_tmp = df_tmp.set_index('text')
        
        # Map tokenized passages back
        df['passage_input_ids'] = df['passage'].map(df_tmp['input_ids'])
        
        for i in range(1, num_neg + 1):
            df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp['input_ids'])
        
        # Save tokenized data
        output_path = os.path.join(output_dir, rel_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved tokenized data to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tokenize datasets for F2LLM training")
    parser.add_argument("--root_dir", type=str, default="training_data",
                       help="Input directory with parquet files")
    parser.add_argument("--output_dir", type=str, default="data_tokenized_generic",
                       help="Output directory for tokenized data")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model or HuggingFace model ID")
    parser.add_argument("--model_id", type=str, default=None,
                       help="Model registry ID for configuration")
    parser.add_argument("--max_seq_length", type=int, default=1023,
                       help="Maximum sequence length")
    parser.add_argument("--num_processes", type=int, default=8,
                       help="Number of parallel processes")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Optional Hugging Face token for gated repos (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    tokenize_dataset(
        args.root_dir,
        args.output_dir,
        args.model_path,
        model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        num_processes=args.num_processes,
        hf_token=args.hf_token,
    )
