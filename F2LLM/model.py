import torch
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None,
                 model_id=None,
                 use_flash_attention=True,
                 torch_dtype=torch.bfloat16,
                 use_model_factory=True
                 ):
        """
        Initialize F2LLM model with flexible configuration support.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            max_seq_length: Maximum sequence length
            args: Training arguments (optional)
            model_id: Model registry ID for configuration (optional)
            use_flash_attention: Whether to use Flash Attention 2
            torch_dtype: Data type for model computations
            use_model_factory: Whether to use the new model factory system
        """
        
        self.args = args
        self.dtype = torch_dtype
        self.device = None # set after accelerator.prepare
        self.model_path = model_path
        self.model_id = model_id
        self.max_seq_length = max_seq_length
        
        # Try to use model factory if available
        if use_model_factory:
            try:
                from model_factory import get_factory
                factory = get_factory()
                logger.info("Using model factory for model initialization")
                self.lm = factory.create_model(
                    model_path,
                    model_id=model_id,
                    use_flash_attention=use_flash_attention,
                    torch_dtype=self.dtype
                )
                self.tokenizer = factory.create_tokenizer(model_path, model_id=model_id)
            except ImportError:
                logger.warning("Model factory not available, falling back to standard initialization")
                self._init_standard(use_flash_attention)
        else:
            self._init_standard(use_flash_attention)
    
    def _init_standard(self, use_flash_attention=True):
        """Standard model initialization (fallback)"""
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': self.dtype,
        }
        
        if use_flash_attention:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        
        logger.info(f"Initializing model from {self.model_path}")
        self.lm = AutoModel.from_pretrained(self.model_path, **model_kwargs)
        self.lm.config.use_cache = False
        
        logger.info(f"Initializing tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )

        passage_features_all_tokens = outputs.last_hidden_state
        return {
            'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
            'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
            'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
        }

