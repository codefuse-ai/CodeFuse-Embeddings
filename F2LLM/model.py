import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = None # set after accelerator.prepare
        
        # Check if CUDA is available and set the attention implementation accordingly
        # Only use flash_attention_2 if CUDA is available and flash_attn is installed
        attn_implementation = None
        if torch.cuda.is_available():
            try:
                import flash_attn
                attn_implementation = 'flash_attention_2'
            except ImportError:
                attn_implementation = 'eager'  # or 'sdpa' if available
        
        self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation=attn_implementation)
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        
        # MRL support
        self.mrl_enabled = getattr(args, 'mrl_enabled', False)
        if self.mrl_enabled:
            self.mrl_dims = args.mrl_dims
            # Create projection layers for each target dimension
            hidden_size = self.lm.config.hidden_size
            self.mrl_projections = nn.ModuleDict({
                str(dim): nn.Linear(hidden_size, dim) for dim in self.mrl_dims
            })
            # Move projection layers to the same device as the model
            self.mrl_projections.to(self.lm.device)

    def set_device(self):
        self.device = self.lm.device
        # Move MRL projections to the correct device if they exist
        if self.mrl_enabled:
            self.mrl_projections.to(self.device)
    
    def get_mrl_embeddings(self, full_embeddings, target_dim):
        """Get embeddings for a specific target dimension"""
        if not self.mrl_enabled:
            return full_embeddings
            
        if target_dim == self.lm.config.hidden_size:
            # No projection needed for full dimension
            return full_embeddings
        elif str(target_dim) in self.mrl_projections:
            # Use projection layer
            return self.mrl_projections[str(target_dim)](full_embeddings)
        else:
            # Fallback to truncation
            return full_embeddings[:, :target_dim]
    
    def get_all_mrl_embeddings(self, full_embeddings):
        """Get embeddings for all MRL dimensions"""
        if not self.mrl_enabled:
            return {str(self.lm.config.hidden_size): full_embeddings}
            
        embeddings_dict = {}
        # Full dimension
        embeddings_dict[str(self.lm.config.hidden_size)] = full_embeddings
        # Projected dimensions
        for dim in self.mrl_dims:
            embeddings_dict[str(dim)] = self.get_mrl_embeddings(full_embeddings, dim)
        return embeddings_dict
    
    def forward(self, batch, target_dim=None):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )

        passage_features_all_tokens = outputs.last_hidden_state
        # Extract [CLS] token embeddings
        full_embeddings = torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(len(batch['seq_lens']))])
        
        if target_dim is not None:
            # Return embeddings for specific dimension
            embeddings = self.get_mrl_embeddings(full_embeddings, target_dim)
        elif self.mrl_enabled:
            # Return embeddings for all dimensions
            embeddings_dict = self.get_all_mrl_embeddings(full_embeddings)
        else:
            # Return full dimension embeddings only
            embeddings = full_embeddings
            embeddings_dict = None
        
        # Split embeddings back to original format
        if self.mrl_enabled and target_dim is None:
            # Return dict with embeddings for all dimensions
            result = {}
            for dim, embs in embeddings_dict.items():
                result[f'query_passage_features_{dim}'] = embs[:bs]
                result[f'passage_passage_features_{dim}'] = embs[bs:2*bs]
                result[f'negative_passage_features_{dim}'] = None if num_hard_neg == 0 else embs[2*bs:].view(bs, num_hard_neg, -1)
            return result
        else:
            # Return single dimension embeddings
            query_embs = embeddings[:bs] if target_dim is not None or not self.mrl_enabled else embeddings_dict[str(self.lm.config.hidden_size)][:bs]
            passage_embs = embeddings[bs:2*bs] if target_dim is not None or not self.mrl_enabled else embeddings_dict[str(self.lm.config.hidden_size)][bs:2*bs]
            negative_embs = None if num_hard_neg == 0 else embeddings[2*bs:].view(bs, num_hard_neg, -1) if target_dim is not None or not self.mrl_enabled else embeddings_dict[str(self.lm.config.hidden_size)][2*bs:].view(bs, num_hard_neg, -1)
            
            return {
                'query_passage_features': query_embs,
                'passage_passage_features': passage_embs,
                'negative_passage_features': negative_embs
            }

