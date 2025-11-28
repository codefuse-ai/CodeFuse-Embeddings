import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


class BaseEmbeddingAdapter:
    """基础嵌入适配器"""
    
    def get_embeddings(self, outputs, batch):
        """获取嵌入向量"""
        passage_features_all_tokens = outputs.last_hidden_state
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)
        
        return {
            'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
            'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
            'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
        }


class BertEmbeddingAdapter(BaseEmbeddingAdapter):
    """BERT系列模型的嵌入适配器"""
    
    def get_embeddings(self, outputs, batch):
        """获取BERT模型的嵌入向量，使用[CLS]标记"""
        passage_features_all_tokens = outputs.last_hidden_state
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)
        
        # BERT使用[CLS]标记的嵌入 (第0个位置)
        return {
            'query_passage_features': passage_features_all_tokens[:bs, 0, :].unsqueeze(1),
            'passage_passage_features': passage_features_all_tokens[bs:2*bs, 0, :].unsqueeze(1),
            'negative_passage_features': None if num_hard_neg == 0 else passage_features_all_tokens[2*bs:, 0, :].view(bs, num_hard_neg, -1)
        }


class LlamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """LLaMA系列模型的嵌入适配器"""
    
    def get_embeddings(self, outputs, batch):
        """获取LLaMA模型的嵌入向量，使用最后一个标记"""
        passage_features_all_tokens = outputs.last_hidden_state
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)
        
        # LLaMA使用最后一个标记的嵌入
        return {
            'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
            'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
            'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
        }


class MeanPoolingAdapter(BaseEmbeddingAdapter):
    """平均池化适配器"""
    
    def get_embeddings(self, outputs, batch):
        """使用平均池化获取嵌入向量"""
        passage_features_all_tokens = outputs.last_hidden_state
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)
        
        # 计算平均池化嵌入
        attention_mask = batch['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(passage_features_all_tokens.size()).float()
        embeddings = torch.sum(passage_features_all_tokens * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 重新组织输出格式
        query_embeddings = embeddings[:bs]
        passage_embeddings = embeddings[bs:2*bs]
        negative_embeddings = None if num_hard_neg == 0 else embeddings[2*bs:].view(bs, num_hard_neg, -1)
        
        return {
            'query_passage_features': query_embeddings.unsqueeze(1),
            'passage_passage_features': passage_embeddings.unsqueeze(1),
            'negative_passage_features': negative_embeddings
        }


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None,
                 model_type="auto",
                 embedding_strategy="last_token",
                 pooling_strategy="cls"
                 ):

        self.args = args
        self.dtype = torch.bfloat16
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
        
        # 设置模型类型和嵌入策略
        self.model_type = self.lm.config.model_type if model_type == "auto" else model_type
        self.embedding_strategy = embedding_strategy
        self.pooling_strategy = pooling_strategy
        
        # 根据模型类型和策略选择适配器
        self.embedding_adapter = self._get_embedding_adapter()

    def _get_embedding_adapter(self):
        """根据模型类型和策略返回相应的嵌入适配器"""
        # 如果指定了池化策略，使用池化适配器
        if self.pooling_strategy != "cls":
            if self.pooling_strategy == "mean":
                return MeanPoolingAdapter()
            # 可以添加更多池化策略
        
        # 根据模型类型选择适配器
        if self.model_type in ['bert', 'roberta', 'distilbert', 'albert', 'electra']:
            return BertEmbeddingAdapter()
        elif self.model_type in ['llama', 'mistral', 'qwen', 'gemma', 'phi']:
            return LlamaEmbeddingAdapter()
        else:
            # 默认适配器
            return BaseEmbeddingAdapter()

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch):
        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )
        
        # 使用适配器获取嵌入
        return self.embedding_adapter.get_embeddings(outputs, batch)

    def encode(self, texts, max_length=None):
        """编码文本为嵌入向量"""
        if max_length is None:
            max_length = self.max_seq_length
            
        # 对文本进行tokenize
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        
        # 如果模型有设备信息，将输入移动到相应设备
        if self.device is not None:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
        # 获取模型输出
        with torch.no_grad():
            outputs = self.lm(**encoded)
            
        # 构造batch字典用于适配器处理
        batch = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'seq_lens': encoded['attention_mask'].sum(dim=1).tolist(),
            'bs': len(texts)
        }
        
        # 使用适配器获取嵌入
        embeddings_dict = self.embedding_adapter.get_embeddings(outputs, batch)
        
        # 返回查询嵌入作为默认输出
        return embeddings_dict['query_passage_features'].squeeze(1)

