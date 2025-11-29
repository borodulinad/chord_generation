import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')



class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_size)
        
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        token_emb = self.token_embeddings(x)
        position_emb = self.position_embeddings(positions)
        
        embeddings = token_emb + position_emb
        return self.dropout(self.layer_norm(embeddings))
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.embedding_size = embedding_size
        
        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)
        self.out_linear = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_size)
        
        output = self.out_linear(context)
        return self.layer_norm(x + self.dropout(output))
    

class FCNNBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._linear1 = nn.Linear(embedding_size, hidden_size, bias=False)
        self._linear2 = nn.Linear(hidden_size, embedding_size, bias=False)
        self._activation = nn.GELU()
        self._layernorm = nn.LayerNorm(embedding_size)
        self._dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._linear2(self._activation(self._linear1(x)))
        return self._layernorm(x + self._dropout(z))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, fcnn_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(num_heads, embedding_size, dropout)
        self.fcnn = FCNNBlock(embedding_size, fcnn_hidden_size, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, mask)
        return self.fcnn(x)

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, embedding_size: int, 
                 num_heads: int, fcnn_hidden_size: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        
        self.embeddings = BERTEmbedding(vocab_size, embedding_size, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_size, num_heads, fcnn_hidden_size, dropout)
            for _ in range(n_layers)
        ])
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_size
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
        
    def forward(self, x: torch.LongTensor):
        mask = (x != 0).unsqueeze(1).unsqueeze(2)
        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x
    
class MaskedLanguageModel(nn.Module):
    def __init__(self, embedding_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        return self.linear(x) 
    
class BERTLM_MLM_And_Genre(nn.Module):
    def __init__(self, encoder: Encoder, num_genres: int):
        super().__init__()
        self._encoder = encoder
        self._mask_lm = MaskedLanguageModel(self._encoder.embedding_size, self._encoder.vocab_size)
        self._genre_classifier = nn.Linear(self._encoder.embedding_size, num_genres)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x: torch.LongTensor, return_genre_pred: bool = False):
        encoder_output = self._encoder(x)
        
        mlm_output = self._mask_lm(encoder_output)
        
        if return_genre_pred:
            cls_output = encoder_output[:, 0, :]
            genre_output = self._genre_classifier(cls_output)
            return mlm_output, genre_output
        else:
            return mlm_output