import numpy as np
import polars as pl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

import re
from typing import List, Dict, Any, Tuple, Optional, Mapping, Set, Self, NamedTuple, TypedDict

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self._theta = 1 / (torch.pow(torch.tensor(base), (torch.arange(0, embedding_size, 2).float() / embedding_size)))
        self._theta = self._theta.repeat_interleave(2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(0, x.size(-2), device=x.device)
        position_matrix = torch.outer(position_ids, self._theta.to(x.device))
        cos = torch.cos(position_matrix)
        sin = torch.sin(position_matrix)
        x_odd = x[..., ::2]
        x_even = x[..., 1::2]

        _x = torch.empty_like(x, device=x.device)
        _x[..., 0::2] = -x_even
        _x[..., 1::2] = x_odd

        # x_stacked = torch.stack([-x_even, x_odd], dim=-1)
        # _x = x_stacked.flatten(start_dim=-2)
        _x = _x * sin[:x.size(-2), :]
        x = x * cos[:x.size(-2), :]
        return x + _x

class BERTEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self._segment_embeddings = nn.Embedding(
            num_embeddings=3,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.LongTensor, segmet_label: torch.LongTensor) -> torch.Tensor:
        x = self._embeddings(x) + self._segment_embeddings(segmet_label)
        return self.dropout(x)
class RoPEMultiHeadedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        head_embedding_size: int,
        positional_embedding: RotaryPositionEmbedding,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._embedding_size = embedding_size
        self._head_embedding_size = head_embedding_size
        self._positional_embedding = positional_embedding
        self._Q = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._K = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._V = nn.Linear(self._embedding_size, self._num_heads * self._head_embedding_size)
        self._W_proj = nn.Linear(self._num_heads * self._head_embedding_size, self._embedding_size)
        self._dropout = nn.Dropout(p=dropout)
        self._layernorm = nn.LayerNorm(self._embedding_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self._Q.forward(query).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        k = self._K.forward(key).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)
        v = self._V.forward(value).view(batch_size, -1, self._num_heads, self._head_embedding_size).transpose(1, 2)

        q_rope = self._positional_embedding.forward(q)
        k_rope = self._positional_embedding.forward(k)

        attention_numerator = torch.exp(
            torch.matmul(q_rope, k_rope.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        )
        attention_denominator = torch.exp(
            torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        )
        attention_denominator = torch.sum(attention_denominator, dim=-1, keepdim=True)
        a = attention_numerator / attention_denominator
        # a = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / torch.sqrt(torch.tensor(self._head_embedding_size))
        if mask is not None:
            # mask = mask.unsqueeze(1)
            a = a.masked_fill(mask == 0, -torch.inf)
        
        alpha = F.softmax(a, -1)

        z = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(batch_size, -1, self._num_heads * self._head_embedding_size)
        z = self._W_proj(z)
        return self._layernorm(query + self._dropout(z))


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
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_embedding_size: int,
        fcnn_hidden_size: int,
        positional_embedding: RotaryPositionEmbedding,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._mha = RoPEMultiHeadedAttention(
            embedding_size=embedding_size,
            num_heads=num_heads,
            head_embedding_size=head_embedding_size,
            positional_embedding=positional_embedding,
            dropout=dropout,
        )
        self._fcnn = FCNNBlock(
            embedding_size=embedding_size,
            hidden_size=fcnn_hidden_size,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._fcnn(self._mha(x, x, x, mask))
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        embedding_size: int,
        num_heads: int,
        head_embedding_size: int,
        fcnn_hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._embeddings = BERTEmbedding(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )
        self._positional_embeddings = RotaryPositionEmbedding(
            embedding_size=head_embedding_size,
            base=1000,
        )
        self._layers = nn.ModuleList(
            EncoderLayer(
                embedding_size=embedding_size,
                num_heads=num_heads,
                head_embedding_size=head_embedding_size,
                fcnn_hidden_size=fcnn_hidden_size,
                positional_embedding=self._positional_embeddings,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(
        self,
        x: torch.LongTensor,
        segment: torch.LongTensor,
    ) -> torch.Tensor:
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        z = self._embeddings(x, segment)
        for layer in self._layers:
            z = layer(z, mask)
        return z
class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, embedding_size: int):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self._linear = nn.Linear(embedding_size, 2)
        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self._softmax(self._linear(x[:, 0]))

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, embedding_size: int, vocab_size: int):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self._linear = nn.Linear(embedding_size, vocab_size)
        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self._softmax(self._linear(x))

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, encoder: Encoder):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self._encoder = encoder
        self._next_sentence = NextSentencePrediction(self._encoder.embedding_size)
        self._mask_lm = MaskedLanguageModel(self._encoder.embedding_size, self._encoder.vocab_size)

    def forward(
        self,
        x: torch.LongTensor,
        segment_label: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._encoder(x, segment_label)
        return self._next_sentence(x), self._mask_lm(x)
import re
from typing import List, Mapping, Self

class ChordTokenizer:
    def __init__(self):
        self._padding_token = "[PAD]"
        self._unknown_token = "[UNK]"
        self._cls_token = "[CLS]"
        self._sep_token = "[SEP]"
        self._mask_token = "[MASK]"
        
        # Special tokens IDs
        self._padding_id = 0
        self._cls_id = 1
        self._sep_id = 2
        self._mask_token_id = 3
        self._unknown_token_id = 4
        
        # Музыкальные элементы
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.moods = ['m', 'maj', 'min', 'aug', 'dim', 'sus2', 'sus4', 'sus']
        self.extensions = [
            '5', '6', '7', '9', '11', '13', 
            'add9', 'add11', 'add13'
        ]
        self.symbols = ['/', 'b', '#', '(', ')', ' ']
        
        # Сложные аккорды для добавления в словарь
        self.complex_chords = [
            'A5(9)', 'Cadd9', 'Dsus4', 'Emadd9', 'G5(11)',
            'Fmaj7', 'G9', 'Am11', 'C7(9)', 'Dsus2',
            'Cmaj9', 'F#m7', 'Bbmaj7', 'E7sus4', 'Aadd9'
        ]
        
        self._init_vocab()

    @property
    def vocab(self) -> Mapping[int, str]:
        return self._vocab
    
    @property
    def reverse_vocab(self) -> Mapping[str, int]:
        return {token: idx for idx, token in self._vocab.items()}
    
    @property
    def cls_id(self) -> int:
        return self._cls_id
    
    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id
    
    @property
    def padding_id(self) -> int:
        return self._padding_id
    
    @property
    def sep_id(self) -> int:
        return self._sep_id
    
    @property
    def unknown_token_id(self) -> int:
        return self._unknown_token_id

    def _init_vocab(self) -> None:
        """Инициализация словаря с специальными токенами"""
        self._vocab = {
            self._padding_id: self._padding_token,
            self._cls_id: self._cls_token,
            self._sep_id: self._sep_token,
            self._mask_token_id: self._mask_token,
            self._unknown_token_id: self._unknown_token,
        }
    
    def fit(self, corpus: List[str]) -> Self:
        """Создание словаря на основе корпуса"""
        self._init_vocab()
        
        # Добавляем базовые музыкальные элементы
        all_elements = (self.notes + self.moods + self.extensions + 
                       self.symbols + self.complex_chords)
        
        for element in all_elements:
            if element not in self._vocab.values():
                self._vocab[len(self._vocab)] = element
        
        # Обрабатываем корпус для извлечения дополнительных аккордов
        for text in corpus:
            chords = text.split()
            for chord in chords:
                if chord not in self.reverse_vocab and chord not in self._vocab.values():
                    self._vocab[len(self._vocab)] = chord
        
        return self
    
    def tokenize_text(self, text: str | List[str]) -> List[str] | List[List[str]]:
        """Токенизация текста в строковые токены"""
        if isinstance(text, str):
            return self._tokenize_text(text)
        assert isinstance(text, list), "`text` should be str or List[str]"
        return [self._tokenize_text(chunk) for chunk in text]
 
    def tokenize_ids(self, text: str | List[str]) -> List[int] | List[List[int]]:
        """Токенизация текста в ID токенов"""
        if isinstance(text, str):
            return self._tokenize_ids(text)
        assert isinstance(text, list), "`text` should be str or List[str]"
        return [self._tokenize_ids(chunk) for chunk in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование ID токенов обратно в строку"""
        content = []
        reverse_vocab = self.reverse_vocab
        
        for token_id in tokens:
            if token_id in [self._padding_id, self._cls_id, self._sep_id, self._mask_token_id]:
                continue
            
            token = self._vocab.get(token_id, self._unknown_token)
            if token == self._unknown_token:
                continue
                
            content.append(token)
        
        # Собираем аккорды из токенов
        result = []
        current_chord = []
        
        for token in content:
            if token == ' ':
                if current_chord:
                    result.append(''.join(current_chord))
                    current_chord = []
            else:
                current_chord.append(token)
        
        if current_chord:
            result.append(''.join(current_chord))
            
        return ' '.join(result)

    def _tokenize_text(self, text: str) -> List[str]:
        """Внутренний метод для токенизации строки в текстовые токены"""
        tokens = [self._cls_token]
        reverse_vocab = self.reverse_vocab
        
        chords = text.split()
        
        for i, chord in enumerate(chords):
            # Пытаемся найти целый аккорд в словаре
            if chord in reverse_vocab:
                tokens.append(chord)
            else:
                # Разбиваем аккорд на составляющие
                chord_parts = self._split_chord(chord)
                for part in chord_parts:
                    if part in reverse_vocab:
                        tokens.append(part)
                    else:
                        tokens.append(self._unknown_token)
            
            # Добавляем пробел между аккордами (кроме последнего)
            if i < len(chords) - 1:
                tokens.append(' ')
        
        tokens.append(self._sep_token)
        return tokens
    
    def _tokenize_ids(self, text: str) -> List[int]:
        """Внутренний метод для токенизации строки в ID токенов"""
        text_tokens = self._tokenize_text(text)
        reverse_vocab = self.reverse_vocab
        return [reverse_vocab.get(token, self._unknown_token_id) for token in text_tokens]
    
    def _split_chord(self, chord: str) -> List[str]:
        """Разбивает аккорд на составляющие элементы"""
        # Регулярное выражение для разбора аккордов
        pattern = r'[A-G][#b]?|[a-z]+|\d+|[\/\(\)#b]'
        parts = re.findall(pattern, chord)
        return parts
    
    def __len__(self) -> int:
        return len(self._vocab)


# Адаптер для Hugging Face (обновленная версия)
class ChordTokenizerHF:
    def __init__(self, chord_tokenizer: ChordTokenizer):
        self.chord_tokenizer = chord_tokenizer

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_masks = []

        for text in texts:
            token_ids = self.chord_tokenizer.tokenize_ids(text)
            
            # Обрезаем если нужно
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            attention_mask = [1] * len(token_ids)

            # Добавляем паддинг если нужно
            if padding:
                padding_length = max_length - len(token_ids)
                token_ids = token_ids + [self.chord_tokenizer.padding_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)

        output = {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

        if return_tensors == 'pt':
            import torch
            output['input_ids'] = torch.tensor(output['input_ids'])
            output['attention_mask'] = torch.tensor(output['attention_mask'])

        return output

    def decode(self, token_ids: List[int]) -> str:
        """Декодирование ID токенов обратно в строку"""
        return self.chord_tokenizer.decode(token_ids)