import torch

class ChordTokenizer:
    def __init__(self, chords_set, vocab_file=None):


        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        for idx, element in enumerate(chords_set):
            self.vocab[element] = len(self.vocab)


        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        chords = text.split(' ')

        tokens = []

        for chord in chords:
            tokens.append(chord)

        if tokens and tokens[-1] == ' ':
            tokens.pop()

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.reverse_vocab.get(id, '[UNK]') for id in ids]

    def __len__(self):
        return len(self.vocab)

class ChordTokenizerHF:
    def __init__(self, chord_tokenizer):
        self.chord_tokenizer = chord_tokenizer

    def __call__(self, texts, padding=True, truncation=True, max_length=50, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_masks = []

        for text in texts:
            tokens = self.chord_tokenizer.tokenize(text)
            token_ids = self.chord_tokenizer.convert_tokens_to_ids(tokens)

            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            attention_mask = [1] * len(token_ids)

            if padding:
                padding_length = max_length - len(token_ids)
                token_ids = token_ids + [self.chord_tokenizer.vocab['[PAD]']] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)

        output = {
            'input_ids': torch.tensor(input_ids) if return_tensors == 'pt' else input_ids,
            'attention_mask': torch.tensor(attention_masks) if return_tensors == 'pt' else attention_masks
        }

        return output