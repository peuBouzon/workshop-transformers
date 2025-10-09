import re
from collections import Counter

class MostFrequentWordsTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}

    def clean_sentence(self, sentence):
        sentence = sentence.lower()
        return re.sub(r'[^a-z0-9\s]', '', sentence)

    def build_vocab(self, sentences):
        words = []
        for sentence in sentences:
            words.extend(self.clean_sentence(str(sentence)).split())

        word_counts = Counter(words)
        most_common_words = word_counts.most_common(self.vocab_size - 4)
        
        self.token_to_idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3} # Tokens especiais
        for i, (word, _) in enumerate(most_common_words, 4):
            self.token_to_idx[word] = i

        self.idx_to_token = {i: w for w, i in self.token_to_idx.items()}

    def tokenize(self, sentence):
        cleaned_sentence = self.clean_sentence(str(sentence))
        return cleaned_sentence.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_idx.get(token, self.token_to_idx['[UNK]']) for token in tokens]
    
    def encode(self, sentence):
        tokens = ['[CLS]'] + self.tokenize(sentence) + ['[SEP]']
        return self.convert_tokens_to_ids(tokens)
    
    def decode(self, ids):
        return ' '.join([self.idx_to_token.get(i, '[UNK]') for i in ids])

    def get_vocab_size(self):
        return len(self.token_to_idx)
