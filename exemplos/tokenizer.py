import re
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

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
        return self.convert_tokens_to_ids(self.tokenize(sentence))

    def get_vocab_size(self):
        return len(self.token_to_idx)

class BytePairEncodingTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"] # Tokens especiais

    def build_vocab(self, sentences):
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(sentences, trainer=trainer)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.token_to_id(token) for token in tokens]
    
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
