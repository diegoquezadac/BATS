import nltk
from typing import List, Any
from collections import Counter
from nltk.tokenize import word_tokenize

nltk.download('punkt')
MAX_SEQUENCE_LENGTH = 15

class SimpleVocab:
    def __init__(self, dataset: Any, max_size: int = 10000):
        self.max_size = max_size
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }
        self._build_vocab(dataset)

    def _build_vocab(self, dataset: Any):
        # Tokenize all the sentences
        tokenized_sentences = [word_tokenize(instance['sentence'].lower()) for instance in dataset]
        all_words = [word for sentence in tokenized_sentences for word in sentence]

        # Count all the words
        word_freq = Counter(all_words)

        # Take the most common words up to `max_size - num_special_tokens`
        most_common_words = word_freq.most_common(self.max_size - len(self.vocab))

        # Add these words to the vocabulary
        for index, (word, _) in enumerate(most_common_words, start=len(self.vocab)):
            self.vocab[word] = index

    def _get_token_id(self, token: str):
        return self.vocab.get(token, self.vocab['<unk>'])

    def encode(self, sentence: str):
        tokens = word_tokenize(sentence.lower())
        token_ids = [self._get_token_id(token) for token in tokens]
        
        # Truncate the sentence if it's too long
        if len(token_ids) > MAX_SEQUENCE_LENGTH:
            token_ids = token_ids[:MAX_SEQUENCE_LENGTH]
        
        # Pad the sentence if it's too short
        elif len(token_ids) < MAX_SEQUENCE_LENGTH:
            token_ids += [self.vocab['<pad>']] * (MAX_SEQUENCE_LENGTH - len(token_ids))
        
        return token_ids

    def decode(self, token_ids: List[int]):
        tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(token_id)] for token_id in token_ids]
        return ' '.join(tokens)