from collections import defaultdict
import random
from nltk import TaggerI, ConditionalFreqDist, corpus, NgramTagger

__author__ = 'itay'


class SimpleUnigramTagger(TaggerI):
    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        self.freq = ConditionalFreqDist(train)

    def tag(self, tokens):
        [self.tag_word(word) for word in tokens]

    def tag_word(self, word):
        if word not in self.freq:
            return None
        m = max(self.freq[word].items(), key=lambda x: x[1])
        return m[0]


# split the brown corpus to test, dev, and test set
all_words = corpus.brown.tagged_words(tagset='universal')
# random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!

ds_length = len(all_words)
train = all_words[:int(0.1 * ds_length)]
dev = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]
test = all_words[int(0.2) * ds_length:]

our_tagger = SimpleUnigramTagger(train=train)
print our_tagger.evaluate(test)

their_tagger = NgramTagger(1, train=train)
print their_tagger.evaluate(test)