from collections import defaultdict
import random
from nltk import TaggerI, ConditionalFreqDist, corpus, NgramTagger

__author__ = 'itay'


class SimpleUnigramTagger(TaggerI):
    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        # list comprehension here just flattens the sentences.
        combined_sents = (item for sublist in train for item in sublist)
        self.freq = ConditionalFreqDist(combined_sents)

    def tag(self, tokens):
        return [self.tag_word(word) for word in tokens]

    def tag_word(self, word):
        if word not in self.freq:
            return word, None
        m = max(self.freq[word].items(), key=lambda x: x[1])
        return word, m[0]


if __name__ == "__main__":
    # split the brown corpus to test, dev, and test set
    all_words = corpus.brown.tagged_sents(tagset='universal', categories='news')
    # random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!
    ds_length = len(all_words)
    train = all_words[:int(0.1 * ds_length)]
    dev = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]
    test = all_words[int(0.2) * ds_length:]

    our_tagger = SimpleUnigramTagger(train=train)
    # import pdb;pdb.set_trace()

    print our_tagger.evaluate(test)

    their_tagger = NgramTagger(1, train=train)
    print their_tagger.evaluate(test)
