from collections import defaultdict
import random
import datetime
import math
from nltk import TaggerI, ConditionalFreqDist, corpus, NgramTagger, AffixTagger

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


class EntropyAffixTagger(AffixTagger):
    def _calc_distribution_of_suffix(self, tag_count_dict):
        overall = 0
        for item in tag_count_dict:
            overall += tag_count_dict[item]
        dist = {}
        for item in tag_count_dict:
            dist[item] = float(float(tag_count_dict[item]) / overall)
        # print dist
        return dist

    def entropy(self, dist):
        # print "************"
        ent = 0
        dist = self._calc_distribution_of_suffix(dist)
        for key, p in dist.items():

            # print key
            #print p

            if p == 0:
                continue
            else:
                ent += p * math.log(p)
        # print -ent
        #print "************"
        return -ent

    def _train(self, tagged_corpus, cutoff=1, verbose=False):
        token_count = hit_count = 0
        # cutoff = 0.99
        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.
        fd = ConditionalFreqDist()
        for sentence in tagged_corpus:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                # Record the event.
                token_count += 1
                context = self.context(tokens, index, tags[:index])
                if context is None:
                    continue
                fd[context][tag] += 1
                # If the backoff got it wrong, this context is useful:
        # print useful_contexts

        # filter out affixes that have entropy lower than cutoff
        for affix in fd:
            dist = fd[affix]
            if self.entropy(dist) <= cutoff:
                self._context_to_tag[affix] = fd[affix].max()
                print affix
            else:
                # print "NOT ADDING!!! ", self.entropy(dist), "  ", cutoff, "  ", self._calc_distribution_of_suffix(dist)
                pass


if __name__ == "__main__":
    # split the brown corpus to test, dev, and test set
    all_words = corpus.brown.tagged_sents(tagset='universal')
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    start = datetime.datetime.now()

    our_tagger = SimpleUnigramTagger(train=train)
    eat = EntropyAffixTagger(train=train, cutoff=0.5)

    print "building our tagger took ", datetime.datetime.now() - start

    # start = datetime.datetime.now()

# /    their_tagger = NgramTagger(1, train=train)
#     print "building nltk tagger took ", datetime.datetime.now() - start

    # print our_tagger.evaluate(test)
    # print their_tagger.evaluate(test)
