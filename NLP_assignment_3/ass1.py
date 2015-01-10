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
            # print p

            if p == 0:
                continue
            else:
                ent += p * math.log(p)
        # print -ent
        # print "************"
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
                # print affix
            else:
                # print "NOT ADDING!!! ", self.entropy(dist), "  ", cutoff, "  ", self._calc_distribution_of_suffix(dist)
                pass


def evaluate_tag(reference, test, tag):
    """
    :param reference: list of tagged tokens
    :param test: list of tagged tokens
    :return:
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for ref_token, test_token in zip(reference, test):
        if ref_token[1] == tag and test_token[1] == tag:
            TP += 1
        if ref_token[1] != tag and test_token[1] != tag:
            TN += 1
        if ref_token[1] == tag and test_token[1] != tag:
            FP += 1
        if ref_token[1] != tag and test_token[1] == tag:
            FN += 1
    try:
        precision = float(TP) / float(TP + FP)

        recall = float(TP) / float(TP + FN)
        f_measure = 2 * precision * recall / (recall + precision)
    except:
        print "tag %r is very uncommong TP %r TN %r FP %r FN %r" % (tag, TP, TN, FP, FN)
        return

    print "results for tag %r are: TP: %r TN: %r FP: %r FN: %r precision: %r recall %r f_measure %r" % (
        tag, TP, TN, FP, FN, precision, recall, f_measure)
    return f_measure


def get_all_tags(corpus_test):
    return set([y[1] for x in corpus_test for y in x])


def get_statistics_per_tag(corpus_test, tagger):
    from nltk.tag import untag

    possible_tags = get_all_tags(corpus_test)
    untaged_test = [untag(x) for x in corpus_test]
    tagged_sents = tagger.tag_sents(untaged_test)
    ref_words = sum(corpus_test, [])
    test_words = sum(tagged_sents, [])
    best_fmeasure = 0
    best_tag = "Chtulhu"
    worst_tag_fmeasure = 100
    worst_tag = "cthulhu"
    for tag in possible_tags:
        f_measure = evaluate_tag(ref_words, test_words, tag)
        if f_measure > best_fmeasure:
            best_tag = tag
            best_fmeasure = f_measure
        if f_measure < worst_tag_fmeasure:
            worst_tag = tag
            worst_tag_fmeasure = f_measure
    print "best tag: ", best_tag
    print "worst tag: ", worst_tag


if __name__ == "__main__":
    # split the brown corpus to test, dev, and test set
    all_words = corpus.brown.tagged_sents(tagset="universal")
    print get_all_tags(all_words)
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    from nltk import UnigramTagger

    u_tagger = UnigramTagger(train)
    get_statistics_per_tag(dev, u_tagger)
    import sys

    sys.exit(1)

    start = datetime.datetime.now()

    our_tagger = SimpleUnigramTagger(train=train)
    eat = EntropyAffixTagger(train=train, cutoff=0.5)

    print "building our tagger took ", datetime.datetime.now() - start

    # start = datetime.datetime.now()

# /    their_tagger = NgramTagger(1, train=train)
# print "building nltk tagger took ", datetime.datetime.now() - start

# print our_tagger.evaluate(test)
# print their_tagger.evaluate(test)
