from nltk.corpus import brown
from nltk import corpus, UnigramTagger, AffixTagger, TaggerI, ConditionalFreqDist
from nltk.tag import untag
import collections
import math


class SimpleUnigramTagger(TaggerI):
    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        # list comprehension here just flattens the sentences.
        combined_sents = (item for sublist in train for item in sublist)
        self.freq = ConditionalFreqDist(combined_sents)
        self.words_per_tag = collections.defaultdict(lambda: 1)
        self.bad_entropy_count = 0
        self.overall_tokens_tagged = 0
        self.overall_distinct = []
        self.distinct_nones = []
        self.distinct_bad_entropy = []
        self.distinct_words_per_tag = {}
        self.overall_nones = 0

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
                ent += p * math.log(p, 2)
        # print -ent
        # print "************"
        return -ent

    def tag(self, tokens):
        return [self.tag_word(word) for word in tokens]

    def tag_word(self, word):
        if word not in self.overall_distinct:
            self.overall_distinct.append(word)
        self.overall_tokens_tagged += 1
        if word not in self.freq:
            self.overall_nones += 1
            if word not in self.distinct_nones:
                self.distinct_nones.append(word)
            return word, None
        if self.entropy(self.freq[word]) > 1:
            if word not in self.distinct_bad_entropy:
                self.distinct_bad_entropy.append(word)
            self.bad_entropy_count += 1
        options_num = len(self.freq[word])
        self.distinct_words_per_tag[word] = options_num
        # if options_num == 6:
        # print word
        # print "******"
        # for item in self.freq[word]:
        # print item
        # print "******"
        self.words_per_tag[options_num] += 1
        m = max(self.freq[word].items(), key=lambda x: x[1])
        return word, m[0]

    def get_number_of_options_per_word(self):
        a = self.words_per_tag
        distinct_dict = collections.defaultdict(lambda: 0)
        number_of_tags = []
        for item in self.distinct_words_per_tag.values():
            if item not in number_of_tags:
                number_of_tags.append(item)
        # print "* ", number_of_tags
        for x in number_of_tags:
            amount = self.distinct_words_per_tag.values().count(x)
            distinct_dict[x] = amount
        return a, distinct_dict

    def get_bad_entropy(self):
        print "overall bad entropy words: ", self.bad_entropy_count
        print "overall unique words with bad entropy: ", len(self.distinct_bad_entropy)
        print "precentage of bad entropy tokes to overall tokens: ", float(self.bad_entropy_count) / float(
            self.overall_tokens_tagged)
        print "precentage of bad entropy unique words to overall unique words: ", float(
            len(self.distinct_bad_entropy)) / float(self.get_overall_distinct())

    def get_overall_words(self):
        return self.overall_tokens_tagged

    def get_overall_distinct(self):
        return len(self.overall_distinct)

    def get_nones(self):
        print "overall nones : ", self.overall_nones
        print "unique words that are None: ", len(self.distinct_nones)
        print "precentage of none tokesns to overall tokes: ", float(self.overall_nones) / float(
            self.overall_tokens_tagged)
        print "precentage of unique nones to overall unique ", float(len(self.distinct_nones)) / float(
            self.get_overall_distinct())


if __name__ == "__main__":
    # split the brown corpus to test, dev, and test set
    all_words = corpus.brown.tagged_sents(tagset='universal')
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    untagged_dev = [untag(item) for item in dev]
    words_in_dev = 0
    for item in untagged_dev:
        words_in_dev += len(item)
    print "overall words in dev : ", words_in_dev
    u1 = SimpleUnigramTagger(train)
    tagged_dev = u1.tag_sents(untagged_dev)
    print len(tagged_dev)
    none_count = 0
    for sent in tagged_dev:
        for tagged_word in sent:
            if tagged_word[1] is None:
                none_count += 1

    print "Number of Nones in dev is: ", none_count
    print "number of options per (token, word) is:"
    options_per_token, options_per_unique_word = u1.get_number_of_options_per_word()
    print options_per_token, options_per_unique_word
    distinct_word_count = len(u1.overall_distinct)
    print "****"
    print dict(
        enumerate(map(lambda x: round((float(x) / float(u1.overall_tokens_tagged)), 2), options_per_token.itervalues()),
                  1))
    print dict(enumerate(
        map(lambda x: round((float(x) / float(distinct_word_count)), 2), options_per_unique_word.itervalues()), 1))
    print "****"
    u1.get_bad_entropy()
    u1.get_nones()
    print "really overall words: ", u1.get_overall_words()
    print " overall distinct words in dev: ", u1.get_overall_distinct()
    print "Note The problematic word is down( word that has 6 tags. the only unique word, but appears 69 times)"