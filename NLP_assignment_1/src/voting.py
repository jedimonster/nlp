import math
import random
from nltk import TaggerI, SequentialBackoffTagger, ConditionalFreqDist, corpus, ContextTagger, AffixTagger

__author__ = 'michael'


class EntropyTaggerI(TaggerI):
    def possible_tags(self, word):
        """
        returns e.g. {'adj':0.6, 'verb':0.4}
        """
        raise Exception(NotImplemented)

    def entropy(self, word):
        """

        :param word:
        :return: either the entropy or None if there are no possible tags.
        """
        ent = 0
        dist = self.possible_tags(word)
        for key, p in dist.items():
            # print dist
            if p == 0:
                continue
            else:
                ent += p * math.log(p, 2)

        return -ent

    def choose_tag(self, w):
        """

        :param w:
        :return: either the chosen tag or None.
        """
        possible_tags = self.possible_tags(w)
        if len(possible_tags) == 0:
            return None

        tag, count = max(possible_tags.items(), key=lambda x: x[1])
        return tag


class EntropyUnigramTagger(EntropyTaggerI):
    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        # list comprehension here just flattens the sentences.
        combined_sents = (item for sublist in train for item in sublist)
        self.freq = ConditionalFreqDist(combined_sents)
        # print self.freq['open']
        # for item in self.freq['open']:
        # print item
        # print self.freq['open'][item]
        # for item in self.freq:
        # print "***"
        # for z in self.freq[item]:
        # print z
        # print "***"
        # print item

    def possible_tags(self, word):
        total_count = sum(self.freq[word].values())
        return dict((tag, float(count) / total_count) for (tag, count) in self.freq[word].iteritems())


        # def tag(self, tokens):
        # return [self.tag_word(word) for word in tokens]
        #
        # def tag_word(self, word):
        # if word not in self.freq:
        # return word, None
        # m = max(self.freq[word].items(), key=lambda x: x[1])
        # return word, m[0]
        #
        # def context(self, tokens, index, history):
        # return tokens[index]


class EntropyAffixTagger(EntropyTaggerI):
    def __init__(self, train, model=None,
                 backoff=None, cutoff=0, verbose=False, affix_length=-3, min_word_length=5):
        # list comprehension here just flattens the sentences.
        self.min_word_length = min_word_length
        self.affix_length = affix_length
        self.freq = ConditionalFreqDist()
        self._train(train)

    def entropy(self, word):
        affix = self._get_word_affix(word)
        if affix is None:
            return None

        return super(EntropyAffixTagger, self).entropy(affix)

    def _get_word_affix(self, token):
        if len(token) < self.min_word_length:
            return None
        elif self.affix_length > 0:
            return token[:self.affix_length]
        else:
            return token[self.affix_length:]

    def possible_tags(self, word):
        suffix = self._get_word_affix(word)
        if suffix is None or suffix not in self.freq:
            return dict()

        return self._calc_distribution_of_suffix(self.freq[suffix])

    def _calc_distribution_of_suffix(self, tag_count_dict):
        overall = 0
        for item in tag_count_dict:
            overall += tag_count_dict[item]
        dist = {}
        for item in tag_count_dict:
            dist[item] = float(float(tag_count_dict[item]) / overall)
        # print dist
        return dist

    def _train(self, tagged_sents, cutoff=0, verbose=False):
        token_count = hit_count = 0
        # cutoff = 0.99
        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.

        for sentence in tagged_sents:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                token_count += 1
                affix = self._get_word_affix(token)
                self.freq[affix][tag] += 1

                # def tag(self, tokens):
                # return [self.tag_word(word) for word in tokens]
                #
                # def tag_word(self, word):
                # if word not in self.freq:
                # return word, None
                # m = max(self.freq[word].items(), key=lambda x: x[1])
                # return word, m[0]


class EntropyVotingTagger(TaggerI):
    def __init__(self, taggers, max_entropy=0.6):
        """

        :param taggers: list of taggers to use
        :param max_entropy:
        """
        self._taggers = taggers
        self.max_entropy = max_entropy

    def tag(self, tokens):
        return [(token, self._tag_one(token)) for token in tokens]

    def _tag_one(self, token):
        """

        :param token:
        :return:
        """
        good_taggers = [x for x in self._taggers if x.entropy(token) is not None]
        best_tagger = min(good_taggers, key=lambda t: t.entropy(token))
        if best_tagger is None or best_tagger.entropy(token) > self.max_entropy:
            return None

        return best_tagger.choose_tag(token)


if __name__ == '__main__':
    # Testing Voting Mechanizem
    # tagger1 = tagger_1()
    # tagger2 = tagger_2(backoff=tagger1)
    #
    # voting_tagger = EntropyVotingTagger(tagger2)
    # result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    # print result

    #
    # voting_tagger = EntropyVotingTagger(tagger2, max_entropy=0.1)
    # result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    # print result
    #
    # #Testing Entropy Taggers
    # print '--------------Testing Entropy Taggers-------------'
    all_words = corpus.brown.tagged_sents(tagset='universal')
    # random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    affix_tagger = EntropyAffixTagger(train)
    unigram_tagger = EntropyUnigramTagger(train)
    taggers = [affix_tagger, unigram_tagger]
    tagger = EntropyVotingTagger(taggers, max_entropy=100)
    print "Accuracy of entropy voting = ", tagger.evaluate(test)

    from nltk import UnigramTagger, AffixTagger

    u1 = UnigramTagger(train)
    a1 = AffixTagger(train, backoff=u1)
    a2 = AffixTagger(train)
    u2 = UnigramTagger(train, backoff=a2)
    print "testing"
    print a1.evaluate(test)
    print u2.evaluate(test)

