import math
import random
from nltk import TaggerI, SequentialBackoffTagger, ConditionalFreqDist, corpus, ContextTagger, AffixTagger
from nltk.compat import xrange, izip

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
        :return: either the entropy or float("inf") if there are no possible tags
        """
        ent = 0
        dist = self.possible_tags(word)
        if len(dist) == 0:
            return float("inf")

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

    # def entropy(self, word):
    # affix = self._get_word_affix(word)
    # if affix is None:
    # return None
    #
    # return super(EntropyAffixTagger, self).entropy(affix)

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
    def __init__(self, taggers, max_entropy=0.6, backoff=None):
        """

        :param taggers: list of taggers to use
        :param max_entropy:
        """
        self.backoff = backoff
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
            if self.backoff is not None:  # use backoff if we're not sure
                return self.backoff.tag([token])[0][1]
            return None

        return best_tagger.choose_tag(token)

    def evaluate(self, gold):
        """
        Score the accuracy of the tagger against the gold standard.
        Strip the tags from the gold standard text, retag it using
        the tagger, then compute the accuracy score.

        :type gold: list(list(tuple(str, str)))
        :param gold: The list of tagged sentences to score the tagger on.
        :rtype: float
        """
        from nltk.tag import untag

        tagged_sents = self.tag_sents(untag(sent) for sent in gold)
        gold_tokens = sum(gold, [])
        test_tokens = sum(tagged_sents, [])
        return self.accuracy(gold_tokens, test_tokens)

    def accuracy(self, reference, test):
        """
        Given a list of reference values and a corresponding list of test
        values, return the fraction of corresponding values that are
        equal.  In particular, return the fraction of indices
        ``0<i<=len(test)`` such that ``test[i] == reference[i]``.

        :type reference: list
        :param reference: An ordered list of reference values.
        :type test: list
        :param test: A list of values to compare against the corresponding
            reference values.
        :raise ValueError: If ``reference`` and ``length`` do not have the
            same length.
        """
        # print "here!"
        if len(reference) != len(test):
            raise ValueError("Lists must have the same length.")
        return float(sum((x == y) for x, y in izip(reference, test))) / len(test)


def plot_data():
    import pylab

    data_nones_are_errors = [(0.0, 0.5770996936018715), (0.1, 0.27396246032183613), (0.2, 0.22664797296765093),
                             (0.3, 0.21289550169739835), (0.4, 0.19455099677848753), (0.5, 0.17860884222465523),
                             (0.6, 0.16773131483392278), (0.7, 0.1534274844634882), (0.8, 0.1427941303885506),
                             (0.9, 0.13102655187895307), (1.0, 0.10392331382572328), (1.1, 0.1004024921431328),
                             (1.2, 0.09960695972715605), (1.3, 0.09878779763545709), (1.4, 0.09232114304618022)]
    data_nones_are_correct = [(0.0, 0.0025362518608369466), (0.1, 0.002748918942335754), (0.2, 0.0036625997369229557),
                              (0.3, 0.004261218188549076), (0.4, 0.005576603470411667), (0.5, 0.0073330760324199495),
                              (0.6, 0.009522759315999663), (0.7, 0.012649753069888692), (0.8, 0.015327782985058214),
                              (0.9, 0.018809221874778514), (1.0, 0.037626320308130956), (1.1, 0.04008380658322763),
                              (1.2, 0.04048551107050302), (1.3, 0.04109200608070318), (1.4, 0.04482549484479237)]
    data_for_graph = [[a for a, b in data_nones_are_correct], [b for a, b in data_nones_are_correct]]

    pylab.title("Nones considered as correct")
    pylab.ylabel('Error')
    pylab.xlabel('Entropy')
    pylab.plot(data_for_graph[0], data_for_graph[1], '-bo')
    pylab.show()


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
    # plot_data()
    # import sys
    # sys.exit(1)

    all_words = corpus.brown.tagged_sents(tagset='universal')
    # random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    from nltk import UnigramTagger, AffixTagger

    unigram = UnigramTagger(train)
    affix_ugram_backoff = AffixTagger(train, backoff=unigram)
    affix = AffixTagger(train)
    unigram_affix_backoff = UnigramTagger(train, backoff=affix)
    # print "testing"
    # print affix_ugram_backoff.evaluate(test)
    # print unigram_affix_backoff.evaluate(test)
    # cutoffs = [x*0.1 for x in range(20)]
    # for c in cutoffs:
    # tagger = EntropyVotingTagger(taggers, c)
    # print "Accuracy of entropy voting = ", tagger.evaluate(test)


    affix_tagger = EntropyAffixTagger(train)
    unigram_tagger = EntropyUnigramTagger(train)
    taggers = [unigram_tagger, affix_tagger]
    tagger = EntropyVotingTagger(taggers, max_entropy=80)

    from nltk.tag import untag

    untagged_test = [untag(x) for x in dev]
    tagged_sents_uni_affix = unigram_affix_backoff.tag_sents(untagged_test)
    tagged_sents_entr = tagger.tag_sents(untagged_test)
    affix_mistake = 0
    unigram_mistake = 0
    overall_mistakes = 0
    print "len of dev: ", len(dev)
    for tagged_reference_sent, tagged_uni_affix_sent, tagged_entropy_sent in izip(dev, tagged_sents_uni_affix,
                                                                                  tagged_sents_entr):
        # import pdb;pdb.set_trace()
        for tagged_reference, tagged_uni_affix, tagged_entropy in izip(tagged_reference_sent, tagged_uni_affix_sent,
                                                                       tagged_entropy_sent):
            if tagged_uni_affix[1] != tagged_entropy[1]:
                overall_mistakes += 1

                print "WE GOT MATCH!"
                print "Word = ", tagged_reference[0]
                print "real tag ", tagged_reference[1]
                print "backoff tag ", tagged_uni_affix[1]
                print "entropy tag ", tagged_entropy[1]
                for t in tagger._taggers:
                    # import pdb
                    # pdb.set_trace()
                    print "Entropy for tagger ", t.__class__.__name__, " ", t.entropy(tagged_reference[0])
                print "******"
                if tagged_reference[1] !=tagged_entropy[1] and tagger._taggers[0].entropy(tagged_reference[0]) > tagger._taggers[1].entropy(tagged_reference[0]):
                    affix_mistake+=1
                if tagged_reference[1] !=tagged_entropy[1] and tagger._taggers[0].entropy(tagged_reference[0]) < tagger._taggers[1].entropy(tagged_reference[0]):
                    unigram_mistake +=1
                # from nltk import UnigramTagger
                # u1 = UnigramTagger(train)
                # print u1.evaluate(test)
                # import sys
                # sys.exit(1)
                #

                #
                # error_rates = []
                # for i in range(15):
                # i = float(i) / 10.0
                # error_rate = 1 - tagger.evaluate(test)
                # print "error rate for %f= " % (i,), error_rate
                # error_rates += [(i, error_rate)]
                # from nltk import UnigramTagger, AffixTagger, NgramTagger
                # #
                # # print "error rates (None=error) = ", error_rates
                # ngram = NgramTagger(2, train)
                # # plot_data()
                # voting_tagger = EntropyVotingTagger(taggers=taggers, max_entropy=20)
                #
                # u1 = UnigramTagger(train)
                # a1 = AffixTagger(train, backoff=u1)
                # a2 = AffixTagger(train)
                # u2 = UnigramTagger(train, backoff=a2)
                # print "testing"
                # print a1.evaluate(test)
                # print u2.evaluate(test)
                # from nltk.tag import untag
                #
                # print "voting eval = ", voting_tagger.evaluate(test)
                #
                # print " now choosing best cutoff on dev and after test on test"
                # #
                # cutoffs = [0.1*x for x in range(30)]
                # for cutoff in cutoffs:
                # print "testing cutoff: ", cutoff
                #     voting_tagger0 = EntropyVotingTagger(taggers=taggers, max_entropy=cutoff, backoff=None)
                #     voting_tagger = EntropyVotingTagger(taggers=taggers, max_entropy=cutoff, backoff=ngram)
                #     voting_tagger2 = EntropyVotingTagger(taggers=taggers, max_entropy=cutoff, backoff=a2)
                #     voting_tagger3 = EntropyVotingTagger(taggers=taggers, max_entropy=cutoff, backoff=u1)
                #     print voting_tagger0.evaluate(dev)
                #     print voting_tagger.evaluate(dev)
                #     print voting_tagger2.evaluate(dev)
                #     print voting_tagger3.evaluate(dev)
                #

                # untagged_dev = [untag(x) for x in dev]
                # tagged_by_voting = voting_tagger.tag_sents(untagged_dev)
                # tagged_by_ua = a1.tag_sents(untagged_dev)
                # count_none_disagreements = 0
                # count_error_disagreemnts = 0
                # for sent_voting, sent_ua in izip(tagged_by_voting, tagged_by_ua):
                #     for token_by_voting, token_by_ua in izip(sent_voting, sent_ua):
                #         # import pdb
                #         # pdb.set_trace()
                #         if token_by_voting != token_by_ua:
                #             if token_by_voting[1] is None:
                #                 count_none_disagreements += 1
                #                 continue
                #             count_error_disagreemnts += 1
                #             print "******"
                #             print "voting tagger answer: ", token_by_voting
                #             print "ua tagger answer: ", token_by_ua
                #             print "******"
                # print "count None disagreements ", count_none_disagreements
                # print "count error disagreemnts ", count_error_disagreemnts
    print "overall disagreements:", overall_mistakes
    print "affix mistakes: ",affix_mistake
    print "uni mistakes: ", unigram_mistake