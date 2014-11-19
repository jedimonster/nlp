import math
import random
from nltk import TaggerI, SequentialBackoffTagger, ConditionalFreqDist, corpus, ContextTagger, AffixTagger

__author__ = 'michael'


class EntropyTaggerI(ContextTagger):

    def entropy(self, word):
        return self._calc_entropy(word)

    def possible_tags(self, word):
        """
        returns {'adj':0.6, 'verb':0.4}
        """
        total_count = sum(self.freq[word].values())
        return dict((tag, float(count)/total_count) for (tag, count) in self.freq[word].iteritems())

    #copy pasted from affix.py
    def _calc_entropy(self, word):
        ent = 0
        dist = self.possible_tags(word)
        for key, p in dist.items():
            #print dist
            if p == 0:
                continue
            else:
                ent += p * math.log(p)

        return -ent


class tagger_1(EntropyTaggerI):

    def choose_tag(self, tokens, index, history):
        token = tokens[index]
        return 'ADJ'

    def entropy(self, w):
        return 0.5


class tagger_2(EntropyTaggerI):

    def choose_tag(self, tokens, index, history):
        token = tokens[index]
        return 'VERB'

    def entropy(self, w):
        return 0.4


class EntropyUnigramTagger(EntropyTaggerI):

    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        # list comprehension here just flattens the sentences.
        combined_sents = (item for sublist in train for item in sublist)
        self.freq = ConditionalFreqDist(combined_sents)
        # print self.freq['open']
        # for item in self.freq['open']:
        #     print item
        #     print self.freq['open'][item]
        # for item in self.freq:
        #     print "***"
        #     for z in self.freq[item]:
        #         print z
        #     print "***"
        #     print item

    def tag(self, tokens):
        return [self.tag_word(word) for word in tokens]

    def tag_word(self, word):
        if word not in self.freq:
            return word, None
        m = max(self.freq[word].items(), key=lambda x: x[1])
        return word, m[0]

    def context(self, tokens, index, history):
        return tokens[index]


class EntropyAffixTagger(EntropyTaggerI, AffixTagger):

    def __init__(self, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        # list comprehension here just flattens the sentences.
        #combined_sents = (item for sublist in train for item in sublist)
        if train:
            self._train(train)

    def _train(self, tagged_corpus, cutoff=0, verbose=False):
        print "training"
        token_count = hit_count = 0
        #cutoff = 0.99
        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.
        self.freq = ConditionalFreqDist()
        for sentence in tagged_corpus:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                # Record the event.
                token_count += 1
                context = self.context(tokens, index, tags[:index])
                if context is None:
                    continue
                self.freq[context][tag] += 1
                # If the backoff got it wrong, this context is useful:
        for affix in self.freq:
            dist = self.freq[affix]
            self._context_to_tag[affix] = self.freq[affix].max()
        print "fgf"
        print self.freq['ly']

    def tag(self, tokens):
        return [self.tag_word(word) for word in tokens]

    def tag_word(self, word):
        if word not in self.freq:
            return word, None
        m = max(self.freq[word].items(), key=lambda x: x[1])
        return word, m[0]

    def context(self, tokens, index, history):
        token = tokens[index]
        if len(token) < self._min_word_length:
            return None
        elif self._affix_length > 0:
            return token[:self._affix_length]
        else:
            return token[self._affix_length:]


class EntropyVotingTagger(SequentialBackoffTagger):

    def __init__(self, taggers, max_entropy=0.6):
        super(EntropyVotingTagger, self).__init__(backoff=taggers)
        self.max_entropy = max_entropy

    def tag_one(self, tokens, index, history):
        token = tokens[index]

        #the slicing is because the first tagger is 'self'
        best_tagger = min(self._taggers[1:], key=lambda tagger: tagger.entropy(token))
        if best_tagger is None or best_tagger.entropy(token) > self.max_entropy:
            return None
        return best_tagger.choose_tag(tokens, index, history)

if __name__ == '__main__':

    #Testing Voting Mechanizem
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
    all_words = corpus.brown.tagged_sents(tagset='universal', categories='news')
    #random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!
    ds_length = len(all_words)
    train = all_words[:int(0.5 * ds_length)]

    tagger = EntropyUnigramTagger(train=train)
    # print "gfg"
    # affix = EntropyAffixTagger(train=train)
    # print "gfg"
    res = tagger.possible_tags('open')
    print 'Result for tagging the word "open":'
    print 'Sum oF Dist (should be allways 1): ', sum(res.values())

    print 'Entropy:', tagger.entropy('open')
    print tagger.tag(['open'])

