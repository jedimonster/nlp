import math
from nltk import TaggerI, SequentialBackoffTagger, ConditionalFreqDist, corpus

__author__ = 'michael'


class EntropyTaggerI(SequentialBackoffTagger):

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

    def tag(self, tokens):
        return [self.tag_word(word) for word in tokens]

    def tag_word(self, word):
        if word not in self.freq:
            return word, None
        m = max(self.freq[word].items(), key=lambda x: x[1])
        return word, m[0]


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
    tagger1 = tagger_1()
    tagger2 = tagger_2(backoff=tagger1)

    voting_tagger = EntropyVotingTagger(tagger2)
    result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    print result

    voting_tagger = EntropyVotingTagger(tagger2, max_entropy=0.1)
    result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    print result

    #Testing Entropy Taggers
    print '--------------Testing Entropy Taggers-------------'
    all_words = corpus.brown.tagged_sents(tagset='universal', categories='news')
    # random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!
    ds_length = len(all_words)
    train = all_words[:int(0.5 * ds_length)]

    tagger = EntropyUnigramTagger(train=train)

    res = tagger.possible_tags('open')
    print 'Result for tagging the word "open":'
    print 'Sum oF Dist (should be allways 1): ', sum(res.values())

    print 'Entropy:', tagger.entropy('open')

