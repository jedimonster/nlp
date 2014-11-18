from nltk import TaggerI, SequentialBackoffTagger

__author__ = 'michael'


class EntropyTaggerI(TaggerI):

    def possible_tags(self, word):
        return {'VERB': 0.4, 'ADJ': 0.6}

    def entropy(self, w):
        return 0.5


class tagger_1(EntropyTaggerI):

    def choose_tag(self, tokens, index, history):
        token = tokens[index]
        return 'ADJ'

    def entropy(self, w):
        return 0.3


class tagger_2(EntropyTaggerI):

    def choose_tag(self, tokens, index, history):
        token = tokens[index]
        return 'VERB'

    def entropy(self, w):
        return 0.4


class EntropyVotingTagger(SequentialBackoffTagger):

    def __init__(self, taggers=[], max_entropy=0.6):
        self._taggers = taggers
        self.max_entropy = max_entropy

    def choose_tag(self, tokens, index, history):
        print '!!!!!!!'
        token = tokens[index]

        best_tagger = min(self._taggers, key=lambda tagger: tagger.entropy(token))
        import pdb; pdb.set_trace()
        if best_tagger is None or best_tagger.entropy() > self.max_entropy:
            return None

        return best_tagger.choose_tag(token, index, history)

if __name__ == '__main__':
    tagger1 = tagger_1()
    tagger2 = tagger_2()

    # voting_tagger = EntropyVotingTagger([tagger1, tagger2])
    # result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    # print result

    voting_tagger = EntropyVotingTagger([tagger1, tagger2], max_entropy=0.1)
    result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    print result
