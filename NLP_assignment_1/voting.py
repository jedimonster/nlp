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

    def __init__(self, taggers=[]):
        self._taggers = taggers

    def choose_tag(self, tokens, index, history):
        token = tokens[index]

        best_tagger = min(self._taggers, key=lambda tagger: tagger.entropy(token))
        if best_tagger is None:
            return None

        return best_tagger.choose_tag(token, index, history)

if __name__ == '__main__':
    tagger1 = tagger_1()
    tagger2 = tagger_2()

    voting_tagger = EntropyVotingTagger([tagger1, tagger2])

    result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    print result
