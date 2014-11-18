from nltk import TaggerI, SequentialBackoffTagger

__author__ = 'michael'


class EntropyTaggerI(SequentialBackoffTagger):

    def possible_tags(self, word):
        return {'VERB': 0.4, 'ADJ': 0.6}

    def entropy(self, w):
        return 0.5


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
    tagger1 = tagger_1()
    tagger2 = tagger_2(backoff=tagger1)

    # voting_tagger = EntropyVotingTagger([tagger1, tagger2])
    # result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    # print result

    voting_tagger = EntropyVotingTagger(tagger2, max_entropy=0.8)
    result = voting_tagger.tag(('hello', 'world', 'this', 'is', 'test'))
    print result
