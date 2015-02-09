from nltk.corpus import reuters


class AbstractTerm(object):
    """
    Represents any kind of term - basic word, ngram, skipgrams, ...
    in the simplest form, WordTerm, _frequency counts the occurrences of the word in the original document.
    more advance forms can apply any sort of preprocessing on the document - stemming, PoS tagging, ...
    """

    def __init__(self):
        self._frequencies = {}

    def frequency(self, document):
        # todo fix this memoization so that each document remember the terms that appear in it.
        # if isinstance(document, list):
        # document = tuple(document)
        # if document not in self._frequencies:
        # self._frequencies[document] = self._frequency(document)
        #
        # return self._frequencies[document]
        return self._frequency(document)

    def _frequency(self, document):
        """
        :param document: document as represented in nltk  - array of words.
        :return: frequency of the term in the document.
        """
        raise NotImplemented

    def __str__(self):
        raise NotImplemented


class WordTerm(AbstractTerm):
    def __init__(self, word):
        AbstractTerm.__init__(self)
        self._word = word

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and other._word == self._word)

    def __hash__(self):
        # todo incorporate the fact this is a WordTerm
        return hash(self._word)


    def _frequency(self, document):
        return document.count(self._word)

    def __str__(self):
        return str(self._word)

    def __repr__(self):
        return self.__class__.__name__ + " " + str(self._word)


if __name__ == '__main__':
    training_fileids = fileids = filter(lambda x: "training" in x, reuters.fileids())
    documents = reuters.sents(training_fileids)
    # dict = set(reuters.words(training_fileids))

    print documents[0]
    print " ".join(documents[0])
    print WordTerm("in").frequency(documents[0])
