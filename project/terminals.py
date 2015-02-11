from nltk.corpus import reuters
from parameters import ProjectParams


class AbstractTerm(object):
    """
    Represents any kind of term - basic word, ngram, skipgrams, ...
    in the simplest form, WordTerm, _frequency counts the occurrences of the word in the original document.
    more advance forms can apply any sort of preprocessing on the document - stemming, PoS tagging, ...
    """

    def __init__(self):
        self._frequencies = {}

    def frequency(self, document):
        # todo fix this memoization so that each document calculates and remember the terms that appear in it.

        # if isinstance(document, list):
        # document = tuple(document)
        # if document not in self._frequencies:
        # self._frequencies[document] = self._frequency(document)
        # 
        # return self._frequencies[document]
        return self._frequency(document)

    def get_terms(document):
        # todo memoization on the result
        raise NotImplemented

    # def _frequency(self, document):
    # """
    # :param document: document as represented in nltk  - array of words.
    #     :return: frequency of the term in the document.
    #     """
    #     raise NotImplemented

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


class WordTermExtractor(object):
    def __init__(self, documents, tws_calculator):
        self._tws_calculator = tws_calculator
        self._documents = documents
        self.logger = ProjectParams.logger

    def all_terms(self):
        return list(map(lambda w: WordTerm(w), set(sum(self._documents, []))))

    def top_max_ig(self, k):
        self.logger.info("calculating top %d word terms according to IG", k)
        terms = self.all_terms()

        self.logger.debug("starting IG algebra")
        term_ig = [(term, self._tws_calculator.max_ig(term)) for term in terms]
        term_ig = sorted(term_ig, key=lambda tig: tig[1], reverse=True)

        self.logger.info("returning top terms")
        return [term for term, ig in term_ig[:k]]

    def top_common_words(self, k):
        self.logger.info("calculating top %d word terms according to frequency", k)

        terms = self.all_terms()
        terms_freq = ((term, sum((term.frequency(doc) for doc in self._documents))) for term in terms)

        terms_freq = sorted(terms_freq, key=lambda x: x[1], reverse=True)

        self.logger.info("returning top %d word terms according to frequency", k)

        return [term for term, freq in terms_freq[:k]]


if __name__ == '__main__':
    training_fileids = fileids = filter(lambda x: "training" in x, reuters.fileids())
    documents = reuters.sents(training_fileids)
    # dict = set(reuters.words(training_fileids))

    print documents[0]
    print " ".join(documents[0])
    print WordTerm("in").frequency(documents[0])

