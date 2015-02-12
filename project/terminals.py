import collections
from nltk.corpus import reuters
import numpy
from parameters import ProjectParams
from sklearn.feature_extraction.text import CountVectorizer


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

    def _frequency(self, document):
        """
        :param document: document as represented in nltk  - array of words.
        :return: frequency of the term in the document.
        """

        raise NotImplemented

    def __str__(self):
        raise NotImplemented


class WordTerm(AbstractTerm):
    TERM_ID = 1

    def __init__(self, word):
        AbstractTerm.__init__(self)
        self._word = word

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and other._word == self._word)

    def __hash__(self):
        # todo incorporate the fact this is a WordTerm
        return hash(self._word)


    def _frequency(self, document):
        return document.get_freq(self)

    def __str__(self):
        return str(self._word)

    def __repr__(self):
        return self.__class__.__name__ + " " + str(self._word)


class WordTermMatrix(object):
    def __init__(self, sklearn_matrix, words_mapping, total_freq):
        self.sklean_matrix = sklearn_matrix
        self.words_mapping = words_mapping
        self.total_freq = total_freq

    def get_freq(self, doc_index, word):
        if word not in self.words_mapping:
            return 0

        word_index = self.words_mapping[word]
        return self.sklean_matrix[doc_index][(0, word_index)]


def get_document_objects(documents, categories):
    doc_obj = []
    for i, doc in enumerate(documents):
        doc_obj.append(Document(i, doc, categories[i]))

    return doc_obj


class Document(object):
    def __init__(self, index, doc, category):
        self.category = category
        self.index = index
        self.doc = doc
        self.MAX_TERM_TYPES = 10
        self._freqs = {}


    def set_freqs(self, term_id, freqs_dict):
        self._freqs[term_id] = freqs_dict

    def get_freq(self, word_term):
        # return ProjectParams.terms_matrix.get_freq(self.index, word_term)
        if self._freqs[word_term.TERM_ID] != 0:
            return self._freqs[word_term.TERM_ID][word_term]
        else:
            raise IndexError("no frequencies for types of terms.")


class WordTermExtractor(object):
    def __init__(self, documents, tws_calculator):
        self._tws_calculator = tws_calculator
        self._documents = documents
        self.logger = ProjectParams.logger
        self._total_freq = {}
        self._build_term_matrix()


    def _build_term_matrix(self):
        # vectorizer = CountVectorizer(lowercase=False)
        # documents_str = [' '.join(doc.doc) for doc in self._documents]
        # matrix = vectorizer.fit_transform(documents_str)

        self._total_freq = collections.defaultdict(int)
        for doc in self._documents:
            word_freq = collections.defaultdict(int)
            for word in doc.doc:
                word = WordTerm(word)
                word_freq[word] += 1
                self._total_freq[word] += 1

            doc.set_freqs(WordTerm.TERM_ID, word_freq)

            # mapping = {w: i for i, w in zip(range(len(vectorizer.get_feature_names())), vectorizer.get_feature_names())}

            # calc total freq
            # total_freq = collections.defaultdict(int)
            # for doc in self._documents:
            # for word in doc.doc:
            # if word in mapping:
            # total_freq[word] += 1

            # ProjectParams.terms_matrix = WordTermMatrix(matrix, mapping, total_freq)

    def all_terms(self):
        return map(lambda w: WordTerm(w), ProjectParams.terms_matrix.words_mapping.keys())

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

        # terms = self.all_terms()
        # terms_freq = [(term, sum((term.frequency(doc) for doc in self._documents))) for term in terms]
        terms_freq = sorted(self._total_freq.items(), key=lambda x: x[1], reverse=True)

        self.logger.info("returning top %d word terms according to frequency", k)

        return [term for term, freq in terms_freq[:k]]


if __name__ == '__main__':
    training_fileids = fileids = filter(lambda x: "training" in x, reuters.fileids())
    documents = reuters.sents(training_fileids)
    # dict = set(reuters.words(training_fileids))

    # print documents[0]
    # print " ".join(documents[0])
    # print WordTerm("in").frequency(documents[0])

    print 'Checking Vectorizer'
    documents = get_document_objects(documents)
    w = WordTermExtractor(documents, None)
    print documents[0].get_freq('BAHIA')
    print documents[0].get_freq('bahia')

    print ProjectParams.terms_matrix.total_freq