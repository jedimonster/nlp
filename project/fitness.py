from logging import Logger
import logging
from nltk.corpus import reuters
from sklearn.feature_extraction import DictVectorizer
from features import TWSCalculator
from parameters import ProjectParams
from terminals import WordTerm, WordTermExtractor

__author__ = 'itay'


class FeatureExtractor(object):
    def __init__(self, train_documents, tws_calculator, terms):
        self._terms = terms
        self._tws_calculator = tws_calculator
        self._train_documents = train_documents

    def get_weighted_features(self, individual, document):
        doc_features = {}
        vectorizer = DictVectorizer()

        for t in self._terms:
            # tws = self._tws_calculator.tws(individual, t, document)
            tws = self._tws_calculator.terminals(t, document)
            # todo use individual to process tws's
            tws = sum(tws)
            doc_features[t] = tws

        vector = vectorizer.fit_transform(doc_features)

        return vector


class TWSFitnessCalculator(object):
    k_fold = ProjectParams.k_fold

    def __init__(self, classifier, training_data, features):
        """
        Fitness calculator for the TWS GP.
        :param classifier: one of sklearn's classifier, expected to support fit and predict.
        :param training_data: collection of tuples (document, class).
        :param features: collection of AbstractTerms.
        """
        super(TWSFitnessCalculator, self).__init__()
        self._terms = features
        self._classifier = classifier
        self._training_docs = training_data
        self._chunk_size = len(self._training_docs) / self.k_fold

    def evaluate(self, individual):
        """

        :param individual: lambda that takes term features and returns a TWS.
        :return:
        """
        for k in range(self.k_fold):
            test = self._docs_chunk(k)
            train = sum((self._docs_chunk(i) for i in range(self.k_fold) if i != k), [])
            self._classifier.fit()

    def classify(self, documents):
        """

        :param documents: collection of tuples (document, category).
        :return: list of tuples (real_category, predicted_category)
        """
        pass

    def _docs_chunk(self, i):
        """
        returns the i'th segment of the documents in the k-fold.
        :param i: chunk to return.
        :return: list of documents
        """
        return self._training_docs[i * self._chunk_size: (i + 1) * self._chunk_size]


if __name__ == '__main__':
    logger = ProjectParams.logger
    logger.setLevel(logging.DEBUG)

    logger.info("Starting program")

    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(categories=['gold', 'money-fx', 'trade']))
    documents = [sum(reuters.sents(fid), []) for fid in training_fileids]
    docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    tws_calculator = TWSCalculator(documents, docs_categories)
    word_term_extractor = WordTermExtractor(documents, tws_calculator)
    doc = documents[0]

    top_terms = word_term_extractor.top_max_ig(5)
    print top_terms
    # doc0terms = list(map(lambda w: WordTerm(w), set(doc)))

    feature_extractor = FeatureExtractor(documents, tws_calculator, top_terms)

    print feature_extractor.get_weighted_features(None, doc)
