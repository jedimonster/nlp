import codecs
import logging
import os
import pickle
import shelve
import deap
import math

from nltk.corpus import reuters
from scipy.sparse import vstack
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC

from features import TWSCalculator
from parameters import ProjectParams
from terminals import WordTermExtractor


__author__ = 'itay'


class FeatureExtractor(object):
    def __init__(self, train_documents, tws_calculator, terms):
        self._terms = terms
        self._tws_calculator = tws_calculator
        self._train_documents = train_documents
        self._terminals = shelve.open('./cache/ig', protocol=pickle.HIGHEST_PROTOCOL)
        self.logger = ProjectParams.logger
        self._VERSION = 1.0

        self.get_terminals()
        self.logger.info("syncing terminals to disk")
        self._terminals.sync()
        self.logger.info("done synching")


    def get_terminals(self):
        logger = self.logger
        logger.info("Prefetching terminals for trees")
        total_docs = len(self._train_documents)
        for i, doc in enumerate(self._train_documents):
            if i % 50 == 0:
                logger.debug("document %d/%d", i, total_docs)

            relevant_terms = set(self._terms).intersection(doc.get_all_terms())
            for term in relevant_terms:
                if repr((doc.index, term)) not in self._terminals:
                    self._terminals[repr((doc.index, term))] = self._tws_calculator.raw_terminals(term, doc)
        pass

        #
        # cache_path = "./cache/ng"
        # if os.path.isfile(cache_path):
        # with codecs.open(cache_path, 'r') as cache_file:
        # self._terminals = pickle.load(cache_file)
        #
        # logger.info("done prefetching (loaded from disk)")
        #
        # else:
        # open(cache_path, 'a').close()  # aka touch
        # cache_file = file(cache_path, 'w')
        # for i, doc in enumerate(self._train_documents):
        #         if i % 50 == 0:
        #             logger.debug("document %d/%d", i, total_docs)
        #
        #         relevant_terms = set(self._terms).intersection(doc.get_all_terms())
        #         for term in relevant_terms:
        #             self._terminals[doc, term] = self._tws_calculator.raw_terminals(term, doc)
        #
        #     pickle.dump(self._terminals, cache_file)
        #     logger.info("done prefetching")

    def get_weighted_features(self, individual_func, document):
        doc_features = dict((k, 0) for k in self._terms)
        vectorizer = DictVectorizer()

        relevant_terms = set(self._terms).intersection(document.get_all_terms())

        for term in relevant_terms:
            # tws = self._tws_calculator.tws(individual, term, document)
            # tws = self._tws_calculator.terminals(term, document)
            tws = self.terminals(document, term)

            # tws is array of (bool, tf, tf-idf, tf-ig, tf-chi, tf-rf)
            doc_features[term] = abs(individual_func(*tws))

        # print doc_features
        vector = vectorizer.fit_transform(doc_features)
        return vector


    def terminals(self, document, term):
        if term not in document.get_all_terms():
            raise RuntimeError("requested term which doesn't appear in document, optimize it out!")
        if repr((document.index, term)) in self._terminals:
            return self._terminals[repr((document.index, term))]

        return self._tws_calculator.raw_terminals(term, document)


class TWSFitnessCalculator(object):
    k_fold = ProjectParams.k_fold

    def __init__(self, classifier, training_documents, features_extractor):
        """
        Fitness calculator for the TWS GP.
        :param classifier: one of sklearn's classifier, expected to support fit and predict.
        :param training_documents: collection of document objects.
        :param features_extractor: collection of AbstractTerms.
        """
        super(TWSFitnessCalculator, self).__init__()
        self._features_extractor = features_extractor
        self._classifier = classifier
        self._training_docs = training_documents
        self._chunk_size = len(self._training_docs) / self.k_fold
        self.logger = ProjectParams.logger

    def evaluate_lambda(self, func):
        # calculate the vectors once, then build matrices to train/test k times
        logger = self.logger
        fmeasures = []
        logger.debug("getting weighted feature vectors")
        feature_vectors = [self._features_extractor.get_weighted_features(func, doc) for doc in self._training_docs]

        for k in range(self.k_fold):
            self.logger.debug("k= " + str(k))
            test_feature_vectors, test_categories = zip(
                *self._featureset_chunk(k, zip(feature_vectors, [d.category for d in self._training_docs])))
            train_feature_vectors, train_categories = zip(*sum(
                (self._featureset_chunk(i, zip(feature_vectors, [d.category for d in self._training_docs])) for i in
                 range(self.k_fold) if i != k), []))

            # test = self._docs_chunk(k)
            # train = sum((self._docs_chunk(i) for i in range(self.k_fold) if i != k), [])
            # train_categories = [d.category for d in
            # train]  # todo since these are now vectors we need to get the documents for the categories.
            # test_categories = [d.category for d in test]

            # logger.debug("getting train feature vectors")
            # train_feature_vectors = [self._features_extractor.get_weighted_features(func, doc) for doc in
            # train]
            # logger.debug("getting test feature vectors")
            # test_feature_vectors = [self._features_extractor.get_weighted_features(func, doc) for doc in
            #                         test]

            train_matrix = vstack(train_feature_vectors)
            test_matrix = vstack(test_feature_vectors)

            logger.debug("training classifier")
            self._classifier.fit(train_matrix, train_categories)
            logger.debug("predicting...")
            predictions = self._classifier.predict(test_matrix)

            logger.debug("calculating metrics")
            fmeasure = sklearn.metrics.precision_recall_fscore_support(test_categories, predictions, average='macro')[2]
            # todo try micro or weighted average. macro is too penalizing..
            fmeasures.append(fmeasure)
            logger.debug("done k, f-measure = %f", fmeasure)
        # print fmeasures
        fitness = sum(fmeasures) / len(fmeasures)
        logger.info("fitness = %f", fitness)
        return [fitness]

    def evaluate(self, individual, pset):
        """

        :param individual: lambda that takes term features and returns a TWS.
        :return:
        """
        self.logger.info("Calculating fitness")
        self.logger.debug("for " + str(individual))
        # func = toolbox.compile(expr=individual)
        func = deap.gp.compile(individual, pset)
        return self.evaluate_lambda(func)

    def classify(self, documents):
        """

        :param documents: collection of tuples (document, category).
        :return: list of tuples (real_category, predicted_category)
        """
        pass

    def _featureset_chunk(self, i, featureset):
        """
        returns the i'th segment of the documents in the k-fold.
        :param i: chunk to return.
        :return: list of documents
        """
        return featureset[i * self._chunk_size: (i + 1) * self._chunk_size]


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

    top_terms = word_term_extractor.top_max_ig(200)
    print top_terms
    # doc0terms = list(map(lambda w: WordTerm(w), set(doc)))

    feature_extractor = FeatureExtractor(documents, tws_calculator, top_terms)

    feature_vectors = [feature_extractor.get_weighted_features(None, doc) for doc in documents[:300]]

    train_matrix = vstack(feature_vectors[:250])
    test_matrix = vstack(feature_vectors[250:300])

    print train_matrix

    classifier = SVC()

    classifier.fit(train_matrix, docs_categories[:250])

    predictions = classifier.predict(test_matrix)
    results = docs_categories[250:300]

    print zip(predictions, results)

