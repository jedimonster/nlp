"""
Test best individuals using hands
"""

import logging
from deap import base
from deap import creator
from deap import tools

import operator
import math
from deap.gp import *
from deap import algorithms
import multiprocessing
from nltk.corpus import reuters
from scipy.sparse import vstack

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from features import TWSCalculator
from fitness import TWSFitnessCalculator, FeatureExtractor
from parameters import ProjectParams
from terminals import WordTermExtractor, get_document_objects
from tws_gp import to_lower


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def cos(a):
    return math.cos(a)


sin = math.sin


def mul(x, y):
    return x * y


def individual_func(tf, max_p_c_t, max_p_c_nt, a, b):
    return sub(mul(protectedDiv(add(max_p_c_t, tf), cos(max_p_c_nt)),
                   add(sub(max_p_c_t, max_p_c_t), protectedDiv(tf, max_p_c_t))), mul(
        sub(add(protectedDiv(sub(mul(max_p_c_nt, max_p_c_nt), tf), cos(max_p_c_nt)), max_p_c_nt),
            protectedDiv(mul(max_p_c_nt, max_p_c_nt), tf)), sub(mul(max_p_c_t, max_p_c_t), add(max_p_c_t, tf))))


if __name__ == "__main__":
    # add(protectedDiv(sub(p_c_t, tf), p_c_t), sub(p_c_t, tf))
    logger = ProjectParams.logger
    logger.setLevel(logging.DEBUG)
    logger.info("Starting program")
    # import pdb
    # pdb.set_trace()

    # cats_limiter = categories = ['gold', 'money-fx', 'trade']
    cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply',
                                 'ship']  # top 8
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(cats_limiter))

    training_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
    print len(set(training_docs_categories))

    training_documents = get_document_objects(training_documents, training_docs_categories)
    # import pdb
    # pdb.set_trace()
    tws_calculator = TWSCalculator(training_documents, training_docs_categories)
    word_term_extractor = WordTermExtractor(training_documents, tws_calculator)
    # doc = documents[0]
    # train_docs = training_documents[:250]
    # todo we take terms from the dev set in the k-fold, which might hurt generalization (but if it works we're OK..)
    top_terms = word_term_extractor.top_common_words(2000)
    # top_terms = word_term_extractor.top_max_ig(500)

    feature_extractor = FeatureExtractor(training_documents, tws_calculator, top_terms)

    # fitness_calculator = TWSFitnessCalculator(SVC(), training_documents, feature_extractor)

    def if_then_else(input, output1, output2):
        return output1 if input else output2

    # import pdb
    # pdb.set_trace()

    # now we're done training
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids(cats_limiter))
    test_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    test_documents = get_document_objects(test_documents, test_docs_categories)
    WordTermExtractor(test_documents, TWSCalculator(test_documents, test_docs_categories))  # just to get counts

    print "test document 0 "
    print test_documents[0].doc
    print "train document 0"
    print training_documents[0].doc

    func = individual_func

    train_feature_vectors = [feature_extractor.get_weighted_features(func, doc) for doc in
                             training_documents]
    test_feature_vectors = [feature_extractor.get_weighted_features(func, doc) for doc in
                            test_documents]

    train_matrix = vstack(train_feature_vectors)
    test_matrix = vstack(test_feature_vectors)

    # print train_matrix
    # import pdb
    # pdb.set_trace()

    classifier = MultinomialNB()
    # print train_matrix
    classifier.fit(train_matrix, training_docs_categories)

    predictions = classifier.predict(test_matrix)
    metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='macro')

    print test_docs_categories
    print predictions

    print "Metrics (percision, recall, fmeasure):", metrics

    accuracy = accuracy_score(test_docs_categories, predictions)

    print "Accuracy:", accuracy

