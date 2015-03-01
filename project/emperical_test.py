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
from readers import NewsgroupsReader
from terminals import WordTermExtractor, get_document_objects, WordTerm
from terms_lists.ng20_ig import ng_20_ig500
from terms_lists.r8_ig import r_eight_terms
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


def if_then_else(input, output1, output2):
    return output1 if input else output2


def lte(n1, n2):
    return n1 >= n2


def individual_func(bool, tf, max_p_t_c, max_p_t_nc, avg_p_t_c, avg_p_t_nc, first_occ_perc):
    # this is the 93% one, from a micro run.
    # return protectedDiv(mul(tf, max_p_t_c),
    # sub(mul(sub(mul(mul(max_p_t_c, mul(tf, max_p_t_c)), max_p_t_c), avg_p_t_c), tf), avg_p_t_c))

    # as of 21/2 17:00 this is running on Hedwig
    # return add(add(sub(max_p_t_c, max_p_t_nc), add(sub(max_p_t_c, max_p_t_nc), sub(
    # add(add(sub(max_p_t_c, avg_p_t_nc), add(sub(max_p_t_c, max_p_t_nc), sub(add(tf, tf), avg_p_t_nc))), tf),
    # avg_p_t_nc))), add(add(sub(max_p_t_nc, max_p_t_c), tf), add(sub(max_p_t_c, avg_p_t_nc), add(tf, tf))))
    #
    # first one after adding word first occurrence.
    # return add(mul(protectedDiv(sub(avg_p_t_c, first_occ_perc), sub(max_p_t_nc, max_p_t_c)),
    # protectedDiv(sub(tf, avg_p_t_c), add(tf, avg_p_t_c))),
    # add(add(cos(first_occ_perc), mul(first_occ_perc, max_p_t_nc)),
    # sub(sub(tf, max_p_t_c), sub(first_occ_perc, first_occ_perc))))
    # long GP with first occurrence:
    # return add(tf, mul(protectedDiv(cos(protectedDiv(max_p_t_c, tf)), tf), mul(
    # mul(cos(protectedDiv(max_p_t_c, add(protectedDiv(tf, tf), max_p_t_c))), sub(max_p_t_nc, max_p_t_c)),
    # protectedDiv(mul(mul(cos(protectedDiv(max_p_t_c, max_p_t_nc)), sub(max_p_t_nc, max_p_t_c)),
    # protectedDiv(add(first_occ_perc, max_p_t_nc), add(tf, max_p_t_c))), add(tf, max_p_t_c)))))
    #
    # second long GP run (Yuri):
    # return protectedDiv(tf, cos(sub(sub(cos(max_p_t_c), max_p_t_nc), max_p_t_nc)))

    # 3rd long GP:
    # return mul(add(first_occ_perc, tf), cos(cos(first_occ_perc)))

    # return if_then_else(bool, sub(add(tf, avg_p_t_nc), mul(add(tf, tf), add(first_occ_perc, max_p_t_nc))),
    #                     sub(tf, mul(max_p_t_nc, tf)))

    #ng 20:
    return if_then_else(lte(protectedDiv(
        if_then_else(lte(sub(first_occ_perc, avg_p_t_c), if_then_else(bool, tf, avg_p_t_c)), avg_p_t_c, first_occ_perc),
        mul(if_then_else(
            lte(add(mul(tf, avg_p_t_nc), mul(avg_p_t_c, avg_p_t_c)), add(cos(avg_p_t_nc), mul(max_p_t_nc, avg_p_t_c))),
            mul(mul(cos(avg_p_t_c), if_then_else(bool, first_occ_perc, avg_p_t_nc)),
                sub(protectedDiv(avg_p_t_nc, avg_p_t_c), add(max_p_t_c, avg_p_t_c))),
            mul(mul(add(avg_p_t_c, tf), protectedDiv(max_p_t_c, max_p_t_nc)),
                if_then_else(lte(max_p_t_c, max_p_t_nc), protectedDiv(avg_p_t_nc, avg_p_t_c),
                             sub(first_occ_perc, max_p_t_c)))),
            mul(mul(protectedDiv(cos(max_p_t_c), sub(avg_p_t_nc, avg_p_t_c)), cos(cos(max_p_t_nc))),
                cos(sub(mul(max_p_t_nc, first_occ_perc), if_then_else(bool, avg_p_t_nc, avg_p_t_c)))))),
                            sub(cos(avg_p_t_c), if_then_else(bool, avg_p_t_c, avg_p_t_nc))),
                        mul(mul(add(first_occ_perc, avg_p_t_c), avg_p_t_c), cos(avg_p_t_c)),
                        if_then_else(lte(first_occ_perc, if_then_else(bool, first_occ_perc, first_occ_perc)),
                                     if_then_else(lte(tf, max_p_t_nc),
                                                  cos(mul(max_p_t_c, if_then_else(bool, avg_p_t_c, tf))), max_p_t_nc),
                                     mul(max_p_t_c, if_then_else(bool, avg_p_t_c, tf))))


if __name__ == "__main__":
    # add(protectedDiv(sub(p_c_t, tf), p_c_t), sub(p_c_t, tf))
    logger = ProjectParams.logger
    logger.setLevel(logging.DEBUG)
    logger.info("Starting program")
    # import pdb
    # pdb.set_trace()

    # cats_limiter = categories = ['gold', 'money-fx', 'trade']
    # R-10:
    cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply',
                                 'ship']

    # R-10:
    # cats_limiter = [u'earn', u'acq', u'crude', u'trade', u'money-fx', u'interest', u'money-supply', u'ship', u'sugar',
    # u'coffee']

    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(cats_limiter))

    training_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
    print len(set(training_docs_categories))
    training_documents = get_document_objects(training_documents, training_docs_categories)
    # training_documents = NewsgroupsReader(True).get_training()
    # training_docs_categories = [d.category for d in training_documents]

    tws_calculator = TWSCalculator(training_documents, training_docs_categories)
    word_term_extractor = WordTermExtractor(training_documents, tws_calculator)
    # doc = documents[0]
    # train_docs = training_documents[:250]
    # todo we take terms from the dev set in the k-fold, which might hurt generalization (but if it works we're OK..)
    # top_terms = word_term_extractor.top_common_words(500)

    top_terms = ng_20_ig500
    top_terms = map(lambda x: WordTerm(x), top_terms)

    print "using terms:"
    print [str(w) for w in top_terms]
    # top_terms = word_term_extractor.top_max_ig(500)

    feature_extractor = FeatureExtractor(training_documents, tws_calculator, top_terms)

    # fitness_calculator = TWSFitnessCalculator(SVC(), training_documents, feature_extractor)

    def if_then_else(input, output1, output2):
        return output1 if input else output2

    # now we're done training
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids(cats_limiter))
    test_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    test_documents = get_document_objects(test_documents, test_docs_categories)


    # test_documents = NewsgroupsReader(True).get_test()
    # test_docs_categories = [d.category for d in test_documents]

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

    classifier = OneVsRestClassifier(MultinomialNB())
    classifier.fit(train_matrix, training_docs_categories)

    predictions = classifier.predict(test_matrix)
    metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='macro')
    metrics_per_c = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions)

    print test_docs_categories
    print predictions

    print "Metrics per category (percision, recall, fmeasure):", metrics_per_c
    print "Metrics (percision, recall, fmeasure):", metrics

    accuracy = accuracy_score(test_docs_categories, predictions)

    print "Accuracy:", accuracy
    print "usenet 20 with 500 by freq"
    print "individual: long GP with first occurrence"
