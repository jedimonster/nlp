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

__author__ = 'itay'


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evalOne(one, pset):
    return 0.5


def to_lower(str):
    return str.lower()


if __name__ == '__main__':
    logger = ProjectParams.logger
    logger.setLevel(logging.DEBUG)
    logger.info("Starting program")

    # cats_limiter = categories = ['gold', 'money-fx', 'trade']
    # cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply', 'ship'
    # ]  # top 8
    # cats_limiter = ['veg-oil', 'retail', 'bop', 'nat-gas', 'copper'] # bunch of small, evenly sized, categories.
    # training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
    # reuters.fileids(cats_limiter))
    #
    # training_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in training_fileids]
    # training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
    # training_documents = get_document_objects(training_documents, training_docs_categories)

    training_documents = NewsgroupsReader(True).get_training()
    training_docs_categories = [d.category for d in training_documents]

    tws_calculator = TWSCalculator(training_documents, training_docs_categories)
    word_term_extractor = WordTermExtractor(training_documents, tws_calculator)
    # doc = documents[0]
    # train_docs = training_documents[:250]
    # todo we take terms from the dev set in the k-fold, which might hurt generalization (but if it works we're OK..)
    # top_terms = word_term_extractor.top_common_words(2000)
    # top_terms = word_term_extractor.top_max_ig(500)
    top_terms = map(lambda x: WordTerm(x), ng_20_ig500)

    feature_extractor = FeatureExtractor(training_documents, tws_calculator, top_terms)

    fitness_calculator = TWSFitnessCalculator(OneVsRestClassifier(MultinomialNB()), training_documents,
                                              feature_extractor)

    def if_then_else(input, output1, output2):
        return output1 if input else output2

    def lte(n1, n2):
        return n1 >= n2

    # pset = PrimitiveSetTyped("MAIN", [bool, float, float, float, float, float], float)
    pset = PrimitiveSetTyped("MAIN", [bool, float, float, float, float, float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)
    # pset.addPrimitive(math.cos, [float], float)
    # pset.addPrimitive(if_then_else, [bool, float, float], float)
    # pset.addPrimitive(lte, [float, float], bool)
    pset.addPrimitive(lambda x, y: min(x, y), [float, float], float, name="minOfTwo")
    pset.addPrimitive(lambda x, y: max(x, y), [float, float], float, name="maxOfTwo")
    pset.addEphemeralConstant("random1", lambda: random.random(), float)
    # pset.addEphemeralConstant("random2", lambda: random.random(), float)

    pset.renameArguments(ARG0='bool')
    pset.renameArguments(ARG1='tf')
    pset.renameArguments(ARG2='max_p_t_c')
    pset.renameArguments(ARG3='max_p_t_nc')
    pset.renameArguments(ARG4='avg_p_t_c')
    pset.renameArguments(ARG5='avg_p_t_nc')
    pset.renameArguments(ARG6='first_occ_perc')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    pool = multiprocessing.Pool()
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=3, max_=7, type_=pset.ret)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", compile, pset=pset)

    # toolbox.register("evaluate", evalOne, pset=pset)
    toolbox.register("evaluate", fitness_calculator.evaluate, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePoint)
    toolbox.register("expr_mut", genFull, min_=0, max_=6)
    toolbox.register("mutate", mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=17))

    # toolbox.register("map", futures.map)


    pop = toolbox.population(n=25)
    hof = tools.HallOfFame(5)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10,
                                   halloffame=hof, verbose=True)

    for i in pop:
        print str(i)


    # now we're done training
    # test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
    # reuters.fileids(cats_limiter))
    # test_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in test_fileids]
    # test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    # test_documents = get_document_objects(test_documents, test_docs_categories)

    test_documents = NewsgroupsReader(True).get_test()
    test_docs_categories = [d.category for d in test_documents]

    WordTermExtractor(test_documents, TWSCalculator(test_documents, test_docs_categories))  # just to get counts

    print "---"
    print "testing with words ", top_terms
    # print "test document 0 "
    # print test_documents[0].doc
    # print "train document 0"
    # print training_documents[0].doc
    # best_individual = hof.items[0]
    for best_individual in hof.items:
        print "individual: ", str(best_individual)

        func = toolbox.compile(best_individual)

        train_feature_vectors = [feature_extractor.get_weighted_features(func, doc) for doc in
                                 training_documents]
        test_feature_vectors = [feature_extractor.get_weighted_features(func, doc) for doc in
                                test_documents]

        train_matrix = vstack(train_feature_vectors)
        test_matrix = vstack(test_feature_vectors)

        # print train_matrix
        # import pdb
        # pdb.set_trace()
        print "Testing with Naive Bayes"

        classifier = OneVsRestClassifier(MultinomialNB())
        # print train_matrix
        classifier.fit(train_matrix, training_docs_categories)

        predictions = classifier.predict(test_matrix)
        metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='macro')

        print test_docs_categories
        print predictions

        print "Metrics (percision, recall, fmeasure):", metrics

        accuracy = accuracy_score(test_docs_categories, predictions)

        print "Accuracy:", accuracy

        print "---"
        # print "Testing with SVC"
        # classifier = SVC()
        # # print train_matrix
        # classifier.fit(train_matrix, training_docs_categories)
        #
        # predictions = classifier.predict(test_matrix)
        # metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='macro')
        #
        # print test_docs_categories
        # print predictions
        #
        # print "Metrics (percision, recall, fmeasure):", metrics
        #
        # accuracy = accuracy_score(test_docs_categories, predictions)
        #
        # print "Accuracy:", accuracy

