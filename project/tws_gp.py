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
from scoop import futures
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from features import TWSCalculator
from fitness import TWSFitnessCalculator, FeatureExtractor
from parameters import ProjectParams
from terminals import WordTermExtractor, get_document_objects

__author__ = 'itay'


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evalOne(one, pset):
    return 0.5


if __name__ == '__main__':
    print 'test'
    pass

    print 'a'

    logger = ProjectParams.logger
    logger.setLevel(logging.DEBUG)
    logger.info("Starting program")

    # cats_limiter = categories = ['gold', 'money-fx', 'trade']
    cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply', 'ship',
                                 'sugar']  # top 9
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(cats_limiter))

    training_documents = [sum(reuters.sents(fid), []) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    training_documents = get_document_objects(training_documents, training_docs_categories)


    tws_calculator = TWSCalculator(training_documents, training_docs_categories)
    word_term_extractor = WordTermExtractor(training_documents, tws_calculator)
    # doc = documents[0]
    # train_docs = training_documents[:250]
    # todo we take terms from the dev set in the k-fold, which might hurt generalization (but if it works we're OK..)
    top_terms = word_term_extractor.top_common_words(500)
    # top_terms = word_term_extractor.top_max_ig(500)

    feature_extractor = FeatureExtractor(training_documents, tws_calculator, top_terms)

    fitness_calculator = TWSFitnessCalculator(SVC(), training_documents, feature_extractor)

    def if_then_else(input, output1, output2):
        return output1 if input else output2

    # pset = PrimitiveSetTyped("MAIN", [bool, float, float, float, float, float], float)
    pset = PrimitiveSetTyped("MAIN", [float, float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)
    pset.addPrimitive(math.cos, [float], float)
    # pset.addPrimitive(if_then_else, [bool, float, float], float)

    pset.renameArguments(ARG0='tf')
    pset.renameArguments(ARG1='p_c_t')
    pset.renameArguments(ARG2='p_c_nt')
    # pset.renameArguments(ARG3='tf_ig')
    # pset.renameArguments(ARG4='tf_chi')
    # pset.renameArguments(ARG5='tf_rf')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    pool = multiprocessing.Pool()
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=2, max_=4, type_=pset.ret)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", compile, pset=pset)

    # toolbox.register("evaluate", evalOne, pset=pset)
    toolbox.register("evaluate", fitness_calculator.evaluate, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePoint)
    toolbox.register("expr_mut", genFull, min_=0, max_=2)
    toolbox.register("mutate", mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=17))

    # toolbox.register("map", futures.map)


    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10,
                                   halloffame=hof, verbose=True)

    for i in pop:
        print str(i)


    # now we're done training
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids(cats_limiter))
    test_documents = [sum(reuters.sents(fid), []) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    test_documents = get_document_objects(test_documents, test_docs_categories)
    WordTermExtractor(test_documents, TWSCalculator(test_documents, test_docs_categories)) # just to get counts

    best_individual = hof.items[0]
    print "---"
    print "testing with words ", top_terms
    print "individual: ", str(best_individual)
    print "test document 0 "
    print test_documents[0].doc
    print "train document 0"
    print training_documents[0].doc

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

