import logging
from deap import base
from deap import creator
from deap import tools

import operator
import math
from deap.gp import *
from deap import algorithms
from nltk.corpus import reuters
from scipy.sparse import vstack
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from features import TWSCalculator
from fitness import TWSFitnessCalculator, FeatureExtractor
from parameters import ProjectParams
from terminals import WordTermExtractor, get_document_objects

__author__ = 'itay'

if __name__ == '__main__':
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    logger = ProjectParams.logger
    logger.setLevel(logging.DEBUG)

    logger.info("Starting program")

    # cats_limiter = categories = ['gold', 'money-fx', 'trade']
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids())
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids())

    training_documents = [sum(reuters.sents(fid), []) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    training_documents = get_document_objects(training_documents, training_docs_categories)

    test_documents = [sum(reuters.sents(fid), []) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]

    tws_calculator = TWSCalculator(training_documents, training_docs_categories)
    word_term_extractor = WordTermExtractor(training_documents, tws_calculator)
    # doc = documents[0]
    # train_docs = training_documents[:250]
    # todo we take terms from the dev set in the k-fold, which might be problematic.
    top_terms = word_term_extractor.top_common_words(500)

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

    pset.renameArguments(ARG0='bool')
    pset.renameArguments(ARG1='tf')
    pset.renameArguments(ARG2='tf_idf')
    # pset.renameArguments(ARG3='tf_ig')
    # pset.renameArguments(ARG4='tf_chi')
    # pset.renameArguments(ARG5='tf_rf')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=3, max_=6, type_=pset.ret)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", compile, pset=pset)

    toolbox.register("evaluate", fitness_calculator.evaluate, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePoint)
    toolbox.register("expr_mut", genFull, min_=0, max_=2)
    toolbox.register("mutate", mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50,
                                   halloffame=hof, verbose=True)

    for i in pop:
        print str(i)

    best_individual = hof.items[0]
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
    classifier.fit(train_matrix, training_docs_categories)

    predictions = classifier.predict(test_matrix)
    metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='macro')

    print "Metrics (percision, recall, fmeasure):", metrics