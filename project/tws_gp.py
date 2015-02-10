import logging
from deap import base
from deap import creator
from deap import tools

import operator
import math
from deap.gp import *
from deap import algorithms
from nltk.corpus import reuters
from sklearn.svm import SVC
from features import TWSCalculator
from fitness import TWSFitnessCalculator, FeatureExtractor
from parameters import ProjectParams
from terminals import WordTermExtractor

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

    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(categories=['gold', 'money-fx', 'trade']))
    documents = [sum(reuters.sents(fid), []) for fid in training_fileids[:300]]
    docs_categories = [reuters.categories(fid)[0] for fid in training_fileids[:300]]

    tws_calculator = TWSCalculator(documents, docs_categories)
    word_term_extractor = WordTermExtractor(documents, tws_calculator)
    doc = documents[0]
    train_docs = documents[:250]
    top_terms = word_term_extractor.top_common_words(50)

    feature_extractor = FeatureExtractor(train_docs, tws_calculator, top_terms)
    train_categories = docs_categories[:250]
    fitness_calculator = TWSFitnessCalculator(SVC(), zip(train_docs, train_categories), feature_extractor)

    pset = PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)

    pset.renameArguments(ARG0='tf')
    pset.renameArguments(ARG1='tfidf')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=2, max_=8)
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

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 2,
                                   halloffame=hof, verbose=True)