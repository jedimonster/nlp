from collections import defaultdict
import math
import nltk
import scipy

__author__ = 'itay'
from nltk import PCFG
from nltk import Tree, DictionaryProbDist
from nltk.grammar import Nonterminal, Production


def sample_tree(grammar, current_symbol):
    if not isinstance(current_symbol, Nonterminal):
        return current_symbol
    # sample a RHS, then recursively generate trees for each of the symbols in RHS.
    possible_prods = grammar.productions(current_symbol)
    prods_dist = DictionaryProbDist(dict(((production, production.prob()) for production in possible_prods)))

    # sample a production:
    prod = prods_dist.generate()

    # now we need to return a tree whose root is LHS, and childs are recursive calls to self.
    return Tree(prod.lhs(), [sample_tree(grammar, symbol) for symbol in prod.rhs()])


def pcfg_generate(grammar):
    """
    return a tree sampled from the language described by the PCFG grammar
    :type grammar: PCFG
    :param grammar: grammar to sample from, represented as PCFG.
    """
    start_symbol = grammar.start()
    return sample_tree(grammar, start_symbol)


def sample_corpus(n, generator, filename="./out/toy_pcfg2.gen"):
    with open(filename, "a") as fh:
        fh.truncate(0)
        for i in range(n):
            fh.write(str(generator()).replace("\n", ""))
            fh.write("\n")


def extract_trees(filename="./out/toy_pcfg2.gen"):
    trees = []
    with open(filename) as fh:
        for line in fh:
            trees.append(Tree.fromstring(line))

    return trees


def extract_freqdist(trees):
    """
    extracts a ConditionalFreqDist from the trees, where the Cond is any Nonterminal encountered.
    :param trees:
    :return:
    """
    freqdist = nltk.ConditionalFreqDist()
    for t in trees:
        _extract_freqdist(t, freqdist)

    return freqdist


def _extract_freqdist(tree, freqdist):
    for production in tree.productions():
        # Tree.productions() returns a list of RHS even if it's a terminal, we go ahead and extract it
        # production_rhs = production.rhs() if len(production.rhs()) > 1 else production.rhs()[0]
        # then count the occurrence.
        freqdist[production.lhs()][production.rhs()] += 1


def kl_divergence(probs_tuples):
    kl = 0
    for probx, proby in probs_tuples:
        if probx != 0:
            kl += probx * math.log(probx / proby)

    return kl


def validate_divergence(grammar, observed_cond_freqdist):
    conds = set([prod.lhs() for prod in grammar.productions()])

    for cond in conds:
        empirical_probdist = nltk.MLEProbDist(observed_cond_freqdist[cond])

        test_probs = grammar.productions(cond)
        test_probs = map(lambda prod: (prod.rhs(), prod.prob()), test_probs)
        # print test_probs
        probs_tuples = [(empirical_probdist.prob(nonterminal), prob ) for nonterminal, prob in test_probs]
        yield cond, kl_divergence(probs_tuples)


if __name__ == '__main__':
    # sample_corpus(1000, lambda: pcfg_generate(nltk.grammar.toy_pcfg2))
    observed_freqdist = extract_freqdist(extract_trees())
    test_grammar = nltk.grammar.toy_pcfg2
    divs = validate_divergence(test_grammar, observed_freqdist)

    print([d for d in divs])
