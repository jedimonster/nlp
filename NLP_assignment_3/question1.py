from collections import defaultdict
import math
import nltk
from nltk.corpus import LazyCorpusLoader
from nltk.corpus import BracketParseCorpusReader
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


def simplify_functional_tag(tag):
    if '-' in tag:
        tag = tag.split('-')[0]
    return tag


def get_tag(tree):
    if isinstance(tree, Tree):
        return Nonterminal(simplify_functional_tag(tree.label()))
    else:
        return tree


def leads_to_none(tree):
    if isinstance(tree, Tree):
        return not any((not leads_to_none(c)) for c in tree)

    return tree is None


def tree_to_production(tree):
    return Production(get_tag(tree), [get_tag(child) for child in tree if not leads_to_none(child)])


def tree_to_productions(tree):
    production = tree_to_production(tree)
    if production.rhs() and production.rhs() != production.lhs():
        yield production
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions(child):
                if len(prod.rhs()) != 0:
                    yield prod
                    # else:
                    # print child


def print_leaves(tree):
    """

    :type tree: Tree
    """
    # tree.
    pass


if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    sents = treebank.parsed_sents()
    t = 0
    o = 0
    for s in sents[:200]:
        l = list(tree_to_productions(s))
        print l
        t += len(l)
        o += len(s.productions())

    print t, o

    s = sents[490]  # print list(tree_to_productions(s))
    # print s.productions()
    # print list(tree_to_productions(s))
    # prods = tree_to_productions(s)
    # print ([str(p) for p in prods])



