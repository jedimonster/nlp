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


def smooth_probability(xs, epsilon=0.001):
    zero_count = xs.count(0)
    nonzero_count = len(xs) - zero_count
    return map(lambda x: epsilon if x == 0 else x + ((zero_count * epsilon) / nonzero_count), xs)


def smooth_probabilities(prob_tuples, epsilon=0.001):
    xs, ys = map(list, zip(*prob_tuples))
    xs = smooth_probability(xs, epsilon)
    ys = smooth_probability(ys, epsilon)
    return zip(xs, ys)

def kl_divergence(probs_tuples):
    kl = 0
    probs_tuples = smooth_probabilities(probs_tuples)
    for probx, proby in probs_tuples:
        kl += probx * math.log(probx / proby)

    return kl

def validate_divergence(grammar, observed_cond_freqdist):
    conds = set([prod.lhs() for prod in grammar.productions()])

    for cond in conds:
        empirical_probdist = nltk.MLEProbDist(observed_cond_freqdist[cond])

        test_probs = grammar.productions(cond)
        test_probs = map(lambda prod: (prod.rhs(), prod.prob()), test_probs)
        # print test_probs
        probs_tuples = [(empirical_probdist.prob(nonterminal), prob) for nonterminal, prob in test_probs]
        yield cond, kl_divergence(probs_tuples)


def pcfg_validate_divergence(pcfg1, pcfg2):
    conds = set([p.lhs() for p in pcfg1.productions()] + [p.lhs() for p in pcfg2.productions()])

    for cond in conds:
        cfg1_prods = pcfg1.productions(cond)
        cfg1_derivations = defaultdict(int, {prod.rhs(): prod.prob() for prod in cfg1_prods})

        cfg2_prods = pcfg2.productions(cond)
        cfg2_derivations = defaultdict(int, {prod.rhs(): prod.prob() for prod in cfg2_prods})

        all_derivations = set(cfg1_derivations.keys() + cfg2_derivations.keys())

        probs_tuples = [(cfg1_derivations[d], cfg2_derivations[d]) for d in all_derivations]

        yield cond, kl_divergence(probs_tuples)


# def pcfg_validate(pcfg1, pcfg2):
#
#     productions1 = pcfg1.productions()
#     productions2 = pcfg2.productions()
#
#     #make production1 and pcfg1 the bigger
#     if len(productions1) < len(productions2):
#         productions1, productions2 = productions2, productions1
#         pcfg1, pcfg2 = pcfg2, pcfg1
#
#     prob_tuples = []
#
#     for prod in productions1:
#         other_prob = pcfg_get_production_prob(pcfg2, prod)
#         prob_tuples.append((prod.prob(), other_prob))
#
#     return kl_divergence(prob_tuples)



def pcfg_get_production_prob(pcfg, production):
    prods = pcfg.productions(production.lhs(), production.rhs()[0])

    for p in prods:
        if p.rhs() == production.rhs():
            return p.prob()

    return 0


def simplify_functional_tag(tag):
    if '-' in tag:
        tag = tag.split('-')[0]
    return tag


def get_tag(tree):
    if isinstance(tree, Tree):
        return Nonterminal(simplify_functional_tag(tree.label()))
    else:
        return tree


def leads_to_something(tree):
    if isinstance(tree, Tree):
        return tree.label() != '-NONE-' and any((leads_to_something(c)) for c in tree)

    return tree != '-NONE-'


def tree_to_production(tree):
    return Production(get_tag(tree), [get_tag(child) for child in tree])


def tree_to_productions(tree):
    yield tree_to_production(tree)
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions(child):
                yield prod


def filter_tree(tree):
    if isinstance(tree, Tree):
        return Tree(tree.label(), map(filter_tree, filter(leads_to_something, list(tree))))
    else:
        return tree  # tehee


def print_leaves(tree):
    """

    :type tree: Tree
    """
    # tree.
    pass


if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    sents = treebank.parsed_sents()
    # t = 0
    # o = 0
    # productions = []
    # for s in sents:
    #     l = list(tree_to_productions(filter_tree(s)))
    #     productions += l
    #     # print l
    #     t += len(l)
    #     o += len(s.productions())
    #
    # print "productions = ", t, "out of", o

    from nltk.grammar import induce_pcfg

    def get_productions(sents, number_of_trees):
        productions = []
        for s in sents[:number_of_trees]:
            productions += tree_to_productions(filter_tree(s))

        return productions

    pcfg_200 = induce_pcfg(Nonterminal('S'), get_productions(sents, 200))
    pcfg_400 = induce_pcfg(Nonterminal('S'), get_productions(sents, 400))

    print 'Number of rules in pcfg_200: ',  len(pcfg_200.productions())
    print 'Number of rules in pcfg_400: ', len(pcfg_400.productions())

    for i in pcfg_validate_divergence(pcfg_200, pcfg_400):
        print i

    # s = sents[490]  # print list(tree_to_productions(s))
    # print s.productions()
    # print list(tree_to_productions(s))
    # print len(s.productions())
    # print len(list(tree_to_productions(s)))
    #
    # prods = tree_to_productions(s)
    # print ([str(p) for p in prods])
