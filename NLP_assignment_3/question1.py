from collections import defaultdict
import math
import nltk
from nltk.corpus import LazyCorpusLoader
from nltk.corpus import BracketParseCorpusReader
from nltk.treetransforms import chomsky_normal_form
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
    probs_tuples = smooth_probabilities(probs_tuples)
    for probx, proby in probs_tuples:
        kl += probx * math.log(probx / proby)

    return kl


def smooth_probability(xs, epsilon=0.001):
    zero_count = xs.count(0)
    nonzero_count = len(xs) - zero_count
    return map(lambda x: epsilon if x == 0 else x + ((zero_count * epsilon) / nonzero_count), xs)


def smooth_probabilities(prob_tuples, epsilon=0.001):
    xs, ys = map(list, zip(*prob_tuples))
    xs = smooth_probability(xs, epsilon)
    ys = smooth_probability(ys, epsilon)
    return zip(xs, ys)


def validate_divergence(grammar, observed_cond_freqdist):
    conds = set([prod.lhs() for prod in grammar.productions()])

    for cond in conds:
        empirical_probdist = nltk.MLEProbDist(observed_cond_freqdist[cond])

        test_probs = grammar.productions(cond)
        test_probs = map(lambda prod: (prod.rhs(), prod.prob()), test_probs)
        # print test_probs
        probs_tuples = [(empirical_probdist.prob(nonterminal), prob) for nonterminal, prob in test_probs]
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


def tree_to_productions_parents(tree, parent_label):
    yield (parent_label, tree_to_production(tree))
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions_parents(child, tree.label()):
                yield prod


def filter_tree(tree):
    if isinstance(tree, Tree):
        return Tree(simplify_functional_tag(tree.label()), map(filter_tree, filter(leads_to_something, list(tree))))
    else:
        return tree  # teehee


def print_leaves(tree):
    """

    :type tree: Tree
    """
    # tree.
    pass


def pcfg_cnf_learn(treebank, n):
    for tree in treebank.parsed_sents()[:n]:
        tree = filter_tree(tree)
        chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
        yield tree


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


def get_productions(sents, number_of_trees):
    productions = []
    for s in sents[:number_of_trees]:
        productions += tree_to_productions(filter_tree(s))

    return productions


import pylab

if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    trees = treebank.parsed_sents()
    trees = map(filter_tree, trees)
    prods = sum([list(tree_to_productions_parents(t, 'DUKIS')) for t in trees], list())

    # only LHS = NP, no parent restriction:
    nps = filter(lambda prod: prod[1].lhs().symbol() == 'NP', list(prods))
    rhs = map(lambda x: x[1], nps)
    rhs_dist = nltk.FreqDist(rhs)
    print len(rhs_dist.keys())
    print rhs_dist.tabulate()

    # only LHS = NP below S
    nps_below_s = map(lambda x: x[1],
                      filter(lambda prod: prod[0] == 'S' and prod[1].lhs().symbol() == 'NP', list(prods)))
    nps_below_s_freq = nltk.FreqDist(nps_below_s)
    # print len(set(nps_below_s))
    # print nps_below_s[:10]

    # only LHS = NP below VP
    nps_below_vp = map(lambda x: x[1],
                       filter(lambda prod: prod[0] == 'VP' and prod[1].lhs().symbol() == 'NP', list(prods)))
    nps_below_vp_freq = nltk.FreqDist(nps_below_vp)

    nps_below_s_dist = nltk.MLEProbDist(nps_below_s_freq)
    nps_below_vp_dist = nltk.MLEProbDist(nps_below_vp_freq)

    samples = set(nps_below_s_dist.samples() + nps_below_vp_dist.samples())
    rhs_probs = [(nps_below_s_dist.prob(rhs), nps_below_vp_dist.prob(rhs)) for rhs in samples]

    print "NP BELOW S RHS:"
    print nps_below_s_dist.samples()

    print '--------'
    print 'NP BELOW VP RHS:'
    print  nps_below_vp_dist.samples()

    print sum([x[0] != 0 and x[1] != 0 for x in rhs_probs])

    print kl_divergence(rhs_probs)

    # print len(set(nps_below_vp))
    # print nps_below_vp[:10]
    # pylab.xlabel("Rule frequency")
    # pylab.ylabel("number of rules")
    # pylab.plot(rhs_dist.values(), '-bo', )
    # pylab.show()
    # for prod in prods:
    # print list(prod)
    # # if prod[1].lhs().label() == 'NP':
    # # print 'yippie'
