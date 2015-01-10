from nltk import induce_pcfg, Nonterminal, ViterbiParser, Tree, Production, DefaultTagger
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk.treetransforms import chomsky_normal_form
from question1 import filter_tree, tree_to_productions, pcfg_cnf_learn

GRAMMAR_TO_POS = {
    "CC": "CONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "ADVERB",
    "FW": "X",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": ".",
    "MD": "VERB",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "NOUN",
    "NNPS": "NOUN",
    "PDT": "DET",
    "POS": "PRON",
    "PRP": "PRON",
    "PRP$": "PRON",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "PRT",
    "SYM": ".",
    "TO": "ADP",
    "UH": "PRT",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "PRON",
    "WRB": "ADV",
}


def tree_to_constituents(tree):
    tree = tree.copy(True)
    enumerate_tree(tree)
    return _subtree_to_constituents(tree)


def _subtree_to_constituents(tree):
    subtree_leaves = tree.leaves()
    leftmost_leaf = subtree_leaves[0]
    rightmost_leaf = subtree_leaves[-1]
    yield (tree, leftmost_leaf, rightmost_leaf)

    if isinstance(tree, Tree):
        for subtree in tree:
            if isinstance(subtree, Tree):
                for const in _subtree_to_constituents(subtree):
                    yield const


def enumerate_tree(tree, i=0):
    if isinstance(tree, Tree):
        if tree.height() == 2:  # just enumerate the stupid leaf
            tree[0] = i
            return i + 1
        for subtree in tree:
            i = enumerate_tree(subtree, i)
        return i


from ass1 import *
from nltk import UnigramTagger


def get_pos_tagger():
    all_words = corpus.brown.tagged_sents(tagset='universal')
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    u0 = UnigramTagger(train, backoff=DefaultTagger("NOUN"))

    return EntropyAffixTagger(train=train, cutoff=0.5, backoff=u0)


tagger = get_pos_tagger()


def pos_uncovered_tokens(test_sentence, training_pcfg):
    pos = tagger.tag(test_sentence)
    print pos
    for i, token in enumerate(test_sentence):
        if token not in training_pcfg._lexical_index:
            test_sentence[i] = "$" + pos[i][1]

    return test_sentence


def get_parser(training_trees):
    training_prods = sum([list(tree_to_productions(t)) for t in training_trees], list())
    pos_rules = [Production(Nonterminal(lhs), ["$" + rhs]) for lhs, rhs in GRAMMAR_TO_POS.iteritems()]
    training_prods += pos_rules
    training_pcfg = induce_pcfg(Nonterminal("S"), training_prods)
    parser = ViterbiParser(training_pcfg)

    return parser, training_pcfg


if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    trees = treebank.parsed_sents()
    #
    eighty_perc = int(len(trees) * 0.8)
    training_trees, training_pcfs = pcfg_cnf_learn(treebank, eighty_perc)
    test_trees = trees[eighty_perc:]

    # test_prods = sum([list(tree_to_productions(t)) for t in test_trees], list())

    parser, training_pcfg = get_parser(training_trees)

    test_tree = filter_tree(test_trees[0])
    test_sentence = test_tree.leaves()
    test_sentence = pos_uncovered_tokens(test_sentence, training_pcfg)

    print list(parser.parse(test_sentence))

    # chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
    #
    # training_pcfg.check_coverage()

    # tree_to_constituents(tree)
    # for c in tree_to_constituents(tree):
    # print c

    # tree.draw()