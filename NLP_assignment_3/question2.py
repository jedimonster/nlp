from nltk import induce_pcfg, Nonterminal, ViterbiParser, Tree
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk.treetransforms import chomsky_normal_form
from question1 import filter_tree, tree_to_productions, pcfg_cnf_learn


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


# def enumerate_tree(tree):
# leaves_numbers = dict()
# leaves = tree.leaves()
#
# i = 0
# for leaf in leaves:
# leaves_numbers[id(leaf)] = i
# i += 1
#
# return leaves_numbers

# def enumerate_leaves(tree):
# for child in tree:
# if isinstance(child, Tree):
# child.
# else:



if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    trees = treebank.parsed_sents()
    #
    eighty_perc = int(len(trees) * 0.8)
    # training_trees = pcfg_cnf_learn(treebank, eighty_perc)
    test_trees = trees[eighty_perc:]
    #
    # print len(training_trees), len(test_trees)
    #
    # training_prods = sum([list(tree_to_productions(t)) for t in training_trees], list())
    # # test_prods = sum([list(tree_to_productions(t)) for t in test_trees], list())
    #
    # training_pcfg = induce_pcfg(Nonterminal("S"), training_prods)
    #
    # parser = ViterbiParser(training_pcfg)
    tree = filter_tree(test_trees[0])
    chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
    tree_to_constituents(tree)
    for c in tree_to_constituents(tree):
        print c

    tree.draw()