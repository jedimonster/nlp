__author__ = 'user'

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

def exist_same(con, cons_list):
    for item in cons_list:
        if (item[0].label() == item[0].label()) and (item[1] == con[1]) and (item[2] == con[2]):
            return True
    return False


def calculate_joint_metrics(origin_cons, guess_cons):
    origin_cons = list(origin_cons)
    guess_cons = list(guess_cons)
    origin_len = len(list(origin_cons))
    guess_len = len(list(guess_cons))

    pre_count = 0
    recall_count = 0
    # calculate precision
    for item in guess_cons:
        if exist_same(item, origin_cons):
            pre_count += 1

    for item in origin_cons:
        if exist_same(item, guess_cons):
            recall_count += 1

    recall = float(recall_count)/float(origin_len)
    precision = float(pre_count)/float(guess_len)
    f_measure = 2*(recall*precision)/(recall + precision)

    return precision, recall, f_measure


def calculate_index_metrics(origin_cons, guess_cons):

    origin_indexes = set([(x[1], x[2]) for x in origin_cons])
    guess_indexes = set([(x[1], x[2]) for x in guess_cons])
    origin_len = len(origin_indexes)
    guess_len = len(guess_indexes)
    pre_count = 0
    recall_count = 0
    for item in guess_indexes:

        if item in origin_indexes:
            pre_count += 1

    for item in origin_indexes:
        if item in guess_indexes:
            recall_count += 1

    recall = float(recall_count)/float(origin_len)
    precision = float(pre_count)/float(guess_len)
    f_measure = 2*(recall*precision)/(recall + precision)

    return precision, recall, f_measure


def calculate_labeled_metrics(origin_cons, guess_cons):

    origin_labels = set([x[0].label() for x in origin_cons])
    guess_labels = set([x[0].label() for x in guess_cons])

    origin_len = len(origin_labels)
    guess_len = len(guess_labels)

    pre_count = 0
    recall_count = 0
    for item in guess_labels:
        if item in origin_labels:
            pre_count += 1

    for item in origin_labels:
        if item in guess_labels:
            recall_count += 1

    recall = float(recall_count)/float(origin_len)
    precision = float(pre_count)/float(guess_len)
    f_measure = 2*(recall*precision)/(recall + precision)

    return precision, recall, f_measure

def eval_tree(orig_tree, guess_tree):
    origin_cons = tree_to_constituents(orig_tree)
    guess_cons = tree_to_constituents(guess_tree)
    origin_cons = list(origin_cons)
    guess_cons = list(guess_cons)
    # print "*"*100
    # print "index d  ", calculate_index_metrics(origin_cons, guess_cons)
    # print "joint    ", calculate_joint_metrics(origin_cons, guess_cons)
    # print "label_only   ", calculate_labeled_metrics(origin_cons, guess_cons)
    a, b, c = calculate_joint_metrics(origin_cons, guess_cons)
    return a, b

def eval_trees(treebank, parser):
    for tree in treebank:
        pass


if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    trees = treebank.parsed_sents()
    #
    eighty_perc = int(len(trees) * 0.8)
    training_trees = pcfg_cnf_learn(treebank, eighty_perc)
    test_trees = trees[eighty_perc:]

    # print len(training_trees), len(test_trees)

    training_prods = sum([list(tree_to_productions(t)) for t in training_trees], list())
    # test_prods = sum([list(tree_to_productions(t)) for t in test_trees], list())

    training_pcfg = induce_pcfg(Nonterminal("S"), training_prods)

    parser = ViterbiParser(training_pcfg)

    tree = filter_tree(test_trees[0])
    chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
    tree_to_constituents(tree)
    for c in tree_to_constituents(tree):
        print c

    # tree.draw()
    eval_tree(tree, tree)

    eval_trees()