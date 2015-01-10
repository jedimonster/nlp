__author__ = 'user'

from nltk import induce_pcfg, Nonterminal, ViterbiParser, Tree
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk.treetransforms import chomsky_normal_form
from question1 import filter_tree, tree_to_productions, pcfg_cnf_learn
from question2 import *


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
        if (item[0].label().split('^')[0].split('|')[0] == con[0].label().split('^')[0].split('|')[0]) and (item[1] == con[1]) and (item[2] == con[2]):
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
        distance = item[2]-item[1]+1
        label = item[0].label()
        if label not in ACCURACY_PER_LABEL:
            ACCURACY_PER_LABEL[label] = {'total': 0, 'matches': 0}
        if distance not in ACCURACY_PER_DISTANCE_LABELED:
            ACCURACY_PER_DISTANCE_LABELED[distance] = {'total': 0, 'matches': 0}
        ACCURACY_PER_LABEL[label]['total'] += 1
        ACCURACY_PER_DISTANCE_LABELED[distance]['total'] += 1
        if exist_same(item, origin_cons):
            ACCURACY_PER_DISTANCE_LABELED[distance]['matches'] += 1
            ACCURACY_PER_LABEL[label]['matches'] += 1
            pre_count += 1

    for item in origin_cons:
        if exist_same(item, guess_cons):
            recall_count += 1

    recall = float(recall_count)/float(origin_len)
    precision = float(pre_count)/float(guess_len)
    f_measure = 2*(recall*precision)/(recall + precision)

    return precision, recall, f_measure

ACCURACY_PER_DISTANCE = {}
ACCURACY_PER_DISTANCE_LABELED = {}
ACCURACY_PER_LABEL = {}


def calculate_index_metrics(origin_cons, guess_cons):
    origin_indexes = set([(x[1], x[2]) for x in origin_cons])
    guess_indexes = set([(x[1], x[2]) for x in guess_cons])
    origin_len = len(origin_indexes)
    guess_len = len(guess_indexes)
    pre_count = 0
    recall_count = 0
    for item in guess_indexes:
        distance = item[1]-item[0]+1
        if distance not in ACCURACY_PER_DISTANCE:
            ACCURACY_PER_DISTANCE[distance] = {'total': 0, 'matches': 0}

        ACCURACY_PER_DISTANCE[distance]['total'] += 1

        if item in origin_indexes:
            ACCURACY_PER_DISTANCE[distance]['matches'] += 1
            pre_count += 1

    for item in origin_indexes:
        if item in guess_indexes:
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
    a, b, c = calculate_joint_metrics(origin_cons, guess_cons)
    d, e, f = calculate_index_metrics(origin_cons,guess_cons)
    return (a,b), (d,e)


def eval_trees(trees, parser, pcfg):
    counter = 0
    overall_prec_labeled = 0
    overall_recall_labeled = 0
    overall_prec_index = 0
    overall_recall_index = 0
    for tree in trees:
        counter += 1
        tokens = pos_uncovered_tokens(tree.leaves(), pcfg)
        guess_tree = parser.parse(tokens)
        guess_tree = list(guess_tree)[0]
        # guess_tree.draw()
        # tree.draw()
        labaled_metrics, index_metrics = eval_tree(tree, guess_tree)
        pre_labeled, recall_labeled = labaled_metrics
        pre_index, recall_index = index_metrics
        overall_prec_labeled += pre_labeled
        overall_recall_labeled += recall_labeled
        overall_prec_index += pre_index
        overall_recall_index += recall_index

    overall_recall_labeled = overall_recall_labeled/float(counter)
    overall_prec_labeled = overall_prec_labeled/float(counter)
    overall_recall_index = overall_recall_index/float(counter)
    overall_prec_index = overall_prec_index/float(counter)

    print "precision for labeled: ", overall_prec_labeled
    print "recall_labeled: ", overall_recall_labeled
    print "fmeasure labeled ", 2*(overall_prec_labeled*overall_recall_labeled)/(overall_prec_labeled+overall_recall_labeled)

    print "precision for index: ", overall_prec_index
    print "recall_index: ", overall_recall_index
    print "fmeasure index ", 2*(overall_prec_index*overall_recall_index)/(overall_prec_index+overall_recall_index)


def calculate_accuracy_per_distance():
    x_axis = ACCURACY_PER_DISTANCE.keys()
    x_axis.sort()
    y_axis = [ACCURACY_PER_DISTANCE[x]['matches']/float(ACCURACY_PER_DISTANCE[x]['total']) for x in x_axis]
    x_axis_labeled = ACCURACY_PER_DISTANCE_LABELED.keys()
    x_axis_labeled.sort()
    y_axis_labeled = [ACCURACY_PER_DISTANCE_LABELED[x]['matches']/float(ACCURACY_PER_DISTANCE_LABELED[x]['total']) for x in x_axis_labeled]
    print x_axis
    print y_axis
    import matplotlib.pyplot as plt
    plt.title("Accuracy per distance")
    plt.scatter(x_axis, y_axis, c="blue", marker='*', label="accuracy index")
    plt.scatter(x_axis_labeled, y_axis_labeled, c="red", marker='o', label="accuracy label", alpha=0.5)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    trees = treebank.parsed_sents()

    trees = trees[:5]
    cleaned_trees = [filter_tree(tree) for tree in trees]
    for t in cleaned_trees:
        chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')

    parser, pcfg = get_parser(cleaned_trees)
    eval_trees(cleaned_trees, parser, pcfg)

    print "----------- Reporting Per Label -----------"
    print ACCURACY_PER_LABEL
    print len(ACCURACY_PER_LABEL)
    for item in ACCURACY_PER_LABEL:
        print item, "--- total -------> ", ACCURACY_PER_LABEL[item]['total']
        print item, "--- precision ---> ", ACCURACY_PER_LABEL[item]['matches']/float(ACCURACY_PER_LABEL[item]['total'])
    print '&'*100

    print ACCURACY_PER_DISTANCE
    calculate_accuracy_per_distance()