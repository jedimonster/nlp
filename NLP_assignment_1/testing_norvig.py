from collections import defaultdict
import difflib
import itertools
import codecs
import numpy

__author__ = 'michael'

import string
from wordSegment import segment2
from nltk.corpus import brown

############################################
#    Testing Norvig's Word Segmenter
############################################

TRANSLATION_TABLE = {ord(c): None for c in string.punctuation}

def segment2_format(sent):
    return [word.lower().translate(TRANSLATION_TABLE) for word in sent]

def accuracy_of_segment2_hebrew(segment_func, sents):
    ratios = []

    for sent in sents:
        formatted_words = segment2_format(sent)
        # print 'printing original sentences'
        # for f in formatted_words:
        #     print f,
        print '\n---------------------'
        segment2_words = segment_func(''.join(formatted_words))

        # for s in segment2_words:
        #     print s,
        # print '\n=============='
        accuracy = difflib.SequenceMatcher(None, formatted_words, segment2_words).ratio()
        # print accuracy
        ratios.append(accuracy)

    print 'Accuracy of segment2(): %r' % (numpy.mean(ratios),)


def accuracy_of_segment2(segment_func, sents):
    ratios = []

    for sent in sents:
        formatted_words = segment2_format(sent)
        # for f in formatted_words:
        #     print f,
        # print '---------------------'
        segment2_words = segment_func(''.join(formatted_words))[1]

        # for s in segment2_words:
        #     print s,
        accuracy = difflib.SequenceMatcher(None, formatted_words, segment2_words).ratio()
        # print accuracy
        ratios.append(accuracy)

    print 'Accuracy of segment2(): %r' % (numpy.mean(ratios),)

# uncomment this line to get the answer below
# accuracy_of_segment2(segment2, brown.sents()[:150])
"""
Answer: accuracy of segment2() is 0.9~ (not tested on all brown corpus)
"""


######################################################################
#   Testing Norvig's Word Segmenter Dependency on the Language Model
######################################################################

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def build_model_1w(words):
    res = defaultdict(lambda: 0)
    for word in words:
        word = segment2_format([word])[0]
        res[word] += 1
    return res

def build_model_2w(words):
    res = defaultdict(lambda: 0)
    for w1, w2 in pairwise(words):
        w1, w2 = segment2_format([w1, w2])
        res[(w1, w2)] += 1
    return res

def write_models_to_files(words, file1='small_1w.txt', file2='small_2w.txt'):

    print 'Building new models'
    model_1w = build_model_1w(words)
    model_2w = build_model_2w(words)

    f = codecs.open(file1, "w", encoding="utf-8")
    for k, v in model_1w.items():
        if not k:
            continue
        str = '%s\t%d\n' % (k, v)
        f.write(str)

    f = codecs.open(file2, "w", encoding="utf-8")
    for k, v in model_2w.items():
        if not k[0] or not k[1]: #if empty strings
            continue
        str = '%s %s\t%d\n' % (k[0], k[1], v)
        f.write(str)


#uncomment next section to get results
#
# import wordSegment_small
#
# words = brown.words()[:1000000]
# write_models_to_files(words)
#
# print 'Testing Accuracy on *first* 150 sentences'
# accuracy_of_segment2(wordSegment_small.segment2, brown.sents()[:150])
#
# print 'Testing Accuracy on *last* 150 sentences'
# accuracy_of_segment2(wordSegment_small.segment2, brown.sents()[-150:])

"""
Answer: with smaller model (1,000,000 words, not all brown corpus) we got 0.92 accuracy
when testing on first 150 sentences in brown.
(This is cheating because our test & train data are the same).
When testing on last 150 sentences in brown corpus the accuracy dropped to 0.66
The original data files performed better on new data.
Both original data and our data performed the same on learned brown data (but we cheated).
(Or Maybe They also "cheated" and learned from brown...)
"""





