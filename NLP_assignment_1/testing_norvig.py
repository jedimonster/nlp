from collections import defaultdict
import difflib
import itertools
import numpy

__author__ = 'michael'

import string
from wordSegment import segment2
from nltk.corpus import brown

############################################
#    Testing Norvig's Word Segmenter
############################################

def brown_to_segment2_format(sent):
    translation_table = {ord(c): None for c in string.punctuation}
    return [word.lower().translate(translation_table) for word in sent]

def accuracy_of_segment2(segment_func, sents):
    ratios = []

    for sent in sents:
        formatted_words = brown_to_segment2_format(sent)
        segment2_words = segment_func(''.join(formatted_words))[1]
        ratios.append(difflib.SequenceMatcher(None, formatted_words, segment2_words).ratio())

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
        word = brown_to_segment2_format([word])[0]
        res[word] += 1
    return res

def build_model_2w(words):
    res = defaultdict(lambda: 0)
    for w1, w2 in pairwise(words):
        w1, w2 = brown_to_segment2_format([w1, w2])
        res[(w1, w2)] += 1
    return res

def write_models_to_files():

    print 'Building new models'
    words = brown.words()[:1000000]
    model_1w = build_model_1w(words)
    model_2w = build_model_2w(words)

    f = open('small_1w.txt', 'w+')
    for k, v in model_1w.items():
        if not k:
            continue
        f.write('%s\t%d\n' % (k, v))

    f = open('small_2w.txt', 'w+')
    for k, v in model_2w.items():
        if not k[0] or not k[1]: #if empty strings
            continue

        f.write('%s %s\t%d\n' % (k[0], k[1], v))


#uncomment next section to get results
#
# import wordSegment_small
#
# write_models_to_files()
#
# print 'Testing Accuracy on *first* 150 sentences'
# accuracy_of_segment2(wordSegment_small.segment2, brown.sents()[:150])
#
# print 'Testing Accuracy on *last* 150 sentences'
# accuracy_of_segment2(wordSegment_small.segment2, brown.sents()[-150:])

"""
Answer: with smaller model (1,000,000 words, not all brown corpus) we got 0.92 accuracy
when testing on first 150 sentences in brown.
(This is cheating because out test & train data are the same).
When testing on last 150 sentences in brown corpus the accuracy dropped to 0.66
The original data files performed better on new data.
Both original data and our data performed the same on brown corpus (but we cheated).
"""





