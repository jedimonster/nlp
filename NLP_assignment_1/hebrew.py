# coding=utf-8
import itertools

__author__ = 'michael'

from BguCorpusReader import BguCorpusReader
c = BguCorpusReader()
tagged_words = c.tagged_words()

#Note: This dict is handmade using Meni Adler's api.
ALL_PREFIXES = [u'ה', u'ב', u'כ', u'ל', u'מ', u'מכש' u'מב', u'בכש',  u'ו', u'כש',u'מש', u'לכש', u'ש']
PREFIXES = {
    'DEF': [u'ה'],
    'PREPOSITION': [u'ב', u'כ', u'ל', u'מ', u'מכש' u'מב', u'בכש'],
    'CONJ': [u'ו'],
    'TEMP': [u'כש', u'מש', u'לכש'],
    'REL': [u'ש']
}
#Sort them by length
ALL_PREFIXES.sort(key=len, reverse=True)
for k in PREFIXES:
    PREFIXES[k].sort(key=len, reverse=True)

def getWords(word, tag):

    prefix_tags = tag.getBguTag()[0]
    word_index = 0
    result = [] # (prefix_index, prefix len)

    for prefix_tag, x in prefix_tags:
        # print prefix_tag

        if prefix_tag in PREFIXES:
            possible_prefixes = PREFIXES[prefix_tag] + ALL_PREFIXES
        else:
            possible_prefixes = ALL_PREFIXES

        if prefix_tag == 'DEF' and word[word_index] != PREFIXES['DEF'][0]:
            # print 'Special Case. Invisible "Ha"'
            result.append(PREFIXES['DEF'][0])
            continue

        for prefix in possible_prefixes:
            if word[word_index:].startswith(prefix):
                # print 'FOUND!!'
                result.append(word[word_index:len(prefix)])
                word_index += len(prefix)
                # print prefix
                break

    result.append(word[word_index:])

    return result

from testing_norvig import segment2_format, write_models_to_files, accuracy_of_segment2

def build_hebrew_models(tagged_words):
    all_words = []
    for w, t in tagged_words:
        words = getWords(w, t)
        words = segment2_format(words)
        all_words += words

    write_models_to_files(all_words, 'hebrew_1w.txt', 'hebrew_2w.txt')

#Uncomment this line to build hebrew models (takes time..)
# build_hebrew_models(tagged_words)

############################################
# Test segment() performance on hebrew model
############################################

import wordSegment_hebrew
from testing_norvig import accuracy_of_segment2_hebrew
print 'Calculating Accuracy of hebrew model...'


def segment2_hebrew(sents_no_spaces):
    res = wordSegment_hebrew.segment2(sents_no_spaces)[1]
    output = []
    orig_index = 0
    while orig_index < len(res)-2:
        word = res[orig_index]
        while res[orig_index] in ALL_PREFIXES and orig_index < len(res)-2:
            word += res[orig_index+1]
            orig_index += 1

        output.append(word)
        orig_index += 1

    return output

def get_hebrew_sentences(tagged_words, how_much):
    all_sents = []
    sent = []
    if how_much:
        tagged_words = tagged_words[:how_much]

    for w, t in tagged_words:
            # sent += getWords(w, t)
            sent.append(w)
            if len(sent) >= 30:
                all_sents.append(sent)
                sent = []

    return all_sents

# print 'Calculating Accuracy *without* aggregation..'
# accuracy_of_segment2(wordSegment_hebrew.segment2, get_hebrew_sentences(tagged_words, 100))

# print 'Calculating Accuracy *with* aggregation..'
# accuracy_of_segment2_hebrew(segment2_hebrew, get_hebrew_sentences(tagged_words, 100))

"""
Answer:
Calculating Accuracy *without* aggregation..
Accuracy of segment2(): 0.52121271246210821

Calculating Accuracy *with* aggregation..
Accuracy of segment2(): 0.73326348833278887
"""