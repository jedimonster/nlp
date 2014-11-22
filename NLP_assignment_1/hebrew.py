# coding=utf-8
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
print 'Calculating Accuracy of hebrew model...'

all_sents = []
sent = []
for w, t in tagged_words:
        sent += getWords(w, t)
        if len(sent) >= 30:
            all_sents.append(sent)
            sent = []


accuracy_of_segment2(wordSegment_hebrew.segment2, all_sents[:200])
"""
Answer:
Accuracy of segment2 on hebrew words is 0.8
(But again we are cheating, using same data for train & test)
"""

