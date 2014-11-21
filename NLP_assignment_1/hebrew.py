# coding=utf-8
__author__ = 'michael'

from BguCorpusReader import BguCorpusReader
c = BguCorpusReader()
tagged_words = c.tagged_words()
print tagged_words


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
        print prefix_tag
        possible_prefixes = PREFIXES[prefix_tag] + ALL_PREFIXES

        if prefix_tag == 'DEF' and word[word_index] != PREFIXES['DEF'][0]:
            print 'Special Case. Invisible "Ha"'
            result.append(PREFIXES['DEF'][0])
            continue

        for prefix in possible_prefixes:
            if word[word_index:].startswith(prefix):
                print 'FOUND!!'
                result.append(word[word_index:len(prefix)])
                word_index += len(prefix)
                print prefix
                break

    result.append(word[word_index:])

    return result


for i in range(2000):
    w,t = tagged_words[i]
    print 'Word ', w
    print 'Raw', t.getRaw()
    print 'BguTag', t.getBguTag()
    print 'PosTag', t.getPosTag()
    print 'MyCode'
    res = getWords(w,t)
    for r in res:
        print "%s," % (r,),
    print