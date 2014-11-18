"""
assignment 1
"""
import datetime
import google
import nltk
from nltk import UnigramTagger
from tagger import DefaultTagger, TaggerI
from nltk.corpus import brown
from nltk import NgramTagger
from nltk.corpus.reader import tagged
from nltk.tokenize import sent_tokenize
import codecs
import sys
from nltk.tag import untag




def MostAmbiguousWords(corpus, n):
    """
    function that finds words with more than N observed tags
    """

    cfd = nltk.ConditionalFreqDist(corpus.tagged_words(tagset='universal'))
    for item in cfd.keys():
        if len(cfd[item]) <= n:
            del cfd[item]
    return cfd


def TestMostAmbiguousWords(cfd, n):

    print 'find most ambiguous (n=%d)' % (n,)
    res = []
    for item in cfd:
        if len(cfd[item]) > n:
            res.append(item)
            print item
    return res


def ShowExamples_v2(word, cfd, corpus):
    """
    show examples of word with different tag
    """
    print "Exemples for word: ", word
    tagged_sentences = corpus.tagged_sents(tagset='universal')
    words_tags = cfd[word]
    examples = {}

    for tag in words_tags:
            sent = next((sent for sent in tagged_sentences if (word, tag) in sent))
            print tag, " ---> ", untag(sent)
            examples[tag] = ' '.join(untag(sent))

    return examples

def ShowExamples(word, cfd, corpus):
    """
    show examples of word with different tag
    """
    print "Exemples for word: ", word
    flag = 0
    tagged_sentences = corpus.tagged_sents(tagset='universal')
    words_tags = cfd[word]
    examples = []
    for tag in words_tags:
        for sent in tagged_sentences:
            for word_tag in sent:
                if word_tag[0] == word and word_tag[1] == tag:
                    print tag, " ---> ", untag(sent)
                    examples.append(sent)
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0

# todo rename this
def plot_nice_data():
    import pylab

    cfd = nltk.ConditionalFreqDist(brown.tagged_words(tagset='universal'))
    tags_by_word = map(len, cfd.values())
    max_tags = max(tags_by_word)
    c = [tags_by_word.count(i) for i in range(1, max_tags)]
    print(c)
    pylab.title('Number of words in brown corpus per number of tags')
    pylab.ylabel('Number of Words')
    pylab.xlabel('Number of tags')
    pylab.plot(range(1, max_tags), c, '-bo')
    pylab.show()


def get_data_from_web(search_query, best_tagger):
    """
    doc
    """
    links_dict = google.google(search_query)
    clean_text = ""
    tokens = []
    sents = []
    for key, value in links_dict.items():
        tuple_text = value[1]
        clean_text += tuple_text[0]
        tokens += tuple_text[2]
        # check if we got enough text( around 50 sents)
        sents += sent_tokenize(clean_text)
        print len(sents)
        if len(sents) >= 50:
            break
    print len(sents)
    clean_text_file = codecs.open("clean/clean_text.txt", 'w', encoding="utf-8").write(clean_text)
    tagged_tokens = best_tagger.tag(tokens)

    tagged_text = "".join([word + "/" + str(tag) + " " for word, tag in tagged_tokens])
    tagged_text = tagged_text.replace("./.", "./.\n")
    # print tagged_text
    tagged_text_file = codecs.open("tagged/tagged_text.txt", 'w', encoding="utf-8").write(tagged_text)


def report_most_ambigous(cfd, corpus):
    words = TestMostAmbiguousWords(cfd, 4)
    report = ""
    for word in words:
        report += '---------- %s ----------\r\n' % (word,)
        examples = ShowExamples_v2(word, cfd, corpus)
        for tag, sent in examples.iteritems():
            report += '%s --> %s\r\n' % (tag, sent)
        report += '\n'

    open('MostAmbiguous.txt', 'w+').write(report)

if __name__ == "__main__":

    corpus = brown
    print 'Running ConditionalFreqDist (will take some time..)'
    full_cfd = nltk.ConditionalFreqDist(brown.tagged_words(tagset='universal'))
    print 'Done'
    # cfd = MostAmbiguousWords(brown, 4)
    # print len(cfd)

    report_most_ambigous(full_cfd, corpus)

    # start_time = datetime.datetime.now()
    # ShowExamples('open', full_cfd, brown)
    # print 'Total Time for V1: %s' % (datetime.datetime.now() - start_time)
    # 
    # start_time = datetime.datetime.now()
    # ShowExamples_v2('open', full_cfd, brown)
    # print 'Total Time for V2: %s' % (datetime.datetime.now() - start_time)

    # hw_tagged = brown.tagged_sents(categories='homework')
    # brown_news_tagged = brown.tagged_sents(categories='news', tagset='universal')
    # brown_train = brown_news_tagged[0:]
    # brown_test = brown_news_tagged[:100]
    # # query = google.google("NLTK")
    # # google.AnalyzeResults(query)
    # default_tagger = DefaultTagger('NOUN')
    # ugram_tagger = UnigramTagger(brown_train, backoff=None)
    # # text = ['Hello', "World", "sdklj", "sdfdsf", "frgfg", ".", "!"]
    # # res = ugram_tagger.tag_sents([text])
    # affix_tagger = nltk.AffixTagger(brown_train, backoff=default_tagger)
    # one_gram_tagger = NgramTagger(1, train=brown_train, backoff=affix_tagger)
    # two_gram_tagger = NgramTagger(2, train=brown_train, backoff=one_gram_tagger)
    #
    # get_data_from_web("How to eat an apple", two_gram_tagger)

    # from nltk.corpus import brown
    # print brown.tagged_sents(categories='homework')

    # print two_gram_tagger.evaluate(hw_tagged)


