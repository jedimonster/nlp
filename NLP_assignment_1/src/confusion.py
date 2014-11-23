# import nltk
import nltk
from tagger import DefaultTagger
from nltk import *
from voting import EntropyVotingTagger, EntropyUnigramTagger, EntropyAffixTagger

__author__ = 'itay'


def confusion_matrix(corpus_test, tagger):
    tagged_words = sum(corpus_test, [])
    ref = [token for word, token in tagged_words]
    test = [str(token) for word, token in tagger.tag(nltk.tag.untag(tagged_words))]

    matrix = nltk.metrics.ConfusionMatrix(ref, test)
    return matrix


if __name__ == "__main__":
    all_words = corpus.brown.tagged_sents(tagset='universal')
    ds_length = len(all_words)
    train = all_words[int(0.2 * ds_length):]
    dev = all_words[:int(0.1 * ds_length)]
    test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]

    class_tagger = UnigramTagger(train=train, backoff=AffixTagger(train=train, backoff=DefaultTagger('NOUN')))
    print "success rate for class tagger = ", class_tagger.evaluate(dev)
    print confusion_matrix(dev, class_tagger)
    #
    # proposed_tagger = EntropyVotingTagger(max_entropy=0.65,
    #                                       taggers=[EntropyUnigramTagger(train), EntropyAffixTagger(train)],
    #                                       backoff=DefaultTagger('NOUN'))
    #
    proposed_tagger = NgramTagger(2, train, backoff=class_tagger)
    print "success rate for proposed tagger = ", proposed_tagger.evaluate(dev)
    print confusion_matrix(dev, proposed_tagger)
    print proposed_tagger.evaluate(dev)