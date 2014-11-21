from collections import defaultdict
import random
from nltk import AffixTagger, ConditionalFreqDist, FreqDist, corpus
import math



all_words = corpus.brown.tagged_sents(tagset='universal')
ds_length = len(all_words)
train = all_words[int(0.2 * ds_length):]
dev = all_words[:int(0.1 * ds_length)]
test = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]
# t =  EntropyAffixTagger(train=train,cutoff=0.3)
from nltk.tag import UnigramTagger

cutoffs = range(0, 15)
cutoffs = [x * 0.1 for x in cutoffs]
print cutoffs
u0 = UnigramTagger(train)
nt = AffixTagger(train=train, backoff=u0)

print "result of evaluating nltk affix tagger: ", nt.evaluate(test)
print "finding best cutoff"
# for cutoff in cutoffs:
#     print "cutoff: ", cutoff
#     eat = EntropyAffixTagger(train=train, cutoff=cutoff, backoff=u0)
#     print "evaluating entropy affix tagger: ", eat.evaluate(dev)
eat = EntropyAffixTagger(train=train, cutoff=500, backoff=u0)
print "evaluating entropy affix tagger: ", eat.evaluate(test)
# eat = EntropyAffixTagger(train=train, cutoff=0.5, backoff=u0)
# print "evaluating entropy affix tagger: ", eat.evaluate(test)


