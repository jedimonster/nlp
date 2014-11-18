from collections import defaultdict
import random
from nltk import AffixTagger, ConditionalFreqDist, FreqDist, corpus
import math


class EntropyAffixTagger(AffixTagger):

    def _calc_distribution_of_suffix(self, tag_count_dict):
        overall = 0
        for item in tag_count_dict:
            overall += tag_count_dict[item]
        dist = {}
        for item in tag_count_dict:
            dist[item] = float(float(tag_count_dict[item])/overall)
        #print dist
        return dist

    def entropy(self, dist):
        #print "************"
        ent = 0
        dist = self._calc_distribution_of_suffix(dist)
        for key, p in dist.items():

            #print key
            #print p

            if p == 0:
                continue
            else:
                ent += p * math.log(p)
        #print -ent
        #print "************"
        return -ent

    def _train(self, tagged_corpus, cutoff=1, verbose=False):
        token_count = hit_count = 0
        #cutoff = 0.99
        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.
        fd = ConditionalFreqDist()
        for sentence in tagged_corpus:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                # Record the event.
                token_count += 1
                context = self.context(tokens, index, tags[:index])
                if context is None:
                    continue
                fd[context][tag] += 1
                # If the backoff got it wrong, this context is useful:
                if (self.backoff is None or
                    tag != self.backoff.tag_one(tokens, index, tags[:index])):
                    useful_contexts.add(context)
        print useful_contexts

        # filter out affixes that have entropy lower than cutoff
        for affix in useful_contexts:
            dist = fd[affix]

            if self.entropy(dist) <= cutoff:
                self._context_to_tag[affix] = fd[affix].max()
            else:
                print "NOT ADDING!!! ",self.entropy(dist), "  ", cutoff, "  ",self._calc_distribution_of_suffix(dist)


all_words = corpus.brown.tagged_sents(tagset='universal', categories='news')
# random.shuffle(all_words)  # we shuffle it so we don't get a specific category as the test set!
ds_length = len(all_words)
train = all_words[:int(0.1 * ds_length)]
dev = all_words[int(0.1 * ds_length):int(0.2 * ds_length)]
test = all_words[int(0.2) * ds_length:]
#t =  EntropyAffixTagger(train=train,cutoff=0.3)

# range_of_cuttofs = range(-5,10)
# for r in range_of_cuttofs:
#     t =  EntropyAffixTagger(train=train,cutoff=1)
#
#     print t.evaluate(test)
# print "aaaaaaaa"
# t = EntropyAffixTagger(train=train, cutoff=500)
#print t.evaluate(test)

nt = AffixTagger(train=train)
print nt.evaluate(test)
from nltk.tag import UnigramTagger
u0 = UnigramTagger(train)
print "U0", u0.evaluate(test)

t2 =  EntropyAffixTagger(train=train, cutoff=1.01, backoff=u0)
t3 = AffixTagger(train=train,backoff=u0)
print t2.evaluate(test)
print t3.evaluate(test)

