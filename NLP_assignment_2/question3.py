from nltk import NaiveBayesClassifier, DictionaryProbDist, sum_logs, SklearnClassifier


class OneVsAllNaiveBayesClassifier(NaiveBayesClassifier):
    def prob_classify(self, featureset):
        # we know for a fact we've seen all the features before because we use constant words.

        # we now want to define a dummy label representing the average of all labels save for one at a time.

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                # if (label, fname) in self._feature_probdist:
                feature_probs = self._feature_probdist[label, fname]
                logprob[label] += feature_probs.logprob(fval)
                # else:
                # nb: This case will never come up if the
                # classifier was created by
                # NaiveBayesClassifier.train().
                # logprob[label] += sum_logs([])  # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)

# we create two classes, where the only feature linearly separates them:
a_featureset = ({'money': 10, 'bits': 0}, 'finance')
b_featureset = ({'money': 0, 'bits': 10}, 'computers')

classifier = NaiveBayesClassifier.train((a_featureset, b_featureset))
# print [classifier.classify({0: i}) for i in range(10)]
classes = [classifier.classify({'money': 1, 'bits': 9}),
           classifier.classify({'money': 9, 'bits': 1}),
           classifier.classify({'money': 0, 'bits': 9})]

print classes

from sklearn.naive_bayes import GaussianNB

# classifier = SklearnClassifier(GaussianNB(), sparse=False)
# classifier.train((a_featureset, b_featureset))
# print [classifier.classify({'feature': i}) for i in range(10)]
# probs = classifier.prob_classify_many({'feature': i} for i in range(10))
# # probs =  [classifier.prob_classify({0: i}) for i in range(10)]
# print probs