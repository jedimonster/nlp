from nltk import NaiveBayesClassifier, DictionaryProbDist, sum_logs


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


from nltk.corpus import reuters
from collections import defaultdict

words = reuters.words()
word_freq = defaultdict(int)
for word in words:
    word_freq[word] += 1

sorted_words = sorted(set(words), key=word_freq.get, reverse=True)
most_frequent_words = filter(lambda w: len(w) > 1, sorted_words)


def bag_of_words(document, words):
    document_set = set(document)
    intersection = document_set.intersection(words)
    return dict([(word, (word in intersection)) for word in words])


def bag_of_words_freq(document, words):
    bag = dict((word, 0) for word in words)

    for word in document:
        if word in words:
            bag[word] += 1

    return bag


def get_dataset(K, feature_extractor):
    train_featuresets = list()  # list of pairs (featureset, category)
    test_featuresets = list()

    for category in reuters.categories():
        for fileid in reuters.fileids(categories=category):
            featureset = feature_extractor(reuters.words(fileids=[fileid]), most_frequent_words[:K])
            if fileid[:4] == 'test':
                test_featuresets.append((featureset, category))
            else:
                train_featuresets.append((featureset, category))

    return train_featuresets, test_featuresets


train_featuresets, test_featuresets = get_dataset(1000, bag_of_words)

from nltk.classify.naivebayes import NaiveBayesClassifier
import time
from joblib import Parallel, delayed  # For classification is a long task, and the deadline is near.
import multiprocessing

# num_cores = multiprocessing.cpu_count()
start = time.time()

classifer = NaiveBayesClassifier.train(train_featuresets)
# print "finished training, now Calculating on %d cores" % num_cores

def classify(item):
    featureset, tag = item
    return 1 if classifer.classify(featureset) == tag else 0


classifer.classify(test_featuresets[0][0])

# binary_list = Parallel(n_jobs=num_cores,  max_nbytes=1e3)(delayed(classify)(item) for item in test_featuresets)
#
# correct = sum(binary_list)
# correct = 0
# for featureset, tag in test_featuresets:
# if tag == classifer.classify(featureset):
# correct += 1

# print float(correct) / len(test_featuresets)

end = time.time()
print "finished in %fs" % (end - start)
