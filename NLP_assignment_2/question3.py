from threading import Thread
from nltk import NaiveBayesClassifier, DictionaryProbDist, sum_logs, SklearnClassifier, ELEProbDist, FreqDist
from nltk.corpus import reuters
import time
from joblib import Parallel, delayed  # For classification is a long task, and the deadline is near.
import multiprocessing
from nltk.corpus import reuters
from collections import defaultdict
#
# words = reuters.words()
# word_freq = defaultdict(int)
# for word in words:
#     word_freq[word] += 1
#
# sorted_words = sorted(set(words), key=word_freq.get, reverse=True)
# most_frequent_words = filter(lambda w: len(w) > 1, sorted_words)

def get_classifier(classifier, feature_extractor, K):
    train_featuresets, test_featuresets = get_dataset(K, feature_extractor)


def get_featuresets(feature_extractor, v, section):
    for category in reuters.categories():
        for fileid in reuters.fileids(categories=category):
            featureset = feature_extractor(reuters.words(fileids=[fileid]), v)
            if fileid[:len(section)] == section:
                yield (featureset, category)

def bag_of_words_freq(document, words):
    bag = dict((word, 0) for word in words)

    for word in document:
        if word in words:
            bag[word] += 1

    return bag

def bag_of_words(document, words):
    document_set = set(document)
    intersection = document_set.intersection(words)
    return dict([(word, (word in intersection)) for word in words])


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


    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):
        """
        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.
        """
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            label_freqdist[label] += 1
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname][fval] += 1
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                # Only add a None key when necessary, i.e. if there are
                # any samples with feature 'fname' missing.
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label,fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)

# apparently the following class is useless because Python's threads are a fraud.
class HTClassifierTester:
    def __init__(self, classifier, featuresets_generator, workers_count=multiprocessing.cpu_count()):
        self.classifier = classifier
        self.featuresets_generator = featuresets_generator
        self.thread_count = workers_count + 1  # for the queue generator
        self.featuresets_queue = multiprocessing.Queue(1000)
        self.done = False
        self.counter_lock = multiprocessing.Lock()

    def test(self):
        self.counter_lock.acquire()
        self.correct = 0
        self.total = 0
        self.counter_lock.release()

        running_threads = []
        for _ in range(self.thread_count):
            t = Thread(target=self.worker_run)
            t.start()
            running_threads.append(t)

        self.generator_run()

        for t in running_threads:
            t.join()

        return {"accuracy": float(self.correct / self.total)}

    def generator_run(self):
        for labled_featureset in self.featuresets_generator:
            self.featuresets_queue.put(labled_featureset)

        self.done = True

    def worker_run(self):
        while not self.done or not self.featuresets_queue.empty():
            featureset, real_label = self.featuresets_queue.get()
            label = self.classifier.classify(featureset)

            self.counter_lock.acquire()
            if label == real_label:
                self.correct += 1
            self.total += 1
            self.counter_lock.release()


if __name__ == "__main__":
    words = reuters.words()
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1

    sorted_words = sorted(set(words), key=word_freq.get, reverse=True)
    most_frequent_words = filter(lambda w: len(w) > 1, sorted_words)

    test_featuresets = get_featuresets(bag_of_words, most_frequent_words[:1000], "test")
    ls = list(test_featuresets)
    from nltk import NaiveBayesClassifier as NB
    # # the threaded classifier simply wraps a trained classifier and calls .classify() on different threads.
    #
    # train_featuresets = get_featuresets(bag_of_words_freq, most_frequent_words[:1000], "training")
    # test_featuresets = get_featuresets(bag_of_words_freq, most_frequent_words[:1000], "test")
    #
    # classifier = NB.train(train_featuresets)
    # print "finished training, now testing"
    # ht_classifier = HTClassifierTester(classifier, test_featuresets)
    # print ht_classifier.test()