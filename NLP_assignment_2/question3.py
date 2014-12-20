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
# word_freq[word] += 1
#
# sorted_words = sorted(set(words), key=word_freq.get, reverse=True)
# most_frequent_words = filter(lambda w: len(w) > 1, sorted_words)

def get_classifier(classifier, feature_extractor, K):
    train_featuresets, test_featuresets = get_dataset(K, feature_extractor)


def get_featuresets(feature_extractor, v, section):
    for fileid in reuters.fileids():
        featureset = feature_extractor(reuters.words(fileids=[fileid]), v)
        category = reuters.categories(fileid)[0]
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
    def __init__(self, label_probdist, feature_probdist, feature_comp_probdist):
        """
        :param label_probdist: P(label), the probability distribution
            over labels.  It is expressed as a ``ProbDistI`` whose
            samples are labels.  I.e., P(label) =
            ``label_probdist.prob(label)``.

        :param feature_probdist: P(fname=fval|label), the probability
            distribution for feature values, given labels.  It is
            expressed as a dictionary whose keys are ``(label, fname)``
            pairs and whose values are ``ProbDistI`` objects over feature
            values.  I.e., P(fname=fval|label) =
            ``feature_probdist[label,fname].prob(fval)``.  If a given
            ``(label,fname)`` is not a key in ``feature_probdist``, then
            it is assumed that the corresponding P(fname=fval|label)
            is 0 for all values of ``fval``.
        """
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._feature_comp_probdist = feature_comp_probdist
        self._labels = list(label_probdist.samples())


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
                feature_comp_probs = self._feature_comp_probdist[label, fname]
                logprob[label] += feature_probs.logprob(fval)
                logprob[label] -= feature_comp_probs.logprob(fval)
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

        # Create the compliment of the above freqdist for each [label, fname]
        feature_comp_freqdist = defaultdict(FreqDist)
        labels = label_freqdist.keys()
        for label in labels:
            other_labels = filter(lambda x: x != label, labels)

            # for each fval, sum the freqdists of other labels:
            for (other_label, fname), other_freqdist in feature_freqdist.iteritems():
                if label != other_label:
                    for fvalue, fcount in other_freqdist.iteritems():
                        feature_comp_freqdist[label, fname][fvalue] += fcount



        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        feature_comp_probdist = {}

        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        for ((label, fname), freqdist) in feature_comp_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_comp_probdist[label, fname] = probdist

        return OneVsAllNaiveBayesClassifier(label_probdist, feature_probdist, feature_comp_probdist)


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
    pass
    import sys
    # from question3 import OneVsAllNaiveBayesClassifier

    words = reuters.words()
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1

    sorted_words = sorted(set(words), key=word_freq.get, reverse=True)
    most_frequent_words = filter(lambda w: len(w) > 1, sorted_words)

    TP = defaultdict(int)
    FP = defaultdict(int)
    TN = defaultdict(int)
    FN = defaultdict(int)

    start = time.time()

    train_featuresets = get_featuresets(bag_of_words, most_frequent_words[:2000], "training")
    test_featuresets = get_featuresets(bag_of_words, most_frequent_words[:2000], "test")
    # finance_doc = "money money money money money money money money money money".split(" ")  # 10 occurences of money
    # cs_doc = "bit bit bit bit bit bit bit bit bit bit".split(" ")  # 10 occurency of bit
    # finance_test = "money money money money money money money money money money money".split(" ")  # 11 occurences
    # cs_test = "bit bit bit bit bit bit bit bit bit bit bit".split(" ")  # 11 occurences
    #
    # v = {"bit", "money"}
    #
    # train_featuresets = [(bag_of_words_freq(finance_doc, v), "finance"), (bag_of_words_freq(cs_doc, v), "CS")]
    # test_featuresets = [(bag_of_words_freq(finance_test, v), "finance"), (bag_of_words_freq(cs_test, v), "CS")]

    classifier = OneVsAllNaiveBayesClassifier.train(train_featuresets)

    def classify_one(labeled_featureset):
        featureset, real_label = labeled_featureset
        return (classifier.classify(featureset), real_label)


    print "Trained, now classifying"
    sys.stdout.flush()

    tests_refs = Parallel(n_jobs=4, verbose=0)(delayed(classify_one)(item) for item in test_featuresets)

    end = time.time()
    print "finished in %fs" % (end - start)

    N = 0  # count because the generator dies with the loop
    for test, ref in tests_refs:
        N += 1
        if test == ref:
            TP[ref] += 1
        else:
            FP[test] += 1
            FN[ref] += 1
    # for label in labels - set([test, ref]):
    #         TN[label] += 1 # we add a TN to all labels except test (that one is either TP or FP) and ref (same)

    print "accuracy: %f\n\n" % (float(sum(TP.values())) / N)
    #
    # print train_featuresets
    #
    # classifier = NaiveBayesClassifier.train(train_featuresets)
    #
    # for featureset, label in test_featuresets:
    #     print "classified %s document as %s" % (label, classifier.classify(featureset))
    #
    # test_featuresets = get_featuresets(bag_of_words, most_frequent_words[:1000], "test")
    # ls = list(test_featuresets)
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