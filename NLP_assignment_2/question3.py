from nltk import NaiveBayesClassifier, DictionaryProbDist, sum_logs, SklearnClassifier
from nltk.corpus import reuters
import time
from joblib import Parallel, delayed  # For classification is a long task, and the deadline is near.
import multiprocessing


def get_classifier(classifier, feature_extractor, K):
    train_featuresets, test_featuresets = get_dataset(K, feature_extractor)


def get_featuresets(K, feature_extractor, v, section):
    train_featuresets = list()  # list of pairs (featureset, category)
    test_featuresets = list()

    for category in reuters.categories():
        for fileid in reuters.fileids(categories=category):
            featureset = feature_extractor(reuters.words(fileids=[fileid]), v)
            if fileid[:4] == section:
                yield (featureset, category)


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


class HTClassifierTester:
    def __init__(self, classifier, featuresets_generator, workers_count=multiprocessing.cpu_count()):
        self.classifier = classifier
        self.featuresets_generator = featuresets_generator
        self.thread_count = workers_count + 1  # for the queue generator
        self.featuresets_queue = multiprocessing.Queue(100)
        self.done = False
        self.counter_lock = multiprocessing.Lock()

    def test(self):
        self.counter_lock.acquire()
        self.correct = 0
        self.total = 0
        self.counter_lock.release()

        for _ in range(self.thread_count):



        return {"accuracy": float(self.correct / self.total)}

    def generator_run(self):
        for labled_featureset in self.featuresets_generator:
            self.featuresets_queue.put(labled_featureset)

    def worker_run(self):
        while not self.done or not self.featuresets_queue.empty():
            pass


if __name__ == "__main__":
    # we create two classes, were the only feature linearly separates them:
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


