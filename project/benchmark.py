from time import clock
import timeit
from nltk.corpus import reuters
from sklearn.svm import SVC
from features import TWSCalculator
from fitness import TWSFitnessCalculator, FeatureExtractor
from terminals import WordTermExtractor, get_document_objects

__author__ = 'itay'


def start_clock(str=""):
    global start

    print str
    start = clock()


def end_clock():
    global start
    print clock() - start
    print " -- "


if __name__ == '__main__':
    start_clock("initializing data sets")
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids())
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids())

    training_documents = [sum(reuters.sents(fid), []) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    training_documents = get_document_objects(training_documents, training_docs_categories)

    test_documents = [sum(reuters.sents(fid), []) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    end_clock()

    start_clock("creating TWS calculator")
    tws_calculator = TWSCalculator(training_documents, training_docs_categories)
    end_clock()

    word_term_extractor = WordTermExtractor(training_documents, tws_calculator)

    start_clock("getting top words by frequency")
    top_terms = word_term_extractor.top_common_words(2000)
    end_clock()

    start_clock("creating feature extractor")
    feature_extractor = FeatureExtractor(training_documents, tws_calculator, top_terms)
    end_clock()

    fitness_calculator = TWSFitnessCalculator(SVC(), zip(training_documents, training_docs_categories),
                                              feature_extractor)

    print tws_calculator.tf(top_terms[0], training_documents[0])

    start_clock("Calculating df of one term")
    tws_calculator._df(top_terms[0])
    end_clock()

    start_clock("Calculating df of one term 10^5")
    for i in range(10 ** 5):
        tws_calculator._df(top_terms[0])
    end_clock()

    start_clock("Calculating tf of one term")
    tws_calculator.tf(top_terms[0], training_documents[0])
    end_clock()

    start_clock("Calculating tf of one term x10000")
    for i in range(10000):
        tws_calculator.tf(top_terms[0], training_documents[0])
    end_clock()

    start_clock("Calculating tf-idf on one term")
    tws_calculator.tf_idf(top_terms[0], training_documents[0])
    end_clock()
    start_clock("Calculating tf-idf on SAME term")
    tws_calculator.tf_idf(top_terms[0], training_documents[0])
    end_clock()

    start_clock("Calculating chi on one term")
    tws_calculator.chi_square(top_terms[0], training_documents[0])
    end_clock()

    # start_clock("Calculating IG on one term")
    # tws_calculator.ig(top_terms[0], training_documents[0])
    # end_clock()

