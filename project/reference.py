from nltk.corpus import reuters
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from features import TWSCalculator
from terminals import get_document_objects, WordTermExtractor

__author__ = 'itay'
if __name__ == '__main__':
    cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply', 'ship',
                                 'sugar']  # top 9
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(cats_limiter))

    training_documents = [" ".join(sum(reuters.sents(fid), [])) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    map(lambda x: x.lower, training_documents)

    training_documents_objects = get_document_objects(training_documents, training_docs_categories)

    # tws_calculator = TWSCalculator(training_documents_objects, training_docs_categories)
    # word_term_extractor = WordTermExtractor(training_documents_objects, tws_calculator)
    #
    # top_terms = word_term_extractor.top_common_words(500)
    print training_documents[0]
    print training_fileids

    vectorizer = TfidfVectorizer(input='content')
    feature_matrix = vectorizer.fit_transform(training_documents)

    classifier = OneVsRestClassifier(MultinomialNB())
    classifier.fit(feature_matrix, training_docs_categories)


    # Test:
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids(cats_limiter))
    test_documents = [" ".join(sum(reuters.sents(fid), [])) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    map(lambda x: x.lower, test_documents)

    test_features = vectorizer.transform(test_documents)

    predictions = classifier.predict(test_features)

    metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='macro')

    print "Metrics (percision, recall, fmeasure):", metrics

    accuracy = accuracy_score(test_docs_categories, predictions)

    print "Accuracy:", accuracy