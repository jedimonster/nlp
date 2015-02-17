from nltk.corpus import reuters
from terminals import get_document_objects

if __name__ == '__main__':
    cats_limiter = categories = ['gold', 'money-fx', 'trade']
    # cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply', 'ship',
    # 'sugar']  # top 9
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(cats_limiter))

    training_documents = [sum(reuters.sents(fid), []) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    training_documents = get_document_objects(training_documents, training_docs_categories)

    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids(cats_limiter))
    test_documents = [sum(reuters.sents(fid), []) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    test_documents = get_document_objects(test_documents, test_docs_categories)

    print training_documents[0].doc
    print test_documents[0].doc