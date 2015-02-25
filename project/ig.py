from collections import defaultdict
from nltk.corpus import reuters
from features import TWSCalculator
from readers import NewsgroupsReader
from terminals import get_document_objects, WordTermExtractor
from tws_gp import to_lower

__author__ = 'itay'

cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply', 'ship']  # top 8

# training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
#                                     reuters.fileids(cats_limiter))
# training_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in training_fileids]
# training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
# training_documents = get_document_objects(training_documents, training_docs_categories)

training_documents = NewsgroupsReader(True).get_training()
training_docs_categories = [d.category for d in training_documents]


# categories = set(training_docs_categories)

tws_calculator = TWSCalculator(training_documents, training_docs_categories)
word_term_extractor = WordTermExtractor(training_documents, tws_calculator)

top = word_term_extractor.max_ig_per_category(500)

print "500:"
print [w._word for w in top]
