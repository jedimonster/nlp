from collections import defaultdict
from nltk.corpus import reuters
from tws_gp import to_lower

__author__ = 'itay'
training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids())
training_documents = [map(to_lower, sum(reuters.sents(fid), [])) for fid in training_fileids]

training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
categories = set(training_docs_categories)

counts = defaultdict(int)
for c in training_docs_categories:
    counts[c] += 1

counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print [c for c, count in counts[:10]]
print len(counts)


