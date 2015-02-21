from collections import defaultdict
from nltk.corpus import reuters

__author__ = 'itay'
training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids())

training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
categories = set(training_docs_categories)

counts = defaultdict(int)
for c in training_docs_categories:
    counts[c] += 1

print sorted(counts.items(), key=lambda x: x[1])