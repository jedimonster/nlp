from itertools import izip
import math
from nltk.corpus import reuters

__author__ = 'itay'
import terminals


class TWSCalculator(object):
    def __init__(self, training_docs, docs_categories):
        self._EPSILON = 0.0001
        self.docs_categories = docs_categories
        self.categories = sorted(set(docs_categories))
        self.training_docs = training_docs
        self.idf_dict = {}
        self.ig_dict = {}
        self.rf_dict = {}

    def N(self):
        return len(self.training_docs)

    def bool(self, term, document):
        """

        :param term: AbstractTerm to check for present of
        :param document: document to check in
        :return:
        """
        return term.frequency(document) > 0

    def tf(self, term, document):
        return term.frequency(document)

    def idf(self, term):
        if term not in self.idf_dict:
            self.idf_dict[term] = self._idf(term)

        return self.idf_dict[term]

    def _df(self, term):
        df = sum(self.bool(term, document) for document in self.training_docs)
        return df

    def _idf(self, term):
        df = self._df(term)
        return math.log(float(self.N()) / max(df, 1), 2)

    def tf_idf(self, term, document):
        return self.tf(term, document) * self.idf(term)

    def prob_term_and_category(self, term_occures, category_filter):
        """

        :param term_occures: lambda (d) that returns true iff term occurs in d.
        :param category_filter:  lambda (c) true iff that's the category we're interested in.
        :return: probability for document be in said category and contain said term.
        """
        doc_cat = izip(self.training_docs, self.docs_categories)
        occurrences = sum(term_occures(document) for document, category in doc_cat if category_filter(category))
        return float(occurrences) / self.N()

    # def prob_term_category(self, term, category_filter):
    # doc_cat = izip(self.training_docs, self.docs_categories)

    def ig(self, term, category):
        if (term, category) not in self.ig_dict:
            self.ig_dict[term, category] = self._ig(term, category)

        return self.ig_dict[term, category]

    def _ig_inner_sum(self, p_t_c, p_c, p_t):
        # p_t_c may be 0:
        if p_t_c == 0:
            p_t_c = self._EPSILON

        return math.fabs(p_t_c * math.log(p_t_c / (p_t * p_c), 2))

    def _ig(self, term, category):
        # We're looking at the formula from http://dl.acm.org/citation.cfm?doid=952532.952688
        # calculate each of the 4 possibilities for (t, c) - (t,c), (t, c'), (t', c), (t', c')
        acc = 0
        category_equal = lambda c: c == category
        category_complements = lambda c: c != category
        term_occurs = lambda d: term.frequency(d) > 0
        term_complements = lambda d: term.frequency(d) == 0

        # (t, c):
        p = self.prob_term_and_category(term_occurs, category_equal)
        p_t = float(self._df(term)) / self.N()
        p_c = float(sum((category_equal(c) for c in self.docs_categories))) / self.N()
        acc += self._ig_inner_sum(p, p_c, p_t)
        # print category, "p(t,c) = ", p

        # (t, c'):
        p = self.prob_term_and_category(term_occurs, category_complements)
        p_t = float(self._df(term)) / self.N()
        p_c = float(sum((category_complements(c) for c in self.docs_categories))) / self.N()
        acc += self._ig_inner_sum(p, p_c, p_t)
        # print category, "p(t,c') = ", p

        # (t', c):
        p = self.prob_term_and_category(term_complements, category_equal)
        p_t = 1 - float(self._df(term)) / self.N()
        p_c = float(sum((category_equal(c) for c in self.docs_categories))) / self.N()
        acc += self._ig_inner_sum(p, p_c, p_t)

        # (t', c'):
        p = self.prob_term_and_category(term_complements, category_equal)
        p_t = 1 - float(self._df(term)) / self.N()
        p_c = float(sum((category_complements(c) for c in self.docs_categories))) / self.N()
        acc += self._ig_inner_sum(p, p_c, p_t)

        return acc

    def tf_ig(self, term, document):
        cat_information_gain = {}
        # Calculate  ig for every category
        for cat in self.categories:
            ig = self.ig(term, cat)
            cat_information_gain[cat] = ig
        # Calculate weighted ig for every category
        for cat in cat_information_gain:
            p_c = self.docs_categories.count(cat)/float(self.N())
            cat_information_gain[cat] = float(cat_information_gain[cat])*p_c

        weighted_ig = sum(cat_information_gain.values())

        return self.tf(term, document)*weighted_ig

    def rf(self, term, category):
        if (term, category) not in self.rf_dict:
            self.rf_dict[term, category] = self._rf(term, category)

        return self.rf_dict[term, category]

    def _rf(self, term, category):
        doc_cat = izip(self.training_docs, self.docs_categories)
        positive_documents = [x[0] for x in doc_cat if x[1] == category]
        negative_documents = [x[0] for x in doc_cat if x[1] != category]
        a, b, c, d = 0, 0, 0, 0
        # a - number of documents in the positive category which contain this term
        # b - number of documents in the positive category which do not contain this term
        # c - number of documents in the negative category which contain this term
        # d - number of documents in the negative category which do not contain this term
        for item in positive_documents:
            if term.frequency(item) >= 1:
                a += 1
            else:
                b += 1
        for item in negative_documents:
            if term.frequency(item) >= 1:
                c += 1
            else:
                d += 1
        rf = math.log(2 + (float(a)/max(float(1), float(c))))

        return rf

    def tf_rf(self, term, document):
        res = []
        # Calculate  rf for every category
        for cat in self.categories:
            rf = self.rf(term, cat)
            p_c = self.docs_categories.count(cat)/float(self.N())
            res.append(float(rf)*p_c)

        weighted_rf = sum(res)
        return self.tf(term, document)*weighted_rf






if __name__ == '__main__':
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids())
    documents = [sum(reuters.sents(fid), []) for fid in training_fileids]
    doc = documents[0]
    term = terminals.WordTerm("in")
    docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]
    print docs_categories
    print doc
    fe = TWSCalculator(documents, docs_categories)

    print "tf =", fe.tf(term, doc), "idf =", fe.idf(term), "tf-idf =", fe.tf_idf(term, doc)
    term = terminals.WordTerm("in")
    print "tf_ig: ", fe.tf_ig(term, doc), " tf_rf: ", fe.tf_rf(term, doc)
    print "IG for term 'bank':"
    for c in sorted(set(docs_categories)):
        print c, ":", fe.ig(terminals.WordTerm("bank"), c)
    print "RF for term 'bank':"
    for c in sorted(set(docs_categories)):
        print c, ":", fe.rf(terminals.WordTerm("bank"), c)