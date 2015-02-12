from itertools import izip
import math
from nltk.corpus import reuters
import sys
from parameters import ProjectParams

__author__ = 'itay'
import terminals


class TWSCalculator(object):
    def __init__(self, training_docs, docs_categories):
        self._EPSILON = 10 ** -20
        self.docs_categories = docs_categories
        self.categories = sorted(set(docs_categories))
        self.training_docs = training_docs
        self.idf_dict = {}
        self.ig_dict = {}
        self.rf_dict = {}
        self.chi_dict = {}
        self.logger = ProjectParams.logger
        self.df_dict = {}
        self.term_category_dict = {}

        self._pc = dict((cat, float(self.docs_categories.count(cat)) / self.N()) for cat in self.categories)

        print self._pc
        self.logger.debug("Created TWS Calculator with %d documents and %d categories", len(training_docs),
                          len(
                              self.categories))

    def N(self):
        return len(self.training_docs)

    def bool(self, term, document):
        """

        :param term: AbstractTerm to check for present of
        :param document: document to check in
        :return:
        """
        return term.frequency(document) > 0

    def tf_idf(self, term, document):
        return self.tf(term, document) * self.idf(term)

    def tf(self, term, document):
        return term.frequency(document)

    def idf(self, term):
        if term not in self.idf_dict:
            self.idf_dict[term] = self._idf(term)

        return self.idf_dict[term]

    def tf_ig(self, term, document):
        cat_information_gain = {}
        # Calculate  ig for every category
        for cat in self.categories:
            ig = self.ig(term, document)
            cat_information_gain[cat] = ig
        # Calculate weighted ig for every category
        for cat in cat_information_gain:
            p_c = self.docs_categories.count(cat) / float(self.N())
            cat_information_gain[cat] = float(cat_information_gain[cat]) * p_c

        weighted_ig = sum(cat_information_gain.values())

        return self.tf(term, document) * weighted_ig

    def _prob_term_and_category_OLD_DELETE_ME(self, term_occures, category_filter):
        """
        :param term_occures: lambda (d) that returns true iff term occurs in d.
        :param category_filter:  lambda (c) true iff that's the category we're interested in.
        :return: probability for document be in said category and contain said term.
        """

        doc_cat = izip(self.training_docs, self.docs_categories)
        occurrences = sum(term_occures(document) for document, category in doc_cat if category_filter(category))
        return float(occurrences) / self.N()

    def _df(self, term):
        if term not in self.df_dict:
            self.df_dict[term] = sum(self.bool(term, document) for document in self.training_docs)

        return self.df_dict[term]

    # def prob_term_category(self, term, category_filter):
    # doc_cat = izip(self.training_docs, self.docs_categories)

    def _idf(self, term):
        df = self._df(term)
        return math.log(float(self.N()) / max(df, 1), 2)

    def tf_chi(self, term, doc):
        weighed_chi = 0

        for cat in self.categories:
            chi = self.chi_square(term, doc)
            p_c = self.docs_categories.count(cat) / float(self.N())
            weighed_chi += chi * p_c

        return self.tf(term, doc) * weighed_chi

    def max_ig(self, term):
        max((self._ig(term, c) for c in self.categories))

    def ig(self, term, document):
        category = document.category
        if (term, category) not in self.ig_dict:
            self.ig_dict[term, category] = self._ig(term, category)

        return self.ig_dict[term, category]

    def chi_square(self, term, document):
        category = document.category
        if (term, category) not in self.chi_dict:
            self.chi_dict[term, category] = self._chi_square(term, category)

        return self.chi_dict[term, category]

    def _ig_inner_sum(self, p_t_c, p_c, p_t):
        # p_t_c may be 0:
        if p_t_c == 0:
            p_t_c = self._EPSILON
        return p_t_c * math.log(p_t_c / (p_t * p_c), 2)

    def _prob_term_not_category(self, term, category):
        return sum(self._prob_term_and_category(term, c) for c in self.categories if c != category)

    def _ig(self, term, category):
        # We're looking at the formula from http://dl.acm.org/citation.cfm?doid=952532.952688
        # calculate each of the 4 possibilities for (t, c) - (t,c), (t, c'), (t', c), (t', c')
        acc = 0
        category_equal = lambda c: c == category
        category_complements = lambda c: c != category
        term_occurs = lambda d: term.frequency(d) > 0
        term_complements = lambda d: term.frequency(d) == 0

        # (t, c):
        p = self._prob_term_and_category(term, category)
        p_t = float(self._df(term)) / self.N()
        p_c = self._pc[category]
        acc += self._ig_inner_sum(p, p_c, p_t)
        # print category, "p(t,c) = ", p

        # (t, c'):
        p = self._prob_term_not_category(term, category)
        p_t = float(self._df(term)) / self.N()
        p_c = sum([self._pc[c] for c in self.categories if category_complements(c)])
        # p_c = float(sum((category_complements(c) for c in self.docs_categories))) / self.N()
        acc += self._ig_inner_sum(p, p_c, p_t)
        # print category, "p(t,c') = ", p

        # (t', c):
        p_c = self._pc[category]
        p = p_c - self._prob_term_and_category(term, category)
        p_t = 1 - float(self._df(term)) / self.N()
        acc += self._ig_inner_sum(p, p_c, p_t)

        # (t', c'):
        p = (1 - p_c) - self._prob_term_not_category(term, category)
        p_t = 1 - float(self._df(term)) / self.N()
        p_c = sum([self._pc[c] for c in self.categories if category_complements(c)])
        acc += self._ig_inner_sum(p, p_c, p_t)

        return acc

    def _prob_term_and_category(self, term, category):
        if (term, category) not in self.term_category_dict:
            p = float(sum(self.bool(term, d) for d in self.training_docs if d.category == category)) / self.N()
            self.term_category_dict[term, category] = p

        return self.term_category_dict[term, category]

    def _chi_square(self, term, category):
        # category_equal = lambda c: c == category
        # category_complements = lambda c: c != category
        # term_occurs = lambda d: term.frequency(d) > 0
        # term_complements = lambda d: term.frequency(d) == 0

        p_c = self._pc[category]
        p_termC_catC = (1 - p_c) - self._prob_term_not_category(term, category)
        numerator = self._prob_term_and_category(term, category) * p_termC_catC

        numerator -= self._prob_term_not_category(term, category) * (1 - self._prob_term_and_category(term, category))

        # numerator = self._prob_term_and_category(term, category) * self._prob_term_and_category_OLD_DELETE_ME(
        # term_complements, category_complements)
        # numerator -= sum(self._prob_term_and_category(term, c) for c in self.categories if
        # c != category) * self._prob_term_and_category_OLD_DELETE_ME(
        # term_complements, category_equal)

        numerator = math.pow(numerator, 2)

        p_t = float(self._df(term)) / self.N()
        p_not_t = 1 - p_t
        p_c = self._pc[category]
        p_not_c = 1 - p_c

        denominator = p_t * p_c * p_not_t * p_not_c

        # not sure what to do if denominator is zero
        if denominator == 0:
            return 0

        result = float(numerator) / denominator

        return result


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
        rf = math.log(2 + (float(a) / max(float(1), float(c))))

        return rf

    def tf_rf(self, term, document):
        res = []
        # Calculate  rf for every category
        for cat in self.categories:
            rf = self.rf(term, cat)
            p_c = self.docs_categories.count(cat) / float(self.N())
            res.append(float(rf) * p_c)

        weighted_rf = sum(res)
        return self.tf(term, document) * weighted_rf

    def terminals(self, term, document):
        return (
            self.bool(term, document), self.tf(term, document), self.tf_idf(term, document), self.tf_ig(term, document),
            self.tf_chi(term, document), self.tf_rf(term, document))


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

    print 'TF-CHI: ', fe.tf_chi(term, doc)
    print 'TF-CHI: ', fe.tf_chi(term, doc)
    print 'TF-IG: ', fe.tf_ig(term, doc)
    print 'TF-IG: ', fe.tf_ig(term, doc)

    print "tf_ig: ", fe.tf_ig(term, doc), " tf_rf: ", fe.tf_rf(term, doc)
    for c in sorted(set(docs_categories)):
        print c, ": IG", fe.ig(terminals.WordTerm("bank"), c)
        print c, ": CHI", fe.ig(terminals.WordTerm("bank"), c)

    print "RF for term 'bank':"
    for c in sorted(set(docs_categories)):
        print c, ":", fe.rf(terminals.WordTerm("bank"), c)

