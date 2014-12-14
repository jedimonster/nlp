"""
Question 2
"""
import string
from nltk.corpus import conll2002
import collections
from nltk.chunk.util import tree2conlltags
# tree2conlltags(chuncked_sents) => (word,pos, iob)



def word_feature(tagged_word):
    return tagged_word[0]

def pos_feature(tagged_word):
    return tagged_word[1]

def is_number(tagged_word):
    return tagged_word[0].isdigit()

def is_contains_digit(tagged_word):
    return any([x.isdigit() for x in tagged_word[0]])

def contains_hyphen(tagged_word):
    return any([x == '-' for x in tagged_word[0]])

def punctuation(tagged_word):
    return tagged_word[0] in string.punctuation


def capitalized(tagged_word):
    return tagged_word[0][0].isupper()


def all_capitals(tagged_word):
    return all([x.isupper() for x in tagged_word[0]])

ORT_methods = [is_number, is_contains_digit, contains_hyphen, punctuation, capitalized, all_capitals]


def get_classes(categories):
    """
    input: list of categories
    output: dict of classes
    """
    res = {}
    data = []
    for category in categories:
        data = data + conll2002.iob_sents(category)

    class_index = 0
    for sent in data:
        for tagged_word in sent:
            word = tagged_word[2]
            # import pdb
            # pdb.set_trace()
            if word not in res:
                res[word] = class_index
                class_index += 1
    return res


def feature_creator(data):
    """
    :param data:
    :return: vector of all features in data
    """
    all_features = {'number', 'contains-digit', 'contains-hyphen', 'capitalized', 'all-capitals',
                    'URL', 'punctuation', 'regular'}
    for sent in data:
        for tagged_word in sent:
            word = tagged_word[0]
            all_features.add(word)
            pos = tagged_word[1]
            all_features.add(pos)
            suf1= tagged_word[0][-1]
            pref1 = tagged_word[0][0]
            all_features.add(suf1)
            all_features.add(pref1)
            if len(tagged_word[0]) > 1:
                suf2 = tagged_word[0][-2:len(tagged_word[0])]
                pref2 = tagged_word[0][0:2]
                all_features.add(suf2)
                all_features.add(pref2)
            if len(tagged_word[0]) > 2:
                suf3 = tagged_word[0][-3:len(tagged_word[0])]
                pref3 = tagged_word[0][0:3]
                all_features.add(suf3)
                all_features.add(pref3)
    print "number" in all_features
    # print all_features
    print len(all_features)
    numed_features = {x:index for x,index in zip(all_features, range(1, len(all_features)+1))}
    print len(numed_features.values())
    print 0 in numed_features.values()
    numed_features['undefined_feature'] = 0
    return numed_features

def feature_extraction(tagged_word):
    res = {}
    res['word'] = tagged_word[0]
    res['POS'] = tagged_word[1]
    suf1 = tagged_word[0][-1]

    res['suf1'] = suf1

    pref1 = tagged_word[0][0]
    res['pref1'] = pref1

    if len(tagged_word[0])>1:
        suf2 = tagged_word[0][-2:len(tagged_word[0])]
        pref2 = tagged_word[0][0:2]
        res['suf2'] = suf2
        res['pref2'] = pref2

    else:
        res['suf2'] = 'undef'
        res['pref2'] = 'undef'
    if len(tagged_word[0])>2:
        suf3 = tagged_word[0][-3:len(tagged_word[0])]
        pref3 = tagged_word[0][0:3]
        res['suf3'] = suf3
        res['pref3'] = pref3
    else:
        res['suf3'] = 'undef'
        res['pref3'] = 'undef'
    for method in ORT_methods:
        if method(tagged_word):
            res[method.__name__] = 1
        else:
            res[method.__name__] = 0
    return res

def prepear_train(train_data):
    res = []
    for sent in train_data:
        for tagged_word in sent:
            features = feature_extraction(tagged_word)
            res.append((features, tagged_word[2]))
    return res


if __name__ == "__main__":
    etr = conll2002.chunked_sents('esp.train') # In Spanish
    eta = conll2002.chunked_sents('esp.testa') # In Spanish
    etb = conll2002.chunked_sents('esp.testb') # In Spanish

    dtr = conll2002.chunked_sents('ned.train') # In Dutch
    dta = conll2002.chunked_sents('ned.testa') # In Dutch
    dtb = conll2002.chunked_sents('ned.testb') # In Dutch
    # conll2002.iob_sents()
    from nltk.classify import SklearnClassifier
    train_data = [({"a": 4, "b": 1, "c": 0}, "ham"),
                  ({"a": 5, "b": 2, "c": 1}, "ham"),
                  ({"a": 0, "b": 3, "c": 4}, "spam"),
                  ({"a": 5, "b": 1, "c": 1}, "ham"),
                  ({"a": 1, "b": 4, "c": 3}, "spam")]
    test_data = [{"a": 3, "b": 2, "c": 1},
                 {"a": 0, "b": 3, "c": 7}]
    from sklearn.svm import SVC
    classif = SklearnClassifier(SVC(), sparse=False).train(train_data)
    print classif.classify_many(test_data)
    # print get_classes(['esp.train', 'esp.testa', 'esp.testb'])
    train_data = conll2002.iob_sents('ned.train')

    train_data = prepear_train(train_data)

    print len(train_data)
    print "first few of trian:"
    print train_data[0:10]
    # for tup in train_data:
    #     if tup[1] != 'O':
    #         print "&&&&&&&&"
    #         print tup
    #         print "&&&&&&&&"
    test_data = conll2002.iob_sents('ned.testa')
    c_range = range(1000, 50000, 1000)
    test = []
    for sent in test_data:
        for tagged_word in sent:
            test.append(feature_extraction(tagged_word))
    for c in c_range:
        classif = SklearnClassifier(SVC(verbose=False, C=0.5), sparse=False).train(train_data[0:10000])

        alg_res = classif.classify_many(test[0:10000])
        #print alg_res
        test_classes = []
        for sent in test_data:
            for tagged_word in sent:
                test_classes.append(tagged_word[2])
        #print test_classes
        error_count = 0
        correct_count = 0
        not_o_errors = 0
        for item in zip(alg_res, test_classes[0:10000]):
            if item[0] == item[1]:
                correct_count += 1
            else:
                error_count += 1
            if item[1] != 'O' and item[0] != item[1]:
                not_o_errors += 1
        print "C is: ", c
        print correct_count
        print error_count
        print not_o_errors
