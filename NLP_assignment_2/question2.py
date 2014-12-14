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

def feature_extraction(tagged_word, numed_features):
    res = {}
    if tagged_word[0] in numed_features:
        res.update({numed_features[tagged_word[0]]: 1})
    else:
        res.update({0: 1})
    try:
        tagged_word[1]
    except:
        print tagged_word

    if tagged_word[1] in numed_features:
        res.update({numed_features[tagged_word[1]]: 1})
    else:
        res.update({0: 1})
    suf1= tagged_word[0][-1]
    if suf1 in numed_features:
        res.update({numed_features[suf1]: 1})
    else:
        res.update({0: 1})
    pref1 = tagged_word[0][0]
    if pref1 in numed_features:
        res.update({numed_features[pref1]: 1})
    else:
        res.update({0: 1})
    if len(tagged_word[0])>1:
        suf2 = tagged_word[0][-2:len(tagged_word[0])]
        pref2 = tagged_word[0][0:2]
        if suf2 in numed_features:
            res.update({numed_features[suf2]: 1})
        else:
            res.update({0: 1})
        if pref2 in numed_features:
            res.update({numed_features[pref2]: 1})
        else:
            res.update({0: 1})
    if len(tagged_word[0])>2:
        suf3 = tagged_word[0][-3:len(tagged_word[0])]
        pref3 = tagged_word[0][0:3]
        if suf3 in numed_features:
            res.update({numed_features[suf3]: 1})
        else:
            res.update({0: 1})
        if pref3 in numed_features:
            res.update({numed_features[pref3]: 1})
        else:
            res.update({0: 1})
    return res

def prepear_train(train_data, numed_features):
    res = []
    for sent in train_data:
        for tagged_word in sent:
            features = feature_extraction(tagged_word, numed_features)
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
    numed_features = feature_creator(train_data)
    train_data = prepear_train(train_data, numed_features)
    print len(train_data)
    classif = SklearnClassifier(SVC(), sparse=False).train(train_data[0:10000])
    test_data = conll2002.iob_sents('ned.testa')
    print len(test_data)
    test_data = test_data[0:10]
    print test_data
    test = []
    for sent in test_data:
        for tagged_word in sent:
            test.append(feature_extraction(tagged_word, numed_features))
    print test
    for item in test:
        print classif.classify_many(item)
