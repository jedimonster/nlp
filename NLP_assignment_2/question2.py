"""
Question 2
"""
import string
from nltk.corpus import conll2002
# import numpy as np
# import collections
# from nltk.chunk.util import tree2conlltags
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


def URL(tagged_word):
    return "//" in tagged_word[0]

def few_upper(tagged_word):
    res = [x for x in tagged_word[0] if x.isupper()]
    return len(res) > 1

# Additional features



ORT_methods = [is_number, is_contains_digit, contains_hyphen, punctuation, capitalized, all_capitals, URL, few_upper]
undefined_features = {"pref1_undef", "pref2_undef", "pref3_undef", "suf1_undef", "suf2_undef", "suf3_undef", "undefined_pos", "regular_feature",
                      "undefined_word",  'first_word', 'last_word', 'not_first_word', 'len_of_word'}

def is_regular(tagged_word):
    for method in ORT_methods:
        if method(tagged_word):
            return False
    return True

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
    all_features = undefined_features
    for method in ORT_methods:
        all_features.add(method.__name__)

    for sent in data:
        for tagged_word in sent:
            word = tagged_word[0]
            all_features.add('word_'+word)
            pos = tagged_word[1]
            all_features.add('pos_'+pos)
            suf1 = tagged_word[0][-1]
            pref1 = tagged_word[0][0]
            all_features.add('suf1_'+suf1)
            all_features.add('pref1_'+pref1)
            if len(tagged_word[0]) > 1:
                suf2 = tagged_word[0][-2:len(tagged_word[0])]
                pref2 = tagged_word[0][0:2]
                all_features.add('suf2_'+suf2)
                all_features.add('pref2_'+pref2)
            if len(tagged_word[0]) > 2:
                suf3 = tagged_word[0][-3:len(tagged_word[0])]
                pref3 = tagged_word[0][0:3]
                all_features.add('suf3_'+suf3)
                all_features.add('pref3_'+pref3)
    print "is_number" in all_features
    # print all_features
    print len(all_features)
    numed_features = {x: index for x, index in zip(all_features, range(0, len(all_features)))}
    print len(numed_features.values())
    print 0 in numed_features.values()
    return numed_features


def feature_extraction(tagged_word, numed_features, sent):
    # undefined_features = {"pref1", "pref2", "pref3", "suf1", "suf2", "suf3", "undefined_pos", "regular_feature",
    #                   "undefined_word"}
    res = {}
    res.update({numed_features['len_of_word']: len(tagged_word[0])})
    for method in ORT_methods:
        if method(tagged_word):
            res.update({numed_features[method.__name__]: 1})
    if sent[0] == tagged_word:
        res.update({numed_features['first_word']: 1})
    else:
        res.update({numed_features['not_first_word']: 1})
    if sent[-1] == tagged_word:
        res.update({numed_features['last_word']: 1})
    if 'word_'+tagged_word[0] in numed_features:
        res.update({numed_features['word_'+tagged_word[0]]: 1})
    else:
        res.update({numed_features['undefined_word']: 1})

    if 'pos_'+tagged_word[1] in numed_features:
        res.update({numed_features['pos_'+tagged_word[1]]: 1})
    else:
        res.update({numed_features['undefined_pos']: 1})

    suf1= tagged_word[0][-1]

    if 'suf1_'+suf1 in numed_features:
        res.update({numed_features['suf1_'+suf1]: 1})
    else:
        res.update({numed_features['suf1_undef']: 1})
    pref1 = tagged_word[0][0]
    if 'pref1_'+pref1 in numed_features:
        res.update({numed_features['pref1_'+pref1]: 1})
    else:
        res.update({numed_features['pref1_undef']: 1})

    if len(tagged_word[0]) > 1:
        suf2 = tagged_word[0][-2:len(tagged_word[0])]
        pref2 = tagged_word[0][0:2]
        if 'suf2_'+suf2 in numed_features:
            res.update({numed_features['suf2_'+suf2]: 1})
        else:
            res.update({numed_features['suf2_undef']: 1})
        if 'pref2_'+pref2 in numed_features:
            res.update({numed_features['pref2_'+pref2]: 1})
        else:
            res.update({numed_features['pref2_undef']: 1})
    if len(tagged_word[0]) > 2:
        suf3 = tagged_word[0][-3:len(tagged_word[0])]
        pref3 = tagged_word[0][0:3]
        if 'suf3_'+suf3 in numed_features:
            res.update({numed_features['suf3_'+suf3]: 1})
        else:
            res.update({numed_features['suf3_undef']: 1})
        if 'pref3_'+pref3 in numed_features:
            res.update({numed_features['pref3_'+pref3]: 1})
        else:
            res.update({numed_features['pref3_undef']: 1})
    return res

def prepear_train(train_data, numed_features, classes):
    res = []
    for sent in train_data:
        for tagged_word in sent:
            features = feature_extraction(tagged_word, numed_features, sent)
            res.append((features, classes[tagged_word[2]]))
            # import pdb
            # pdb.set_trace()
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
    from sklearn.svm import LinearSVC
    classif = SklearnClassifier(SVC(), sparse=False).train(train_data)
    print classif.classify_many(test_data)
    # print get_classes(['esp.train', 'esp.testa', 'esp.testb'])
    classes = get_classes(['esp.train', 'esp.testa', 'esp.testb'])
    print "classes :", classes
    train_data = conll2002.iob_sents('ned.train')
    numed_features = feature_creator(train_data)
    train_data = prepear_train(train_data, numed_features, classes)
    print len(train_data)
    train_data = train_data[0:30000]

    # from sklearn.feature_extraction import DictVectorizer
    # v = DictVectorizer(sparse=True)
    # list_of_features = []
    # for item in train_data:
    #     import pdb
    #     pdb.set_trace()
    #     list_of_features.append(item[0])
    # X = v.fit_transform(list_of_features)
    # print X
    c_range = range(-2, 20)
    for c in c_range:
        print "C is ", c
        classif = SklearnClassifier(SVC(C=2**c, verbose=False)).train(train_data)
        test_data = conll2002.iob_sents('ned.testa')
        test_data = test_data
        # print test_data
        tag_of_test = []
        test = []
        words = []
        not_o_tags = []
        for sent in test_data:
            for tagged_word in sent:
                tag_of_test.append(classes[tagged_word[2]])
                words.append(tagged_word[0])
                if tagged_word[2] != 'O':
                    not_o_tags.append(tagged_word[2])
                test.append(feature_extraction(tagged_word, numed_features, sent))
        # print test

        res = classif.classify_many(test)
        # print len(res)
        # print res
        # print tag_of_test
        error_count = 0
        for item in zip(tag_of_test, res, words):
            if (item[0] != item[1]):
                #print item
                # import pdb
                # pdb.set_trace()
                error_count += 1
        print " erro count: ", error_count
        print "amount of not Os is: ", len(not_o_tags)
        #print not_o_tags
