import codecs
from os import listdir
import random
from nltk import word_tokenize, TreebankWordTokenizer, WordPunctTokenizer
from terminals import Document

__author__ = 'itay'


class NewsgroupsReader(object):
    def __init__(self, tokenize):
        self._tokenize = tokenize
        self._tokenizer = WordPunctTokenizer()

    def get_training(self):
        return self._get_docs('datasets/20news-bydate-train')

    def get_test(self):
        return self._get_docs('datasets/20news-bydate-test')

    def _get_docs(self, path):
        doc_objects = []
        i = 0

        for category in listdir(path):
            for f in listdir(path + "/" + category):
                with codecs.open(path + "/" + category + "/" + f, 'r', encoding='latin1') as content_file:
                    text = content_file.read()
                    tokens = self._tokenizer.tokenize(text) if self._tokenize else text
                    doc_objects.append(Document(i, tokens, category))
                    i += 1

        random.shuffle(doc_objects)
        return doc_objects


if __name__ == '__main__':
    train = NewsgroupsReader().get_docs('datasets/20news-bydate-train')
    print train[0].category, train[0].doc