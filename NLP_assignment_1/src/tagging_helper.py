import fileinput
import re

__author__ = 'itay'

fname = raw_input("enter file name to work on")

with open("./tagged/" + fname, 'r') as file_content:
    content = file_content.read()

words = re.split(r"\s+", content)

# this can probably be turned into unreadable list comprehension - todo Yuri go for it.
for i, word in enumerate(words):
    w, t = word.split("/")
    correction = raw_input(word + " OK? enter or correction")
    if correction != "":
        words[i] = w + "/" + correction

    with open("./tagged/" + fname, 'w') as f:
        f.write(" ".join(words))


