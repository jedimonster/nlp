__author__ = 'itay'

import re

with open("checked1/fixed_itay.txt", 'r') as f1:
    text1 = f1.read()
with open("checked2/yuris_checked", 'r') as f2:
    text2 = f2.read()

words1 = re.split(r"\s+", text1)
words2 = re.split(r"\s+", text2)

for i, w in enumerate(words1):
    if w != words2[i]:
        print(" tagger1 says: " + w + " tagger2 says: " + words2[i])

