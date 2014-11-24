# coding=utf-8

import urllib2
import urllib
import codecs
import os
import sys

def do_post(text):
    # This is the URL where the Morphological analyzer web service is running
    url = "http://www.cs.bgu.ac.il/~cohenrap/hebmed/"
    #params = urllib.urlencode({'mytext':text.encode("cp1255"),"output":"plain"})
    params = urllib.urlencode({'mytext':text, 'output':'plain'})
    request = urllib2.Request(url, params)
    response = urllib2.urlopen(request)
    return response.read()

"""----------------------EXAMPLE-------------------------"""
if __name__ == '__main__':
    # print do_post("\u05db\u05d5\u05d0\u05d1 \u05dc\u05d9 \u05d4\u05e8\u05d0\u05e9. \u05de\u05ea\u05d9 \u05e0\u05d2\u05de\u05e8\u05ea \u05d4\u05de\u05e2\u05d1\u05d3\u05d4?")

    # Given a unicode string (encoded in utf-8) get back analysis
    # and print result in a utf-8 file.
    # For example, the BguCorpusReader returns Hebrew strings encoded
    # as utf-8 Python unicode strings.
    f = codecs.open("result.txt", "w", encoding="utf-8")
    w = u'\u05d4\u05d7\u05d3\u05e9\u05d4' # Ha-hadasha as a unicode string
    a = w.encode("utf-8") # Ha-hadasha as an array
    analysis = do_post(a)
    Uanalysis = analysis.decode("utf-8")
    f.write(Uanalysis)
    f.close()
    print analysis
    # analysis = do_post(u"בעיר".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"הילד".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"והלכנו".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"כשלג".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"לים".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"ממקום".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"כשאמר".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"משהלך".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"לכשהלך".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"בכשבדק".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"מכשהלך".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"וכשאבקש".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"וגלידה".encode("utf-8"))
    # print analysis
    #
    # analysis = do_post(u"מבתוך".encode("utf-8"))
    # print analysis

    analysis = do_post(u"במשק".encode("utf-8"))
    print analysis

    analysis = do_post(u"שהרפורמה".encode("utf-8"))
    print analysis

PREFIXES = {
    'DEF': [u'ה'],
    'PREPOSITION': [u'ב',u'כ', u'ל', u'מ', u'מכש' u'מב', u'בכש'],
    'CONJ': [u'ו'],
    'TEMP-SUBCONJ': [u'כש',u'מש', u'לכש'],
    'REL-SUBCONJ': [u'ש']
}

ALL_PREFIXES = [u'ה', u'ב',u'כ', u'ל', u'מ', u'מכש' u'מב', u'בכש',  u'ו', u'כש',u'מש', u'לכש', u'ש']