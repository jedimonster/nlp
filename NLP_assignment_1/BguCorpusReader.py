#encoding=cp1255

from nltk.tree import Tree
from nltk.corpus.reader import ChunkedCorpusReader
from nltk.tokenize import RegexpTokenizer



class BguCorpusReader(ChunkedCorpusReader):
    
    #format= : השבוע    DEF:NOUN-M,S,ABS:
    def __init__(self, directory="",fileids=r"haaretz.bgu",myEncoding="utf-8"):
        ChunkedCorpusReader.__init__(self, directory ,fileids , str2chunktree=self.__str2BguTree,sent_tokenizer=RegexpTokenizer('\n\n', gaps=True),encoding=myEncoding)
        self._format = format
        
    def __str2BguTree(self,text):
        lines = text.split('\n')
        tree = Tree('s',[])
        for line in lines:
            if line=='':
                continue
            mlist = line.split("\t")
            word = mlist[0]
            raw = mlist[1]
            tree.append((word,bguTag(raw)))
        return tree
    
"""======================class bguTag=====================================
======================================================================="""
    
class bguTag:
    def __init__(self,raw):
        self.__raw = raw
        
        
    def getRaw(self):
        return self.__raw
    
    def getBguTag(self):
        return self.__parseRaw(self.__raw)
       
    
    def getPosTag(self):
        bguTag = self.getBguTag()
        pos = ''
        pre = bguTag[0]
        mid = bguTag[1]
        suf = bguTag[2]
        for t in pre:
            pos+=t[0]+' '
        pos+=mid[0]+' '
        for t in suf:
            pos+=t[0]+' '
        return pos[:-1]

    def __parseRaw(self,raw):
        pre,mid,suf = raw.split(':')
        pre = pre.split('+')
        pre = [w.split('-') for w in pre]
        suf = suf[:-1]
        suf = suf.split('+')
        suf = [w.split('-') for w in suf]
        mid = mid.split('-')
        if len(mid)>1:
            mid = [mid[0],mid[1].split(',')]
        sufs = []
        for w in suf:
            if len(w)>1:
                sufs.append([w[0],w[1].split(',')])
            else:
                if w[0]!='':
                    sufs.append([w[0],[]]) 
        pres = []
        for w in pre:
            if len(w)>1:
                pres.append([w[0],w[1].split(',')])
            else:
                if w[0]!='':
                    pres.append([w[0],[]])         
        return [pres,mid,sufs]
    

"""----------------------EXAMPLE-------------------------"""
if __name__ == '__main__':
    c = BguCorpusReader()
    tagged_words = c.tagged_words()
    w,t = tagged_words[29]
    print w
    print t.getRaw()
    print t.getBguTag()
    print t.getPosTag()