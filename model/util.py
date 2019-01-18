import re
import numpy as np
class Corpus:
    def __init__(self,file,reader,alpha=1,truncated=1):
        self.file = file
        self.reader = reader
        self.alpha = 1
        self.truncated = 1
    def clean(self,s):
        s = re.sub('[^a-zA-Z]',' ',s)
        s = re.sub('\s{2,}', ' ', s)
        s = s.lower()
        return s.split()
    def load(self):
        doc,label = self.reader(self.file)
        cdocs = []
        for d in doc:
            cdocs.append(self.clean(d))
        ndocs = self.align(cdocs)
        return self.word2idx,self.length,ndocs,label
    def align(self,doc):
        word_freq = {}
        lengths = []
        for d in doc:
            lengths.append(len(doc))
            for w in d:
                word_freq[w] = word_freq.get(w,0) + 1
        word_counter = sorted(word_freq.items(),key=lambda x:x[1],reverse=True)
        vocab = [ word  for word,freq in word_counter[:int(len(word_counter)*self.alpha)]]
        vocab.append('<PAD/>')
        np.random.shuffle(vocab)
        self.word2idx = {w:idx for idx,w in enumerate(vocab)}
        if self.truncated >0 and self.truncated <= 1:
            lengths = sorted(lengths)
            length = lengths[int(len(lengths)*self.truncated)-1]
        else:
            length = self.truncated
        self.length = length
        pad_docs = []
        for d in doc:
            pad_doc = self.padding(d,length)
            pad_docs.append(pad_doc)
        return pad_docs
    def padding(self,d,length):
        d = [ self.word2idx[w] for w in d if w in self.word2idx]
        if len(d) > length:
            d = d[:length]
        else:
            d = d + [self.word2idx['<PAD/>']]*(length-len(d))
        return d


