#!/usr/bin/env python
# -*- coding: utf-8-sig -*-

from configparser import ConfigParser
import os

from gensim.models import TfidfModel
from gensim.corpora import TextCorpus, MmCorpus, Dictionary

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''Name Leyend:
    * sentenceX: string
    *     sentX: string vector
    * bow_sentX: bag of word vector [(word_id,1)]
    * bow_sentX_tfidf: bag of word vector with tfidf coefficients [(word_id,word_tfidf)]
    * nvec_sentX_tfidf: numpy array with tfidf coefficients (it's obtained normalizing
                        words in both sentences.)
'''

#Please modify your all the paths for your resources
print('Modify the paths of your corpus on config.ini file')
input()

config = ConfigParser()
config.read('config.ini')

#TODO: Generalize this step puting corpus_path as your actual corpus
#Config file must allow wikipedia, Gutenberg, ...
corpus_path = config['WIKI']['en'][1:-1]

dictionary = Dictionary.load_from_text(os.path.relpath(corpus_path+'_wordids.txt.bz2'))
bow_corpus = MmCorpus(os.path.relpath(corpus_path+'_bow.mm'))

try:
    tfidf = TfidfModel.load(corpus_path+'wiki-tfidf.model')
except:
    tfidf = TfidfModel()
    tfidf = TfidfModel(bow_corpus,dictionary)
    tfidf._smart_save(corpus_path+'wiki-tfidf.model')
    pass

#testing sentences
sentence1 = 'pilar pescado en la tarde es fatal'
sentence2 = 'machacar pescado al atardecer es terrible'

#Transforming sentences
sent1 = sentence1.split()
sent2 = sentence2.split()

bow_sent1 = dictionary.doc2bow(sent1)
bow_sent2 = dictionary.doc2bow(sent2)

bow_sent1_tfidf = tfidf[bow_sent1]
bow_sent2_tfidf = tfidf[bow_sent2]

#from bowvec to numerical list sent
nvec1 = []
nvec2 = []
vec1 = dict(bow_sent1_tfidf)
vec2 = dict(bow_sent2_tfidf)
words = set(vec1.keys()).union(vec2.keys())
for word in words:
    nvec1.append(vec1.get(word,0.0))
    nvec2.append(vec2.get(word,0.0))
    
#from numerical list sent to numpy vec
nvec_sent1_tfidf = np.asarray(nvec1)
nvec_sent2_tfidf = np.asarray(nvec2)

#Then reshape to work fine with sklearn pairwise similarity measures
A = nvec_sent1_tfidf.reshape(1,-1)
B = nvec_sent2_tfidf.reshape(1,-1)

#Sklearn TfIdf-Cosine similarity
print(cosine_similarity(A,B)[0][0])
'''>>> 0.23215559112005074'''