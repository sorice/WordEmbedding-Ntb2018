#!/usr/bin/env python
# -*- coding: utf-8-sig -*-

'''This packages provide the functions for:
    * preprocesing or wrangling data
'''

import numpy as np

DEFAULT = 9

def wrang_tfidf(sent1, sent2, TfIdfModel, id2word):
    """
    :param s1,s2: Sentences to compare.
    :type s1,s2: str
    :returns:
        * bow_sentX: bag of word vector [(word_id,1)]
        * bow_sentX_tfidf: bag of word vector with tfidf coefficients [(word_id,word_tfidf)]
        * nvec_sentX_tfidf: numpy array with tfidf coefficients (it's obtained normalizing
                            words in both sentences.)
    """
    
    #from raw sent to bowvec sent
    bowvec_sent1 = id2word.doc2bow(sent1)
    bowvec_sent2 = id2word.doc2bow(sent2)

    bowvec_sent1_tfidf = TfIdfModel[bowvec_sent1]
    bowvec_sent2_tfidf = TfIdfModel[bowvec_sent2]
    
    #from bowvec to numerical list sent
    
    nvec1 = []
    nvec2 = []
    vec1 = dict(bowvec_sent1_tfidf)
    vec2 = dict(bowvec_sent2_tfidf)
    words = set(vec1.keys()).union(vec2.keys())
    for word in words:
        nvec1.append(vec1.get(word,0.0))
        nvec2.append(vec2.get(word,0.0))
        
    #from numerical list sent to numpy vec
    nvec_sent1_tfidf = np.asarray(nvec1)
    nvec_sent2_tfidf = np.asarray(nvec2)
    A = nvec_sent1_tfidf.reshape(1,-1)
    B = nvec_sent2_tfidf.reshape(1,-1)
    
    return bowvec_sent1_tfidf,bowvec_sent2_tfidf,nvec1,nvec2, A, B