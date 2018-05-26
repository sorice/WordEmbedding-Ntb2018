#!/usr/bin/env python
# -*- coding: utf-8-sig -*-

'''This packages provide all similarity functions created from papers.
'''

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def n_similarity(sent1,sent2, model):
    '''n_similarity method implemented in w2v and p2v.'''
    return model.n_similarity(sent1,sent2)

def jonh2016_similarity(sent1, sent2, model, ALPHA = 0.25):
    '''John, Adebayo Kolawole and Caro, Luigi Di and Boella, Guido. 
    **NORMAS at SemEval-2016 Task 1: 
        SEMSIM: A Multi-Feature Approach to Semantic Text Similarity**. 
    Publisher ACM, 2016.p2v
    
    Only valid for wordembedded methos that implement wv.similarity
    method.
    '''
    sim_vector = []

    for wordA in sent1:
        for wordB in sent2:
            try:
                sim = model.wv.similarity(wordA,wordB)
                if sim > ALPHA:
                    sim_vector.append(sim)
            except:
                pass

    return sum(sim_vector)/(len(sim_vector) or 1)

def harmonic_best_pair_word_sim(sent1,sent2,model):
    '''Only valid for wordembedded methos that implement wv.similarity
    method.
    '''
    p=0
    for wordA in sent1:
        m = 0
        for wordB in sent2:
            try:
                m = max(m, model.wv.similarity(wordA,wordB))
            except:
                pass
        p += m
    p = p/len(sent1)

    q=0
    for wordA in sent2:
        m = 0
        for wordB in sent1:
            try:
                m = max(m, model.wv.similarity(wordA,wordB))
            except:
                pass
        q += m
    q = q/len(sent2)

    sim = 2*p*q/(p+q or 1)
    return sim

def best_pair_word_overlap(sent1,sent2, pdTfIdf):
    p=0
    for wi in sent1:
        m = 0
        for wc in sent2:
            try:
                winp = pdTfIdf.loc[wi].values.reshape(1,-1)
                wcnp = pdTfIdf.loc[wc].values.reshape(1,-1)
                m = float(max(m, cosine_similarity(winp,wcnp)))
            except:
                pass
        p += m
    p = p/len(sent1)

    q=0
    for wc in sent2:
        m = 0
        for wi in sent1:
            try:
                wcnp = pdTfIdf.loc[wc].values.reshape(1,-1)
                winp = pdTfIdf.loc[wi].values.reshape(1,-1)
                m = float(max(m, cosine_similarity(winp,wcnp)))
                #print(m, type(m))
            except:
                pass
        q += m
    q = q/len(sent2)

    sim = 2*p*q/(p+q or 1)
    return sim