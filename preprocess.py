'''This packages provide the flow for:
    * preprocesing data
    * measure all similarity functions
    * and given a set of pair sentences as input write a csv file with
    data as [pair_sentence_id,similarities]
'''

from configparser import ConfigParser
import os, sys
import pandas as pd

from gensim.models import TfidfModel
from gensim.corpora import TextCorpus, MmCorpus, Dictionary

import numpy as np

#Importing all features
from sklearn.metrics.pairwise import cosine_similarity as kcosine
from textsim.tokendist import cosine_distance_scipy as scosine
from gensim.matutils import cossim as gcosine
from gensim.matutils import jaccard as gjaccard
from distances import n_similarity as nsimilarity
from distances import jonh2016_similarity as jonh2016
from distances import harmonic_best_pair_word_sim as hbestpw

'''Name Leyend:
    * sentenceX: string
    *     sentX: string vector
    * bow_sentX: bag of word vector [(word_id,1)]
    * bow_sentX_tfidf: bag of word vector with tfidf coefficients [(word_id,word_tfidf)]
    * nvec_sentX_tfidf: numpy array with tfidf coefficients (it's obtained normalizing
                        words in both sentences.)
'''

class Model:

    def __init__(self,model,corpus):
        #Please modify your all the paths for your resources
        print('Modify the paths of your corpus on config.ini file')
        input()

        self.name = model
        self.corpus = corpus
        #SIM_VECTOR
        self.features = {
            'gcosine':gcosine,
            'gjaccard':gjaccard,
            'scosine':scosine,
            'kcosine':kcosine,
            'nsimilarity':nsimilarity,
            'jonh2016':jonh2016,
            'hbestpw':hbestpw
        }
        self.columns=list(self.features.keys())

        config = ConfigParser()
        config.read('config.ini')

        '''TODO: Generalize this step puting corpus_path as your actual corpus
        Config file must allow wikipedia, Gutenberg, ...'''
        corpus_path = config['WIKI']['en'][1:-1]
        self.dictionary, self.bow_corpus = self.load_corpus() 
        self.tfidf = self.generate_corpus()

    def load_corpus(self):
        self.dictionary = Dictionary.load_from_text(os.path.relpath(corpus_path+'_wordids.txt.bz2'))
        self.bow_corpus = MmCorpus(os.path.relpath(corpus_path+'_bow.mm'))
        return self.dictionary, self.tfidf

    def generate_corpus(self):
        try:
            self.tfidf = TfidfModel.load(corpus_path+'wiki-tfidf.model')
        except:
            self.tfidf = TfidfModel()
            self.tfidf = TfidfModel(bow_corpus,dictionary)
            self.tfidf._smart_save(corpus_path+'wiki-tfidf.model')
            pass
        return self.tfidf

class Pair:
    def __init__ (self, sentence1, sentence2, model):
        
        self.model = model.lower()

        #Transforming sentences
        self.sent1 = sentence1.split()
        self.sent2 = sentence2.split()

        self.bow_sent1 = dictionary.doc2bow(self.sent1)
        self.bow_sent2 = dictionary.doc2bow(self.sent2)

        self.bow1_tfidf = tfidf_bow_vector(self.bow_sent1)
        self.bow2_tfidf = tfidf_bow_vector(self.bow_sent2)

        self.nvec1, 
        self.nvec2 = bowvec_to_npvec(self.bow1_tfidf, self.bow2_tfidf)

        self.fvec1 = fvec(self.nvec1)
        self.fvec2 = fvec(self.nvec2)

    def tfidf_bow_vector(self, bow_sent):
        if self.model == 'tfidf':
            return(tfidf[bow_sent])
        else:
            return False

    def bowvec_to_npvec(self, bow1_tfidf,bow2_tfidf):
        #from bowvec to numerical list sent
        nvec1 = []
        nvec2 = []
        vec1 = dict(bow1_tfidf)
        vec2 = dict(bow2_tfidf)
        words = set(vec1.keys()).union(vec2.keys())
        for word in words:
            nvec1.append(vec1.get(word,0.0))
            nvec2.append(vec2.get(word,0.0))
        
        #from numerical list sent to numpy vec
        nvec_sent1_tfidf = np.asarray(nvec1)
        nvec_sent2_tfidf = np.asarray(nvec2)

        return nvec_sent1_tfidf,nvec_sent2_tfidf

    def fvec(nvec):
        '''Some similarity measures need 1d or flatten vectors to work, e.g. sklearn pairwise
        '''
        return nvec.reshape(1,-1)

def apply_all_sim(pair,model):
    row = []
    for feature in features:
        if feature in ['kcosine']:
            row.append(features[feature](pair.fvec1,pair.fvec2))
        if feature in ['scosine']:
            row.append(features[feature](pair.nvec1,pair.nvec2))
        if feature in ['gcosine','gjaccard']:
            row.append(features[feature](pair.bow1_tfidf,pair.bow2_tfidf))
        if model.name not in ['w2v','p2v']:
            pass
        else:
            row.append('NaN')
    return row

if __name__ == '__main__':

    if len(sys.argv) == 5:
        indata = sys.argv[2]
        model = sys.argv[3]
        corpus = sys.argv[4]
        outdata = sys.argv[5]

        #input sentences
        data = read_csv('data/'+indata,sep='\t',header=0)

        #Initializing the model
        model = Model(model,corpus)

        #declaring ouput DataFrame
        output = pd.DataFrame(columns=model.columns)

        #measure the first 100 sentence pairs
        for i in range(100):
            sentence1 = data.sentence1[i]
            sentence2 = data.sentence2[i]
            
            pair = Pair(sentence1,sentence2)
            
            sim_vec = []
            sim_vec = apply_all_sim(pair, model)
            output.loc[i] = sim_vec

        output.to_csv('data/'+outdata+'.csv')

    else:
        print('\n'.join(["Unexpected number of commandline arguments.", 
            "Usage: ./process.py {indata} {model} {outdata}"]))

