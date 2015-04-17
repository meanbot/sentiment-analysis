#!/usr/bin/env python
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import NaiveBayes as NB
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import KFold
import re
import nltk

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    stops = set(stopwords.words("english"))
    filtered = [w for w in tokens if not w in stops]
    stemmer=PorterStemmer()
    stems = stem_tokens(tokens,stemmer)
    return stems
class SentimentAnalysis():
    """Use object of this class to build a new model 
        for sentiment analysis
    """
    def __init__(self, filename,classifier='NaiveBayes'):
        self.classifier = NB.NaiveBayes()
        self.filename = filename
        data = pd.read_csv(filename, header=None, \
                                delimiter="\t", quoting=3)
        self.corpus = data[1]
        self.labels = data[0]
        self.build_vocab(self.corpus)
    def build_vocab(self,corpus):
        vectorizer = TfidfVectorizer(max_features=5000,\
                                    analyzer='word', tokenizer=tokenize,\
                                    ngram_range=(1,2), norm=None,\
                                    preprocessor=None, stop_words=None)
        self.vectorizer = vectorizer.fit(corpus)
        self.c_vectorizer = CountVectorizer(max_features=5000,\
                                            analyzer='word', tokenizer=tokenize,\
                                            ngram_range=(1,2), \
                                            preprocessor=None, stop_words=None,\
                                            vocabulary=vectorizer.vocabulary_)

    def feature_extract(self,corpus,vectorizer="tfidf"):
        if vectorizer == "count":
            return self.c_vectorizer.transform(corpus)
        else:
            return self.vectorizer.transform(corpus)

    def fit(self,corpus=None,labels=None):
        if corpus == None:
            corpus = self.corpus
            labels = self.labels
        train_features = self.feature_extract(corpus).toarray()
        self.classifier.fit(train_features,labels)

    def cross_validate(self,folds=3):
        # Do 10 fold cross-validation
        data_size = len(self.corpus)
        kf = KFold(data_size, n_folds=folds, shuffle=True)
        k = 1
        tot = 0
        for train_index, test_index in kf:
            train_data = self.feature_extract(self.corpus[train_index]).toarray()
            train_labels = self.labels[train_index]
            test_data = self.feature_extract(self.corpus[test_index],'count').toarray()
            test_labels = self.labels[test_index]     
            self.classifier.fit(train_data,train_labels)
            score = self.classifier.score(test_data,test_labels)
            tot+=score*100/len(test_data)
            print "Score for fold ", k, " is ", score*100/len(test_data)
            k+=1
        print "Average Score is: ", tot/folds

    def predict(self, corpus, original=True):
        test_features = self.feature_extract(corpus,'count').toarray()
        return self.classifier.predict(test_features,original)

