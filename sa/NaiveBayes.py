#!/usr/bin/env python
import numpy as np
from sklearn import preprocessing

def estimate_parameters(X,Y, nc, nf, ns):
	'''
	X: a mxn array of m TD-IDF features for n samples
	Y: a 1xn array of encoded labels corresponding to the above samples
	nc: number of classes
	nf: size of the feature vector
	Returns: complementary class parameter weights
	'''
	theta = np.zeros([nc,nf])
	alpha = np.ones(nf) # To smooth out the max-likelihood est.
	alpha_sum = np.sum(alpha)
	for label in xrange(nc):
		'''
		Complementary Naive Bayes parameter estimation
		'''
		cmpl_docs_indices = (Y != label)
		cmpl_docs = X[cmpl_docs_indices]
		cmpl_features = np.sum(cmpl_docs)
		normalizer = cmpl_features + alpha_sum
		theta[label] = (np.sum(X[cmpl_docs_indices],0)+alpha)/normalizer
	'''
	Set weights to logarithm of theta
	'''
	weights = np.log(theta)
	'''
	Weight Normalization
	'''
	for label in xrange(nc):
		weights[label] = weights[label]/np.sum(np.abs(weights[label]))
	return weights,theta

class NaiveBayes:
	def __init__(self):
		self.theta = 0
		self.weights = 0
		self.trained = False
		self.le = None
	def fit(self, X,y):
		'''
		X: a mxn array of m TD-IDF features for n samples
		y: a 1xn array of labels corresponding to the above samples
		'''
		n_samples = np.shape(X)[0]
		self.n_features = np.shape(X)[1]
		# Encode the labels into integers from 0 to K-1 wher K is the 
		# number of classes
		self.le = preprocessing.LabelEncoder()
		Y = self.le.fit_transform(y)
		self.n_classes = len(np.unique(Y))
		self.weights,self.theta = estimate_parameters(X, Y, self.n_classes, self.n_features,\
														n_samples)
		self.trained = True
		return
	def predict(self, X, original=False):
		'''
		X: a mxn array of m TD-IDF features for n samples
		Returns: a 1xn array of classes predicted
		'''
		if self.trained == False:
			raise Exception('No model trained yet! Use fit')
		if np.shape(X)[1] != self.n_features:
			raise ValueError('The input array should have the same number of \
								features as the trained model')
		n_samples = np.shape(X)[0]
		predicted_labels = np.ndarray(n_samples)
		class_probs = np.ndarray(self.n_classes)
		'''
		Predict class labels for each sample
		Label(sample) = arg min (class) <weights(class), wordcounts(sample)>
		'''
		for sample in xrange(n_samples):
			for c_label in xrange(self.n_classes):
				class_probs[c_label] = np.dot(self.weights[c_label],X[sample])
			predicted_labels[sample] = np.argmin(class_probs)
		if original == False:
			return predicted_labels.astype(int)
		else:
			return self.le.inverse_transform(predicted_labels.astype(int))
	def score(self, X, y):
		le = preprocessing.LabelEncoder()
		Y = le.fit_transform(y)
		labels = self.predict(X)
		return sum(Y==labels)

