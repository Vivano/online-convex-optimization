import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


### 1 - Preliminaries ##########################################


def create_labels(labels):
	"""
		* input: the array of labels to convert for 0 vs all classification
		return: an array with 1 for 0 elements, -1 otherwise
	"""
	for i in range(labels.shape[0]):
		if labels[i] != 0:
			labels[i] = -1
		else:
			labels[i] = 1
	return labels



def prepare_mnist(path):
	"""
		* path: path to the mnist dataset to load
		* mod: string describing the nature of the dataset ('train' ot 'test')
		return: the labels and normalized data
	"""
	print("uploading data...")
	data = pd.read_csv(path, sep=',', header=None)
	print("data uploaded...")
	# labels for 0 vs all
	labels = create_labels(data[0].to_numpy())
	# dropping labels column
	data = data.drop(columns=[0])
	X = data.to_numpy()
	# normalizing the dataset (dividing all elements by 255)
	X = X / 255
	# adding a vector of 1 to capture the bias (intercept)
	bias = np.ones((X.shape[0], 1))
	X = np.concatenate((X, bias), axis=1)
	print("data preprocessed.\n")
	return labels, X



### 2 - SVM MODEL ##########################################

# Regularized SVM loss function
def svm_loss(lambda_, C_, w, X, y):
	"""
		* X: data to classify
		* y: the corresponding labels
		return: the loss of the trained model
	"""
	hinge_term = C_ * np.mean(np.maximum(0., y * (X @ w)))
	reg_term = (lambda_/2) * np.norm(w)**2
	return hinge_term + reg_term

# Function computing the gradient of the loss function
def grad(w, C, X, y):
	"""
		* X: data to classify
		* y: the corresponding labels
		return: the gradient of loss of the trained model
	"""
	res = np.zeros(w.shape[0])
	for i in range(X.shape[0]):
		if y[i] * (X[i] @ w) <= 1:
			res -= C * y[i] * X[i]
	return np.array(res) + w


class mySVM:

	def __init__(self, lambda_, C, n_iter, gradient_descent, learning_rate):
		self.lbd = lambda_
		self.C = C
		self.epochs = n_iter
		self.GD = gradient_descent
		self.lr = learning_rate

	def fit(self, X, y):
		"""
			* X: train covariates
			* y: train labels
			return: the model trained with the chosen GD algorithm
		"""
		self.coef = self.GD(self.epochs, self.lr, grad, X, y, self.C)
		return self
	
	def decision_function(self, X):
		"""
			* X: data to classify
			return: the frontier 
		"""
		return X @ self.coef
	
	def predict(self, X):
		"""
			* X: data to classify
			return: the predicted labels
		"""
		return 2 * (self.decision_function(X) > 0) - 1




# Fonctions classiques permettant de mesurer les performances du modèle

def loss01(yhat, y):
	return np.mean(yhat != y)

def accuracy(yhat, y):
	return np.mean(yhat == y)