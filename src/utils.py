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

# Fonctions classiques permettant de mesurer les performances du modèle
def loss01(yhat, y):
	return np.mean(yhat != y)
# def accuracy(yhat, y):
# 	 return 1 - loss01(yhat, y)

# Regularized SVM loss function
def svm_loss(lambda_, w, X, y):
	"""
		* X: data to classify
		* y: the corresponding labels
		return: the loss of the trained model
	"""
	hinge_term = np.mean(np.maximum(0., 1 - y * (X @ w)))
	reg_term = (lambda_/2) * np.linalg.norm(w)**2
	return hinge_term + reg_term


def grad_hinge(w, X, y):
	res = np.zeros(w.shape[0])
	for i in range(X.shape[0]):
		if y[i] * (X[i] @ w) <= 1:
			res -= y[i] * X[i] / X.shape[0]
	return res

# Function computing the gradient of the loss function
def grad_svm(w, lambda_, X, y):
	"""
		* X: data to classify
		* y: the corresponding labels
		return: the gradient of loss of the trained model
	"""
	# res = np.zeros(w.shape[0])
	# for i in range(X.shape[0]):
	# 	if y[i] * (X[i] @ w) <= 1:
	# 		res -= y[i] * X[i] / X.shape[0]
	return grad_hinge(w, X, y) + lambda_ * w

def decision_svm(w, X):
	return X @ w

def prediction_svm(w, X):
	return 2 * (decision_svm(w,X) > 0) - 1



def sample_index(n, index_list):
	cond = True
	while cond:
		idx = np.random.randint(n)
		cond = (idx in index_list)
	return idx


# class mySVM:

# 	def __init__(self, lambda_, n_iter, learning_rate, optim_method):
# 		self.lbd = lambda_
# 		self.optim = optim_method
# 		self.epochs = n_iter
# 		self.lr = learning_rate

# 	def ugd_update(self, w, t, grad, X, y):
# 		w -= self.lr[t] * grad(w, self.lbd, X, y)
# 		return w
	
# 	def decision_function(self, X):
# 		"""
# 			* X: data to classify
# 			return: the frontier 
# 		"""
# 		return X @ self.coef
	
# 	def predict(self, X):
# 		"""
# 			* X: data to classify
# 			return: the predicted labels
# 		"""
# 		return 2 * (self.decision_function(X) > 0) - 1

# 	def fit(self, X, y):
# 		"""
# 			* X: train covariates
# 			* y: train labels
# 			return: the model trained with the chosen GD algorithm
# 		"""
# 		self.coef = np.zeros(X.shape[1])
# 		res = [[], []]
# 		for t in range(self.epochs):
# 			if self.optim == 'ugd':
# 				self.coef = self.ugd_update(self.coef, t, grad, X, y)
# 			loss1 = svm_loss(self.lbd, self.coef, X, y)
# 			loss2 = loss01(self.predict(X), y)
# 			res[0].append(loss1)
# 			res[1].append(loss2)
# 			print(f"Iteration {t+1}: svm loss={loss1} ;  0-1 loss={loss2}")
# 		return np.array(res)



def plot_perf(fig, ax, res):
	loss1, loss2 = res[0], res[1]
	xaxis = np.array([i for i in range(loss1.shape[0])])

	ax[0].set_title("SVM Loss through learning")
	ax[0].plot(xaxis, loss1, c='purple')
	ax[0].set_xlabel("Epoch")
	ax[0].set_ylabel("SVM Loss")

	ax[1].set_title("0-1 Loss through learning")
	ax[1].plot(xaxis, loss2, c='lightblue')
	ax[1].set_xlabel("Epoch")
	ax[1].set_ylabel("0-1 Loss")
	return fig, ax