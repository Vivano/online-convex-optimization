import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import time
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

def hinge_loss(lambda_, w, X, y):
	return np.mean(np.maximum(0., 1 - y * (X @ w)))

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


def instgradreg(x, a, b, lambda_):
    threshold = b * (a.dot(x)) # define hard-margin SVM
    gradient = -b * a
    gradient[threshold >= 1] = 0
    return gradient + lambda_ * x

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




def compute_accuracy(xtest, ytest, coef):
	pred = prediction_svm(coef, xtest)
	acc = 1 - loss01(pred, ytest)
	return acc

def accuracies(data, labels, coef_list):
	return np.array([compute_accuracy(data, labels, w) for w in coef_list])

def xval_z(name, algo, X, y, z_list=[10, 50, 100], Epochs=10**4):
	acc_list = []
	for z_value in z_list:
		w_list = algo(X, y, z=z_value, epochs=Epochs)
		acc_list.append(accuracies(X, y, w_list))
	x = np.array([t+1 for t in range(Epochs+1)])
	y = np.array(acc_list)
	fig, ax = plt.subplots(figsize=(7,5))
	for i in range(len(z_list)):
		lab = "z=" + str(z_list[i])
		ax.plot(x, acc_list[i], label=lab, alpha=0.5)
	ax.set_yscale('logit')
	ax.set_xscale('log')
	ax.legend(loc='best')
	# ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force x axis to integer
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.set_xlabel(r"Epoch $t$")
	ax.set_ylabel("Accuracy (decimal)")
	ax.set_title(f"Comparison of the {name} algorithm for different values of radius z")
	plt.show()



def xval_lbd(name, algo, X, y, lambda_list=[1., 1/3, 0.1, 0.01], Epochs=10**4):
	acc_list = []
	for z_value in lambda_list:
		w_list = algo(X, y, z=z_value, epochs=Epochs)
		acc_list.append(accuracies(X, y, w_list))
	x = np.array([t+1 for t in range(Epochs+1)])
	y = np.array(acc_list)
	fig, ax = plt.subplots(figsize=(7,5))
	for i in range(len(lambda_list)):
		lab = "lbd=" + str(lambda_list[i])
		ax.plot(x, acc_list[i], label=lab, alpha=0.5)
	ax.set_yscale('logit')
	ax.set_xscale('log')
	ax.legend(loc='best')
	# ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force x axis to integer
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.set_xlabel(r"Epoch $t$")
	ax.set_ylabel("Accuracy (decimal)")
	ax.set_title(f"Comparison of the {name} algorithm for different values of radius z")
	plt.show()
	


def execution_time(algo, X, y, projected=True, z_value=100, Epochs=10**4):
	start_time = time.time()
	_ = algo(X, y, z=z_value, epochs=Epochs)
	end_time = time.time()
	return end_time - start_time



