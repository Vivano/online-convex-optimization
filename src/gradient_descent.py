from convex_tools import *
from utils import *
import numpy as np


# def GradientDescent(epochs, eta, lambda_, X, y):
# 	n, d = X.shape
# 	w = np.ones(d)
# 	w_list = [w]
# 	for t in range(epochs):
# 		w = w - eta[t] * grad_svm(w, lambda_, X, y)
# 		w_list.append(w)
# 	return w_list

def GradientDescent(X, y, epochs=10e4, eta=None, z=100, lambda_=1/3, projected=True):
	n, d = X.shape
	w = np.ones(d)
	w_list = [w]
	if eta is None:
		eta = np.array([1/lambda_*(t+1) for t in range(epochs)])
	for t in range(epochs):
		w = w - eta[t] * grad_svm(w, lambda_, X, y)
		if projected:
			w = l1_ball_projection(z, w)
		w_list.append(w)
	return w_list

# def StochasticGradientDescent(epochs, eta, lambda_, X, y):
# 	n, d = X.shape
# 	w = np.ones(d)
# 	w_list = [w]
# 	for t in range(epochs):	
# 		idx = np.random.randint(n)
# 		xt = X[idx, :].reshape(1, -1)
# 		yt = np.array([y[idx]]) 
# 		w = (1 - 1/(t+1)) * w - eta[t] * grad_hinge(w, xt, yt)
# 		w_list.append(w)
# 	# print(xt.shape, yt.shape)
# 	w_list = np.cumsum(w_list, axis=0)
# 	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
# 	return w_list


def StochasticGradientDescent(X, y, epochs=10e4, eta=None, z=100, lambda_=1/3, projected=True):
	n, d = X.shape
	w = np.ones(d)
	w_list = [w]
	if eta is None:
		eta = np.array([1/(lambda_*(t+1)) for t in range(epochs)])
	for t in range(epochs):	
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		w = (1 - 1/(t+1)) * w - eta[t] * grad_hinge(w, xt, yt)
		if projected:
			w = l1_ball_projection(z, w)
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list


# def StochasticGradientDescents(X, y, epochs=1e4, eta=None, z=100, lambda_=1/3):
# 	n,d = X.shape
# 	w1, w2 = np.ones(d), np.ones(d)
# 	w_list1, w_list2 = [w1], [w2]
# 	for t in range(epochs):	
# 		idx = np.random.randint(n)
# 		xt = X[idx, :].reshape(1, -1)
# 		yt = np.array([y[idx]])
# 		w1 = (1 - 1/(t+1)) * w1 - eta[t] * grad_hinge(w1, xt, yt)
# 		w_list1.append(w1)
# 		w2 = (1 - 1/(t+1)) * w2 - eta[t] * grad_hinge(w2, xt, yt)
# 		w2 = l1_ball_projection(z, w2)
# 		w_list2.append(w2)
# 	w_list1 = np.cumsum(w_list1, axis=0)
# 	w_list1 = [w_list1[t]/(t+1) for t in range(len(w_list1))]
# 	w_list2 = np.cumsum(w_list2, axis=0)
# 	w_list2 = [w_list2[t]/(t+1) for t in range(len(w_list2))]
# 	return w_list1, w_list2



def StochasticMirrorDescent(X, y, epochs=1e4, eta=None, z=100):
	n,d = X.shape
	w, w_unproj = np.ones(d), np.ones(d)
	w_list = [w_unproj]
	if eta is None:
		eta = np.array([1/np.sqrt(t+1) for t in range(epochs)])
	for t in range(epochs):	
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		w_unproj = w_unproj - eta[t] * grad_hinge(w, xt, yt)
		w = l1_ball_projection(z, w_unproj)
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list



# def StochasticExponentiatedGradient(X, y, epochs=1e4, eta=None, z=100):
# 	n,d = X.shape
# 	w = np.zeros(d)
# 	w_list = [w]
# 	weights = np.ones(2*d)/(2*d)
# 	theta = np.zeros(2*d)
# 	if eta is None:
# 		eta = np.array([1/np.sqrt(t+1) for t in range(epochs)])
# 	for t in range(epochs):	
# 		idx = np.random.randint(n)
# 		xt = X[idx, :].reshape(1, -1)    
# 		yt = np.array([y[idx]]) 
# 		instg = grad_hinge(w, xt, yt)
# 		weights = np.exp(theta - eta[t] * np.r_[-instg, instg]) * weights
# 		weights /= np.sum(weights)
# 		w = z*(weights[:d] - weights[d:])
# 		w_list.append(w)
# 	w_list = np.cumsum(w_list, axis=0)
# 	l = []
# 	for i,w in enumerate(w_list):
# 		l.append(w/(i+1))
# 	return l


def StochasticExponentiatedGradient(X, y, epochs=1e4, eta=None, z=100):
	n, d = X.shape
	weights = np.ones(2*d)/(2*d)
	# theta = np.zeros(2*d)
	w = np.ones(d)
	w_list = [w]
	if eta is None:
		eta = np.array([1/np.sqrt(t+1) for t in range(epochs)])
	for t in range(epochs):	
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)    
		yt = np.array([y[idx]]) 
		instg = grad_hinge(w, xt, yt)
		# theta = theta - eta[t] * np.r_[-instg, instg]
		weights = np.exp(eta[t] * np.r_[-instg, instg]) * weights
		weights = weights / np.sum(weights)
		w = z * (weights[:d] - weights[d:])
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list



def Adagrad(X, y, epochs, z):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	S = 1e-20 * np.ones(d)
	for t in range(epochs):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg = grad_hinge(w, xt, yt)
		
		S = S + instg**2
		D = np.diag(np.sqrt(S))
		w_unproj = w - np.linalg.inv(D) @ instg
		w = generalized_projection(w_unproj, D, z)
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list



def ONS(X, Y, epochs, z, lambda_, gamma):
	n,d = X.shape
	w = np.zeros(d)
	w_list = [w]
	A = np.diag(np.ones(d) / gamma**2)
	Ainv = np.diag(np.ones(d) * gamma**2)    
	for t in range(epochs):
		idx = np.random.randint(n)
		x = X[idx, :].reshape(1, -1)
		y = np.array([Y[idx]])
		instg = grad_svm(w, lambda_, x, y)

		A += instg @ instg.T  
		Ainstg = Ainv@instg
		Ainv -= (Ainstg @ (instg.T @ Ainv)) / (1 + instg.T @ Ainstg) 
        
		yy = w - (1 / gamma) * Ainv @ instg
		D = np.diag(np.diag(A))
		w = generalized_projection(yy, D, z)   # faut trouver un solver
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	l = []
	for i,w in enumerate(w_list):
		l.append(w/(i+1))
	return l
