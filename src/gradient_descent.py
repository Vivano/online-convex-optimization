from convex_tools import *
from utils import *
import numpy as np


def unconstrained_gd(epochs, eta, lambda_, X, y):
	list = []
	w = np.ones(X.shape[1])
	list.append(w)
	for t in range(epochs):
		w = w - eta[t] * grad_svm(w, lambda_, X, y)
		list.append(w)
	return list

def projected_unconstrained_gd(epochs, eta, z, lambda_, X, y):
	list = []
	w = np.ones(X.shape[1])
	list.append(w)
	for t in range(epochs):
		w = w - eta[t] * grad_svm(w, lambda_, X, y)
		w = l1_ball_projection(z, w)
		list.append(w)
	return list

def SGD(X, Y, epochs, lambda_):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	for t in range(1,epochs+1):	
		idx = np.random.randint(n)
		x = X[idx, :].reshape(1, -1)
		y = np.array([Y[idx]]) 
		instg = grad_svm(w, lambda_, x, y)
		
		eta = 1/(lambda_*t)
		w = (1 - 1/t)*w - eta*instg
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	l = []
	for i,w in enumerate(w_list):
		l.append(w/(i+1))
	return l


def projected_SGD(X, Y, epochs, z, lambda_):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	for t in range(1,epochs+1):	
		idx = np.random.randint(n)
		x = X[idx, :].reshape(1, -1)
		y = np.array([Y[idx]]) 
		instg = grad_svm(w, lambda_, x, y)
		
		eta = 1/(lambda_*t)
		w = (1 - 1/t)*w - eta*instg
		w = l1_ball_projection(z, w)
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	l = []
	for i,w in enumerate(w_list):
		l.append(w/(i+1))
	return l

def SMD(X, Y, epochs, z, lambda_):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	for t in range(1,epochs+1):	
		idx = np.random.randint(n)
		x = X[idx, :].reshape(1, -1)
		y = np.array([Y[idx]]) 
		instg = grad_svm(w, lambda_, x, y)
		
		eta = 1/np.sqrt(t)
		w = w - eta*instg
		w = l1_ball_projection(z, w)
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	l = []
	for i,w in enumerate(w_list):
		l.append(w/(i+1))
	return l

def SEG(X, Y, epochs, z, lambda_):
	n,d = X.shape
	w = np.zeros(d)
	w_list = [w]
	w_ = np.ones(2*d)/(2*d)
	theta = np.zeros(2*d)
	for t in range(1,epochs+1):	
		idx = np.random.randint(n)
		x = X[idx, :].reshape(1, -1)    
		y = np.array([Y[idx]]) 
		instg = grad_svm(w, lambda_, x, y)
        
		eta = 1/np.sqrt(t)
		w_ = np.exp(eta * np.r_[-instg, instg])*w_
		w_ /= np.sum(w_)
		w = z*(w_[:d] - w_[d:])
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	l = []
	for i,w in enumerate(w_list):
		l.append(w/(i+1))
	return l

def Adagrad(X, Y, epochs, z, lambda_):
	n,d = X.shape
	w = np.zeros(d)
	w_list = [w]
	S = np.ones(d)
	for t in range(epochs):
		idx = np.random.randint(n)
		x = X[idx, :].reshape(1, -1)
		y = np.array([Y[idx]]) 
		instg = grad_svm(w, lambda_, x, y)
		
		S += instg*instg
		D = np.diag(np.sqrt(S))
		Dinv = np.linalg.inv(D)
		yy = w - Dinv @ instg
		w = generalized_projection(yy, D, z)
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	l = []
	for i,w in enumerate(w_list):
		l.append(w/(i+1))
	return l

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
