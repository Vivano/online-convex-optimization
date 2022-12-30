from convex_tools import *
from utils import *
import numpy as np


def Adagradprimaldual(X, y, epochs, z):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	S = 1e-20 * np.ones(d)
	g_cumsum = 0
	np.random.seed(0)
	for t in range(epochs):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg = grad_hinge(w, xt, yt)
		
		S = S + instg**2
		D = np.diag(np.sqrt(S))
		eta = 1
		g_cumsum = (t*g_cumsum + instg)/(t+1)
		d = np.diag(D)
		yy = - (t+1)*np.sqrt(1/d)*g_cumsum
		w = generalized_projection_diag(yy, d, z) 
		w_list.append(w)

	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list


def AdagradOMD(X, y, epochs, z):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	S = 1e-20 * np.ones(d)
	np.random.seed(0)
	for t in range(1,epochs+1):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg = grad_hinge(w, xt, yt)

		S = S + instg**2
		D = np.diag(np.sqrt(S))
		d = np.diag(D)
		term = w - instg/d
		w = generalized_projection_diag(term, np.diag(D), z)
		w_list.append(w)

	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list


def Adagraddiag(X, y, epochs, z):
	n,d = X.shape
	w1 = np.ones(d)
	w2 = np.ones(d)
	w_list1 = [w1]
	w_list2 = [w2]
	S1 = 1e-20 * np.ones(d)
	S2 = 1e-20 * np.ones(d)
	g_cumsum = 0
	np.random.seed(0)
	for t in range(epochs):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg1 = grad_hinge(w1, xt, yt)
		instg2 = grad_hinge(w2, xt, yt)
		eta = 1

		S1 = S1 + instg1**2
		D1 = np.diag(np.sqrt(S1))
		d1 = np.diag(D1)
		g_cumsum = (t*g_cumsum + instg1)/(t+1)
		y1 = - eta*(t+1)*g_cumsum/d1

		S2 = S2 + instg2**2
		D2 = np.diag(np.sqrt(S2))
		d2 = np.diag(D2)
		y2 = w2 - eta*instg2/d2

		w1 = generalized_projection_diag(y1, d1, z)
		w2 = generalized_projection_diag(y2, d2, z)
		w_list1.append(w1)
		w_list2.append(w2)

	w_list1 = np.cumsum(w_list1, axis=0)
	w_list1 = [w_list1[t]/(t+1) for t in range(len(w_list1))]

	w_list2 = np.cumsum(w_list2, axis=0)
	w_list2 = [w_list2[t]/(t+1) for t in range(len(w_list2))]
	return w_list1, w_list2




def Adagradfull(X, y, epochs, z, eta = 1):
	n,d = X.shape
	w1 = np.ones(d)
	w2 = np.ones(d)
	w_list1 = [w1]
	w_list2 = [w2]
	S1 = 1e-20 * np.ones((d,d))
	S2 = 1e-20 * np.ones((d,d))
	g_cumsum = 0
	np.random.seed(0)
	for t in range(epochs):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg1 = grad_hinge(w1, xt, yt)
		instg2 = grad_hinge(w2, xt, yt)

		S1 = S1 + instg1 @ instg1.T
		D1 = np.sqrt(S1)
		d1 = np.diag(D1)
		g_cumsum = (t*g_cumsum + instg1)/(t+1)
		if np.linalg.det(D1) != 0:
			y1 = eta*(t+1)*g_cumsum
			w1 = np.linalg.inv(D1) @ y1
			print("La matirce est inversible")
		else:
			y1 = - eta*(t+1)*g_cumsum/d1
			w1 = generalized_projection_diag(y1, d1, z)
			print("La matirce n'est pas inversible")

		S2 = S2 + instg2 @ instg2.T
		D2 = np.sqrt(S2)
		d2 = np.diag(D2)
		if np.linalg.det(D2) != 0:
			y2 = eta*instg2 - D2 @ w2
			w2 = np.linalg.inv(D2) @ y2
			print("La matirce est inversible")
		else:
			y2 = w2 - eta*instg2/d2      
			w2 = generalized_projection_diag(y2, d2, z)
			print("La matirce n'est pas inversible")

		w_list1.append(w1)
		w_list2.append(w2)

	w_list1 = np.cumsum(w_list1, axis=0)
	w_list1 = [w_list1[t]/(t+1) for t in range(len(w_list1))]

	w_list2 = np.cumsum(w_list2, axis=0)
	w_list2 = [w_list2[t]/(t+1) for t in range(len(w_list2))]
	return w_list1, w_list2


def Adagraddiagregl1(X, y, epochs, z, lambda_reg=0.01, eta=1):
	n,d = X.shape
	w1 = np.ones(d)
	w2 = np.ones(d)
	w_list1 = [w1]
	w_list2 = [w2]
	S1 = 1e-20 * np.ones(d)
	S2 = 1e-20 * np.ones(d)
	g_cumsum = 0
	np.random.seed(0)
	for t in range(epochs):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg1 = grad_hinge(w1, xt, yt)
		instg2 = grad_hinge(w2, xt, yt)

		S1 = S1 + instg1**2
		D1 = np.diag(np.sqrt(S1))
		d1 = np.diag(D1)
		g_cumsum = (t*g_cumsum + instg1)/(t+1)
		w1 = np.sign(-g_cumsum)*eta*(t+1)*np.maximum(0,np.abs(g_cumsum) - lambda_reg )/d1
		w_list1.append(w1)

		S2 = S2 + instg2**2
		D2 = np.diag(np.sqrt(S2))
		d2 = np.diag(D2)
		y2 = w2 - eta*instg2/d2
		w2 = np.sign(y2)*np.maximum(0,np.abs(y2) - lambda_reg*eta/d2 )
		w_list2.append(w2)

	w_list1 = np.cumsum(w_list1, axis=0)
	w_list1 = [w_list1[t]/(t+1) for t in range(len(w_list1))]

	w_list2 = np.cumsum(w_list2, axis=0)
	w_list2 = [w_list2[t]/(t+1) for t in range(len(w_list2))]
	return w_list1, w_list2
