from convex_tools import *
from utils import *
import numpy as np


def unconstrained_gd(epochs, eta, lambda_, X, y):
	list = []
	w = np.zeros(X.shape[1])
	list.append(w)
	for t in range(epochs):
		w -= eta[t] * grad_svm(w, lambda_, X, y)
		list.append(w)
	return list

def projected_unconstrained_gd(epochs, eta, z, lambda_, X, y):
	list = []
	w = np.zeros(X.shape[1])
	list.append(w)
	list.append(w)
	for t in range(epochs):
		w = w - eta[t] * grad_svm(w, lambda_, X, y)
		w = l1_ball_projection(z, w)
		list.append(w)
	return w

