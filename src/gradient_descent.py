import convex_tools
import numpy as np


def unconstrained_gd(epochs, eta, gradient, X, y, C):
	w = np.zeros(X.shape[1])
	for t in range(epochs):
		w -= eta[t] * gradient(w, C, X, y)
	return w

def projected_unconstrained_gd(epochs, eta, gradient, X, y, C):
	w = np.zeros(X.shape[1])
	for t in range(epochs):
		proj_w = w - eta[t] * gradient(w, C, X, y)
		w = convex_tools.l1_ball_projection(proj_w)
	return w

