from convex_tools import *
from utils import *
import numpy as np
from utils import grad_svm


# def GradientDescent(epochs, eta, lambda_, X, y):
# 	n, d = X.shape
# 	w = np.ones(d)
# 	w_list = [w]
# 	for t in range(epochs):
# 		w = w - eta[t] * grad_svm(w, lambda_, X, y)
# 		w_list.append(w)
# 	return w_list

def GradientDescent(X, y, projected=True, z=100, epochs=10**4, eta=None, lambda_=1/3):
	n, d = X.shape
	w = np.ones(d)
	w_list = [w]
	if eta is None:
		eta = np.array([1/(lambda_*(t+1)) for t in range(epochs)])
	for t in range(epochs):
		w = w - eta[t] * grad_svm(w, lambda_, X, y)
		if projected:
			w = l1_ball_projection(z, w)
		w_list.append(w)
	# print(len(w_list))
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


def StochasticGradientDescent(X, y, projected=True, z=100, epochs=10**4, lambda_=1/3, eta=None, ind=None):
	n, d = X.shape
	w = np.ones(d)
	w_list = [w]
	if eta is None:
		eta = np.array([1/(lambda_*(t+1)) for t in range(epochs)])
	if ind is None:
		ind = np.random.choice(n, epochs)
	for t in range(epochs):	
		idx = ind[t]
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



def StochasticMirrorDescent(X, y, projected=True, z=100, epochs=10**4, eta=None, ind=None):
	n,d = X.shape
	w, w_unproj = np.ones(d), np.ones(d)
	w_list = [w_unproj]
	if eta is None:
		eta = np.array([1/np.sqrt(t+1) for t in range(epochs)])
	if ind is None:
		ind = np.random.choice(n, epochs)
	for t in range(epochs):	
		idx = ind[t]
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		w_unproj = w_unproj - eta[t] * grad_hinge(w, xt, yt)
		if projected:
			w = l1_ball_projection(z, w_unproj)
		else:
			w = w_unproj
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


def StochasticExponentiatedGradient(X, y, z=100, epochs=10**4, eta=None, ind=None):
	n, d = X.shape
	weights = np.ones(2*d)/(2*d)
	# theta = np.zeros(2*d)
	w = np.ones(d)
	w_list = [w]
	if eta is None:
		eta = np.array([1/np.sqrt(t+1) for t in range(epochs)])
	if ind is None:
		ind = np.random.choice(n, epochs)
	for t in range(epochs):	
		idx = ind[t]
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



def Adagrad(X, y, projected=True, z=100, epochs=10**4, ind=None):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	S = 1e-20 * np.ones(d)
	if ind is None:
		ind = np.random.choice(n, epochs)
	for t in range(epochs):
		idx = ind[t]
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		instg = grad_hinge(w, xt, yt)
		
		S = S + instg**2
		D = np.diag(np.sqrt(S))
		Dinv = np.linalg.inv(D)
		w_unproj = w - Dinv @ instg
		if projected:
			w = generalized_projection_diag(w_unproj, np.diag(D), z)
		else:
			w = w_unproj
		w_list.append(w)

	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list


# def AdagradSolver(X, y, epochs, z):
# 	n,d = X.shape
# 	w = np.ones(d)
# 	w_list = [w]
# 	S = 1e-20 * np.ones(d)
# 	for t in range(epochs):
# 		idx = np.random.randint(n)
# 		xt = X[idx, :].reshape(1, -1)
# 		yt = np.array([y[idx]])
# 		instg = grad_hinge(w, xt, yt)
		
# 		S = S + instg**2
# 		D = np.diag(np.sqrt(S))
# 		Dinv = np.linalg.inv(D)
# 		w_unproj = w - Dinv @ instg
# 		w = generalized_projection(w_unproj, Dinv, z)
# 		w_list.append(w)
# 	w_list = np.cumsum(w_list, axis=0)
# 	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
# 	return w_list



# def OnlineNewtonStep(X, y, epochs, z, lambda_, gamma_):
# 	n,d = X.shape
# 	w = np.ones(d)
# 	w_list = [w]
# 	A = np.diag(np.ones(d) / gamma_**2)
# 	print(f"A init shape : {A.shape}")
# 	Ainv = np.diag(np.ones(d) * gamma_**2)
# 	print(f"Ainv init shape : {Ainv.shape}")
# 	for t in range(epochs):
# 		idx = np.random.randint(n)
# 		xt = X[idx, :].reshape(1, -1)
# 		yt = np.array([y[idx]])
# 		instg = grad_svm(w, lambda_, xt, yt)
# 		print(f"instg shape : {instg.shape}")
# 		A = A + instg @ instg.T
# 		print(f"A shape : {A.shape}")
# 		Ainstg = Ainv @ instg
# 		print(f"Ainstg shape : {Ainstg.shape}")
# 		Ainv = Ainv - ((Ainstg @ Ainstg.T) / (1 + (instg.T @ Ainstg)))
# 		print(f"Ainv shape : {Ainv.shape}") 
        
# 		w_unproj = w - (1 / gamma_) * Ainv @ instg
# 		# print(w_unproj.shape)
# 		# D = np.diag(np.diag(A))
# 		w = generalized_projection(w_unproj, Ainv, z)   # faut trouver un solver
# 		w_list.append(w)
# 	w_list = np.cumsum(w_list, axis=0)
# 	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
# 	return w_list



def OnlineNewtonStep(X, y, projected=True, z=100, epochs=10**4, lambda_=1/3, gamma_=1/8, ind=None):
	n,d = X.shape
	if ind is None:
		ind = np.random.choice(n, epochs)
	xt = X[ind]
	yt = y[ind]

	w = np.ones(d) # dim (785,)
	# m = w
	w_list = [w]
	A = np.diag(np.ones(d)/(gamma_**2)) # dim dxd
	Ainv = np.diag(np.ones(d)*(gamma_**2)) # dim dxd
	# params = [m]
        
	for t in range(epochs):
		
		instg = np.expand_dims(instgradreg(w, xt[t,], yt[t], lambda_), axis=1) # dim dx1
		A = A + instg @ (instg.T) # dim dxd
		Ainstg = Ainv @ instg # dim dx1
		Ainv = Ainv - (Ainstg @ Ainstg.T) / (1 + instg.T @ Ainstg) # dim dxd
		w_unproj = np.squeeze(np.expand_dims(w, axis=1)-(1/gamma_) * Ainv @ instg , axis=1) # dim (785,)
        
		if projected:
			# w_unproj = np.squeeze(np.expand_dims(w, axis=1)-(1/gamma_) * Ainv @ instg , axis=1) # dim (785,)
			w = generalized_projection(w_unproj, Ainv, z)
		else:
			w = w_unproj
		w_list.append(w)
        # accumulated mean of x
		# m = (t*m + w)/(t+1) 
		# params.append(m)

	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list



def SREG(X, y, z=100, epochs=10**4, lambda_=1/3):
	n,d = X.shape
	weights = np.ones(2*d)/(2*d)
	w = np.ones(d)
	w_list = [w]
	for t in range(1,epochs+1):
		idx = np.random.randint(n)
		j = np.random.randint(d)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])

		instg_j = grad_hinge(w, xt, yt)[j]
		eta = 1 / np.sqrt((t+1)*d)
		weights[j] = np.exp(-eta*d*instg_j) * weights[j]
		weights[j+d] = np.exp(eta*d*instg_j) * weights[j+d]
		weights = weights / np.sum(weights)
		w = z * (weights[:d] - weights[d:])
		
		w_list.append(w)
	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t]/(t+1) for t in range(len(w_list))]
	return w_list



def SBEG(X, y, z=100, epochs=10**4, lambda_=1/3):
	n,d = X.shape
	w = np.ones(d)
	w_list = [w]
	weights = np.ones(2*d) / (2*d)
	wp = np.ones(2*d) / (2*d)

	for t in range(epochs):
		idx = np.random.randint(n)
		xt = X[idx, :].reshape(1, -1)
		yt = np.array([y[idx]])
		
		A = np.random.choice(2 * d, 1, p=weights)
		j = A * (A<=d) + (A-d) * (A>d) - 1
		s = 2 * (A<=d) - 1
		
		#print(grad_svm(w, lambda_, x, y).shape)
		# print(grad_hinge(w, xt, yt).shape)
		instg_j = grad_hinge(w, xt, yt)[j]
		eta = 1 / np.sqrt((t+1)*d)
		gamma = np.minimum(1, d*eta)
		
		wp[A] = np.exp(-eta*s*instg_j // weights[A]) * wp[j]
		wp /= np.sum(wp)
		weights = (1-gamma) * wp + gamma / (2*d)
		w = z * (wp[:d] - wp[d:])
		w_list.append(w)

	w_list = np.cumsum(w_list, axis=0)
	w_list = [w_list[t] / (t+1) for t in range(len(w_list))]
	return w_list