import numpy as np
import cvxpy as cp

def SoftThreshold(x, y):
	return np.maximum(x-y, 0)

	
def simplex_proj(x):
	cond1 = (np.sum(x) == 1)
	cond2 = all(x >= 0)
	if cond1 and cond2:
		return x
	else:
		j = x.shape[0]
		x_sorted = np.sort(x)[::-1]
		while ( np.sum(x_sorted[:j]) - j*x_sorted[j-1] )  >= 1:
			j -= 1
		d = j
		theta = (np.sum(x_sorted[:d]) - 1) / d
		return SoftThreshold(x, theta)

	
def l1_ball_projection(z, x):
	if np.linalg.norm(x, 1) <= z:
		return x
	else:		
		proj = simplex_proj(abs(x) / z)
		return np.sign(x) * z * proj 

	
def simplex_proj_weighted_norm(x, D):
	d = x.shape[0]
	cond1 = (np.sum(x) == 1)
	cond2 = all(x >= 0)
	if cond1 and cond2:
		return x
	else:
		Dx = np.sort(D @ x)[::-1]
		#dxx = np.sort(dx)[::-1]
		Dinv = np.linalg.inv(D)
		d0 = 0
		max_ = Dx[0] - (1/np.trace(Dinv[:1, :1])) * (x[:1])
		for i in range(1,d):
			tmp = Dx[i] - (np.sum(x[:i+1]) - 1) / np.trace(Dinv[:i+1,:i+1])
			if tmp >= max_:
				max_ = tmp
				d0 = i+1
		# term = Dx - (1/np.cumsum(Dinv))*(np.cumsum(x) -1) 
		# d0 = np.argmax(term) + 1
		theta = (np.sum(x[:d0]) - 1) / np.trace(Dinv[:d0,:d0])
		return Dinv @ SoftThreshold(Dx, theta)


def generalized_projection(x, D, z=1):
	if np.sum(np.abs(x)) <= z:
		return x
	else:
		proj = simplex_proj_weighted_norm(abs(x) / z, D)
	return np.sign(x) * z * proj


def generalized_projection2(y, W, z):   # page 41
    d = y.shape[0]
    x = cp.Variable(d)
    objective = cp.Minimize(cp.matrix_frac(y - x, W)) # min (y-x) W^-1 (y-x)
    constraint = [cp.norm(x, 1) <= z] # s.t. l1 norm(y) <= z
    prob = cp.Problem(objective, constraint)
    result = prob.solve()
    if result != cp.OPTIMAL:
        print(f'Generalized projection error: optimization is not optimal and ends with status {result}')
    return x.value
