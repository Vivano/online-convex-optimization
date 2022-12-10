import numpy as np

def SoftThreshold(x, y):
	return np.maximum(x-y, 0)

def simplex_proj(x):
	x = np.sort(x, reverse=True)
	i = 0
	while ( x[i] - (np.sum(x[:i+1]) - 1)/i ) > 0:
		i +=1
	d = i
	theta = (np.sum(x[:d+1]) - 1) / d
	return SoftThreshold(x, theta)

def l1_ball_projection(z, x):
	if np.norm(x) <= np.norm(x):
		return x
	else:
		proj = simplex_proj(abs(x) / z)
		return np.sign(x) * proj