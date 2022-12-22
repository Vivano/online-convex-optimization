import numpy as np

def SoftThreshold(x, y):
	return np.maximum(x-y, 0)

def simplex_proj(x):
	if np.sum(x) == 1:
		return x
	else:
		x = np.sort(x, reverse=True)
		i = 0
		while ( x[i] - (np.sum(x[:i+1]) - 1)/i ) > 0:
			i +=1
		d = i
		theta = (np.sum(x[:d+1]) - 1) / d
		return SoftThreshold(x, theta)

def l1_ball_projection(z, x):
	if np.norm(x) <= z:
		return x
	else:	
		proj = simplex_proj(np.norm(x) / z)
		return np.sign(x) * proj