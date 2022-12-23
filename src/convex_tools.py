import numpy as np

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
