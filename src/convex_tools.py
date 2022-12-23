import numpy as np

def SoftThreshold(x, y):
	return np.maximum(x-y, 0)

def simplex_proj(x):
	if np.sum(x) == 1:
		return x
	else:
		# print(x.shape)
		x = np.sort(x)[::-1]
		i = 1
		while ( x[i-1] - (np.sum(x[:i]) - 1)/i ) > 0:
			i += 1
		d = i-1
		theta = (np.sum(x[:d+1]) - 1) / d
		return SoftThreshold(x, theta)

def l1_ball_projection(z, x):
	if np.linalg.norm(x) <= z:
		return x
	else:		
		proj = simplex_proj(abs(x) / z)
		return np.sign(x) * proj