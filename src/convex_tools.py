import numpy as np

def norm_1(x):
	res = 0
	for i in range(x.shape[0]):
		res += abs(x[i])
	return res

def SoftThreshold(x, y):
	return np.maximum(x-y, 0)

def simplex_proj(x):
	cond1 = (np.sum(x) == 1)
	cond2 = all(x >= 0)
	if cond1 and cond2:
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
	if norm_1(x) <= z:
		return x
	else:		
		proj = simplex_proj(abs(x) / z)
		return np.sign(x) * proj