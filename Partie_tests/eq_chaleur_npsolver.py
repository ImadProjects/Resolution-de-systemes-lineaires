import numpy as np
from matplotlib import pyplot as plt

# return the matrix A with the correct coefficients
def generate_coef(n):
	mat = np.zeros((n ** 2, n ** 2));
	for i in range(n):
		for j in range(n):
			ni = i * n + j
			# dx
			if i < n - 1:
				mat.itemset(ni, (i + 1) * n + j, 1)
			if i > 0:
				mat.itemset(ni, (i - 1) * n + j, 1)
				
			# dy
			if j < n - 1:
				mat.itemset(ni, i * n + j + 1, 1)
			if j > 0:
				mat.itemset(ni, i * n + j - 1, 1)
				
			mat.itemset(ni, ni, -4);
	return -((n + 1) ** 2) * mat

# create a n * n matrix from a value function
def func_to_matrix(f, n):
	return np.fromfunction(lambda i, j: f(i/(n - 1), j/(n - 1)), (n, n))

# create a vector of size n² from a matri of size n * n
def matrix_to_vector(m):
	return m.flatten().T

# create a matrix of size n * n from a vector of size n²
def vector_to_matrix(v):
	n = int(np.sqrt(v.size))
	return v.reshape((n, n))

# show the imge corresponding to the given function
def show_image(f, solver, N, title):
	A = generate_coef(N)

	b = matrix_to_vector(func_to_matrix(f, N))
	print("b = ")
	print(b)
	
	x = vector_to_matrix(solver(A, b))
	print("x = ")
	print(x)
	
	plt.imshow(x, cmap='hot', extent=(0, 1, 0, 1), interpolation='bilinear')
	plt.title(title)
	plt.show()

# the function f defining the initial heat of a point radiator
def point_radiator(i, j):
	return np.where((i - 0.5) ** 2 + (j - 0.5) ** 2 < 0.01, 1, 0)

# the function f defining the initial heat of a point radiator
def wall_radiator(i, j):
	return np.where(i < 0.0001, 1, 0)


def test_equation():
        N = 32
        show_image(point_radiator, np.linalg.solve, N, "Radiateur Point")
        show_image(wall_radiator, np.linalg.solve, N, "Radiateur Mur")


if __name__ == '__main__':
        test_equation()
