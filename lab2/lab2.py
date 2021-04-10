import numpy as np
from numpy import linalg
def main():
    np.set_printoptions(precision=5)
    epsilon = 0.00001

    matrix = get_matrix()
    vector_b = np.array([1.2, 2.2, 4.0, 0.0, -1.2])
    print("Матрица:\n{}".format(matrix))
    print("Вектор:\n{}\n".format(vector_b))
    print("Метод простых итераций:")
    ans = simple_iterations(matrix, vector_b, epsilon)
    print(ans)
    print("Метод Зейделя:")
    ans = seidel(matrix, vector_b, epsilon)
    print(ans)
    print("Проверка через numpy:\n{}\n\n".format(linalg.solve(matrix, vector_b)))

    matrix1 = np.array([
            [10., 1., -1.],
            [1., 10., -1.],
            [-1., 1., 10.]
    ])
    vector1 = np.array([11., 10., 10.])
    print("Матрица:\n{}".format(matrix1))
    print("Вектор:\n{}\n".format(vector1))
    print("Метод простых итераций:")
    ans = simple_iterations(matrix1, vector1, epsilon)
    print(ans)
    print("Метод Зейделя:")
    ans = seidel(matrix1, vector1, epsilon)
    print(ans)
    print("Проверка через numpy:\n{}\n\n".format(linalg.solve(matrix1, vector1)))


    matrix2 = np.array([
            [5, 1],
            [-2, 8]
    ], dtype = float)
    vector2 = np.array([3, 4], dtype = float)
    print("Матрица:\n{}".format(matrix2))
    print("Вектор:\n{}\n".format(vector2))
    print("Метод простых итераций:")
    ans = simple_iterations(matrix2, vector2, epsilon)
    print(ans)
    print("Метод Зейделя:")
    ans = seidel(matrix2, vector2, epsilon)
    print(ans)
    print("Проверка через numpy:\n{}\n\n".format(linalg.solve(matrix2, vector2)))

    matrix3 = np.array([
            [20.9, 1.2, 2.1, 0.9],
            [1.2, 21.2, 1.5, 2.5],
            [2.1, 1.5, 19.8, 1.3],
            [0.9, 2.5, 1.3, 32.1]
    ])
    vector3 = np.array([21.70, 27.46, 28.76, 49.72])
    print("Матрица:\n{}".format(matrix3))
    print("Вектор:\n{}\n".format(vector3))
    print("Метод простых итераций:")
    ans = simple_iterations(matrix3, vector3, epsilon)
    print(ans)
    print("Метод Зейделя:")
    ans = seidel(matrix3, vector3, epsilon)
    print(ans)
    print("Проверка через numpy:\n{}\n\n".format(linalg.solve(matrix3, vector3)))

def get_matrix():
    k = 9
    matrix_c = np.array([
            [0.01, 0, -0.02, 0, 0],
            [0.01, 0.01, -0.02, 0, 0],
            [0, 0.01, 0.01, 0, -0.02],
            [0, 0, 0.01, 0.01, 1],
            [0, 0, 0, 0.01, 0.01]
    ])
    matrix_d = np.array([
            [1.33, 0.21, 0.17, 0.12, -0.13],
            [-0.13, -1.33, 0.11, 0.17, 0.12],
            [0.12, -0.13, -1.33, 0.11, 0.17],
            [0.17, 0.12, -0.13, -1.33, 0.11],
            [0.11, 0.67, 0.12, -0.13, -1.33]
    ])
    return k*matrix_c + matrix_d

def seidel(matrix, vector, eps):
    n = len(matrix)
    x = vector.copy()

    converge = False
    k = 0
    while not converge:
        k += 1
        next_x = x.copy()
        for i in range(n):
            s1 = sum(matrix[i][j]*next_x[j] for j in range(i))
            s2 = sum(matrix[i][j]*x[j] for j in range(i + 1, n))
            next_x[i] = (vector[i] - s1 - s2) / matrix[i][i]
        norm = abs(next_x -x)
        converge = norm.max() < eps
        x = next_x
    print("{} итераций".format(k))
    return x

def simple_iterations(matrix, vector, eps):
    n = len(matrix)
    res_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                res_matrix[i][j] = -matrix[i][j] / matrix[i][i]
    res_vector = vector.copy()
    for i in range(n):
        res_vector[i] /= matrix[i][i]
    x = res_vector.copy()
    next_x = x.copy()
    k = 1
    while True:
        x = next_x.copy()
        for i in range(n):
            next_x[i] = 0
            for j in range(n):
                next_x[i] += x[j]*res_matrix[i][j]
            next_x[i] += res_vector[i]
        norm = abs(next_x - x)
        if norm.max() < eps:
            break
        k += 1
    print("{} итераций".format(k))
    return next_x

def get_norm(matrix):
    n = len(matrix)
    norm = np.zeros(n)
    for i in range(n):
        for j in range(n):
            norm[i] += abs(matrix[i][j])
    return norm.max()

if __name__ == '__main__':
    main()
