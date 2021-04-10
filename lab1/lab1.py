import numpy as np

def main():
    option = 9
    matrix_c = np.array([
        [0.2, 0, 0.2, 0, 0],
        [0, 0.2, 0, 0.2, 0],
        [0.2, 0, 0.2, 0, 0.2],
        [0, 0.2, 0, 0.2, 0],
        [0, 0, 0.2, 0, 0.2]])
    matrix_d = np.array([
        [2.33, 0.81, 0.67, 0.92, -0.53],
        [-0.53, 2.33, 0.81, 0.67, 0.92],
        [0.92, -0.53, 2.33, 0.81, 0.67],
        [0.67, 0.92, -0.53, 2.33, 0.81],
        [0.81, 0.67, 0.92, -0.53, 2.33]])
    vector_b = np.array([[4.2], [4.2], [4.2], [4.2], [4.2]])
    matrix_a = matrix_c*option + matrix_d
    matrix_a = np.concatenate((matrix_a, vector_b), axis=1)
    np.set_printoptions(precision=6)
    print("Система:\n{}".format(matrix_a))
    print("Метод Гаусса : {}".format(gauss_method1(matrix_a.copy())))
    print("Метод Гаусса с выбором по столбцу : {}".format(gauss_method2(matrix_a.copy())))
    print("Метод Гаусса с выбором по всей матрице : {}".format(gauss_method3(matrix_a.copy())))
    x = gauss_method1(matrix_a.copy())
    print("Невязка : {}".format(get_nev(matrix_a, x)))

    test1 = np.array([
            [2, -1, 0, 0],
            [-1, 1, 4, 13],
            [1, 2, 3, 14]
    ], dtype = float)
    print("\nТест 1")
    print("Система:\n{}".format(test1))
    print("Метод Гаусса : {}".format(gauss_method1(test1.copy())))
    print("Метод Гаусса с выбором по столбцу : {}".format(gauss_method2(test1.copy())))
    print("Метод Гаусса с выбором по всей матрице : {}".format(gauss_method3(test1.copy())))
    x = gauss_method1(test1.copy())
    print("Невязка : {}".format(get_nev(test1, x)))

    test2 = np.array([
            [1, 2, 3, 1],
            [2, -1, 2, 6],
            [1, 1, 5, -1],
    ], dtype = float)
    print("\nТест 2")
    print("Система:\n{}".format(test2))
    print("Метод Гаусса : {}".format(gauss_method1(test2.copy())))
    print("Метод Гаусса с выбором по столбцу : {}".format(gauss_method2(test2.copy())))
    print("Метод Гаусса с выбором по всей матрице : {}".format(gauss_method3(test2.copy())))
    x = gauss_method1(test2.copy())
    print("Невязка : {}".format(get_nev(test2, x)))

    test3 = np.array([
            [3, 2, -5, -1],
            [2, -1, 3, 13],
            [1, 2, -1, 9]
    ], dtype = float)
    print("\nТест 3")
    print("Система:\n{}".format(test3))
    print("Метод Гаусса : {}".format(gauss_method1(test3.copy())))
    print("Метод Гаусса с выбором по столбцу : {}".format(gauss_method2(test3.copy())))
    print("Метод Гаусса с выбором по всей матрице : {}".format(gauss_method3(test3.copy())))
    x = gauss_method1(test3.copy())
    print("Невязка : {}".format(get_nev(test3, x)))


def get_nev(matrix, x):
    n = len(matrix)
    r = np.empty(n)
    for k in range(n):
        r[k] = matrix[k][n]
        sum = 0
        for i in range(n):
            sum += matrix[k][i]*x[i]
        r[k] -= sum
        r[k] = abs(r[k])
    return r.max()

def gauss_method1(matrix_a):
    n = len(matrix_a)
    x = np.zeros(n)
    for k in range(n):
        if matrix_a[k][k] == 0:
            swapped = False
            for j in range(k+1, n):
                if matrix_a[j][k] != 0:
                    matrix_a[[j, k]] = matrix_a[[k, j]]
                    swapped = True
                    break
            if swapped == False:
                raise ValueError()

        for i in range(k+1, n):
            q = matrix_a[i][k] / matrix_a[k][k]
            for j in range(n+1):
                matrix_a[i][j] = matrix_a[i][j] - q*matrix_a[k][j]

    x[n-1] = matrix_a[n-1][n] / matrix_a[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = matrix_a[i][n]
        for j in range(i+1, n):
            x[i] = x[i] - matrix_a[i][j]*x[j]
        x[i] /= matrix_a[i][i]
    return x

def gauss_method2(matrix_a):
    #выбор по столбцу
    n = len(matrix_a)
    x = np.zeros(n)
    for k in range(n):
        max_index = k
        for row_index in range(k, n):
            if abs(matrix_a[row_index][k]) > abs(matrix_a[max_index][k]):
                max_index = row_index
        matrix_a[[max_index, k]] = matrix_a[[k, max_index]]

        for i in range(k+1, n):
            q = matrix_a[i][k] / matrix_a[k][k]
            for j in range(n+1):
                matrix_a[i][j] = matrix_a[i][j] - q*matrix_a[k][j]

    x[n-1] = matrix_a[n-1][n] / matrix_a[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = matrix_a[i][n]
        for j in range(i+1, n):
            x[i] = x[i] - matrix_a[i][j]*x[j]
        x[i] /= matrix_a[i][i]
    return x

def gauss_method3(matrix_a):
    n = len(matrix_a)
    for k in range(n):
        max_elem = matrix_a[k][0]
        max_index = k
        for i in range(k, n):
            for j in range(n):
                if abs(matrix_a[i][j]) > max_elem:
                    max_elem = abs(matrix_a[i][j])
                    max_index = i
        matrix_a[[k, max_index]] = matrix_a[[max_index, k]]

        for i in range(k+1, n):
            q = matrix_a[i][k] / matrix_a[k][k]
            for j in range(n+1):
                matrix_a[i][j] = matrix_a[i][j] - q*matrix_a[k][j]
    x = np.zeros(n)
    x[n-1] = matrix_a[n-1][n] / matrix_a[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = matrix_a[i][n]
        for j in range(i+1, n):
            x[i] = x[i] - matrix_a[i][j]*x[j]
        x[i] /= matrix_a[i][i]
    return x

if __name__ == '__main__':
    main()
