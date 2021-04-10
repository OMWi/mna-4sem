from sympy import *

#variant 9
eps = 0.0001
m = 0.4
a = 0.7
x = Symbol("x")
y = Symbol("y")
def main():

    print("tg(xy + m) = x")
    print("ax^2 + 2y^2 = 1")
    print("x > 0, y > 0")

    print("Метод простых итераций")
    X = sqrt((1 - 2*y**2) / a)
    Y = (atan(x) - m) / x
    x0 = 1.0
    y0 = 0.4
    ans = simple_iters(X, Y, x0, y0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

    print("Метод Ньютона")
    f1 = tan(x*y + m) - x
    f2 = a*x**2 + 2*y**2 - 1
    vec0 = Matrix([1.0, 0.4])
    ans = newton(f1, f2, vec0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")


    test1()
    test2()
    test3()

def simple_iters(X, Y, x0, y0):
    x1 = X.subs(x, x0).subs(y, y0)
    y1 = Y.subs(x, x0).subs(y, y0)
    count = 1
    while abs(x0 - x1) > eps or abs(y0 - y1) > eps:
        x0 = x1
        y0 = y1
        x1 = X.subs(x, x0).subs(y, y0)
        y1 = Y.subs(x, x0).subs(y, y0)
        #print("Iteration {}".format(count))
        #print(f"x1 {x1}; y1 {y1}")
        count += 1
    return [x1, y1, count]

def newton(f1, f2, vec0):
    F = Matrix([f1, f2])
    J = Matrix([[diff(f1, x), diff(f1, y)], [diff(f2, x), diff(f2, y)]])
    J = J.inv()
    vec1 = (vec0 - J.inv()*F).subs(x, vec0[0]).subs(y, vec0[1])
    count = 1

    while (abs(vec0[0] - vec1[0]) > eps or abs(vec0[1] - vec1[1]) > eps):
    	vec0 = vec1
    	vec1 = (vec0 - J*F).subs(x, vec0[0]).subs(y, vec0[1])
    	count += 1
    return [vec1[0], vec1[1], count]

def test1():
    print("\nTest1")
    print("0.1x^2 + x + 0.2y^2 - 0.3 = 0")
    print("0.2x^2 + y - 0.1xy - 0.7 = 0")
    f1 = 0.1*x**2 + x + 0.2*y**2 - 0.3
    f2 = 0.2*x**2 + y - 0.1*x*y - 0.7

    print("Метод простых итераций")
    X = solve(f1, x)[1]
    Y = solve(f2, y)[0]
    x0 = 0.2
    y0 = 0.6
    ans = simple_iters(X, Y, x0, y0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

    print("Метод Ньютона")
    vec0 = Matrix([0.2, 0.6])
    ans = newton(f1, f2, vec0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

def test2():
    print("\nTest2")
    print("0.78x^2 + 2x + y^2 - 1 = 0")
    print("0.44x^2 + y + xy - 0.5 = 0")
    f1 = 0.78*x**2 + 2*x + y**2 - 1
    f2 = 0.44*x**2 + y + x*y - 0.5

    print("Метод простых итераций")
    X = solve(f1, x)[1]
    Y = solve(f2, y)[0]
    x0 = 0.4
    y0 = 0.3
    ans = simple_iters(X, Y, x0, y0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

    print("Метод Ньютона")
    vec0 = Matrix([0.4, 0.3])
    ans = newton(f1, f2, vec0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

def test3():
    print("\nTest3")
    print("x^2 + y^2 = 1")
    print("x^3 - y = 0")
    f1 = x**2 + y**2 - 1
    f2 = x**3 - y

    print("Метод простых итераций")
    X = solve(f2, x)[0]
    Y = solve(f1, y)[1]
    x0 = 0.8
    y0 = 0.5
    ans = simple_iters(X, Y, x0, y0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

    print("Метод Ньютона")
    vec0 = Matrix([0.8, 0.5])
    ans = newton(f1, f2, vec0)
    print(f"x = {ans[0]}\ty = {ans[1]}\t число итераций = {ans[2]}")

if __name__ == '__main__':
    main()
