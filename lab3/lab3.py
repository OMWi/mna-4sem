import numpy as np
from sympy import sturm
from sympy.abc import x
from sympy import Poly
from scipy.misc import derivative as der

def main():
    a = 2.65804
    b = -28.0640
    c = 21.9032
    poly0 = Poly(x**3 + a * x**2 + b*x + c)
    print("Число корней по теореме Штурма : {}\n".format(sturm_method(poly0, -10, 10)))

    ranges = separate(poly0, -10, 10)
    print("Отрезки с отделенными корнями :\n{}\n".format(ranges))

    eps = 0.0001
    n = len(ranges)
    print("Метод половинного деления:")
    print("x = {}".format(bisection(ranges[0][0], ranges[0][1], eps, 0, f0)))
    print()

    print("Метод хорд:")
    print("x = {}".format(chord(ranges[0][0], ranges[0][1], eps, f0)))
    print()

    print("Метод Ньютона:")
    print("x = {}".format(newton(ranges[0][0] + (ranges[0][1] - ranges[0][0])/2, eps, f0)))

    for i in range(1, n):
        print(f"x[{i+1}]")
        print("Метод половинного деления:")
        print("x = {}".format(bisection(ranges[i][0], ranges[i][1], eps, 0, f0)))
        print()
        print("Метод хорд:")
        print("x = {}".format(chord(ranges[i][0], ranges[i][1], eps, f0)))
        print()
        print("Метод Ньютона:")
        print("x = {}".format(newton(ranges[i][0] + (ranges[i][1] - ranges[i][0])/2, eps, f0)))

    poly1 = Poly(-2.32*x**2 - 6.29*x + 9.12)
    print("Число корней по теореме Штурма : {}\n".format(sturm_method(poly1, -10, 10)))

    ranges = separate(poly1, -10, 10)
    print("Отрезки с отделенными корнями :\n{}\n".format(ranges))

    eps = 0.0001
    n = len(ranges)
    for i in range(n):
        print(f"x[{i+1}]")
        print("Метод половинного деления:")
        print("x = {}".format(bisection(ranges[i][0], ranges[i][1], eps, 0, f1)))
        print()

        print("Метод хорд:")
        print("x = {}".format(chord(ranges[i][0], ranges[i][1], eps, f1)))
        print()

        print("Метод Ньютона:")
        print("x = {}".format(newton(ranges[i][0] + (ranges[i][1] - ranges[0][0])/2, eps, f1)))

    poly2 = Poly(1.3*x**2 - 2.2*x)
    print("Число корней по теореме Штурма : {}\n".format(sturm_method(poly2, -10, 10)))
    ranges = separate(poly2, -10, 10)
    print("Отрезки с отделенными корнями :\n{}\n".format(ranges))
    print("Метод половинного деления:")
    print("x = {}".format(bisection(ranges[0][0], ranges[0][1], eps, 0, f2)))
    print()

    print("Метод хорд:")
    print("x = {}".format(chord(ranges[0][0], ranges[0][1], eps, f2)))
    print()

    print("Метод Ньютона:")
    print("x = {}".format(newton(ranges[0][0] + (ranges[0][1] - ranges[0][0])/2, eps, f2)))

def sturm_method(poly, a, b):
    sturm_seq = sturm(poly)
    nums = []
    sign_changes = 0
    for el in sturm_seq:
        num = el.subs({x:a})
        nums.append(num)
    n = len(nums)
    na = 0
    for i in range(n-1):
        if nums[i+1]*nums[i] < 0:
            na += 1

    nums.clear()
    sign_changes = 0
    for el in sturm_seq:
        num = el.subs({x:b})
        nums.append(num)
    n = len(nums)
    nb = 0
    for i in range(n-1):
        if nums[i+1]*nums[i] < 0:
            nb += 1
    return na - nb

def separate(poly, left, right):
    root_num = sturm_method(poly, left, right)
    step = 1.0
    current = left + step
    prev = left
    ranges = np.zeros((root_num, 2))
    for k in range(root_num):
        while True:
            #print(f"prev {prev}; current {current}; step {step}")
            if current > right:
                current -= step
                step /= 2
            elif sturm_method(poly, prev, current) == 0:
                if step < 0:
                    step *= -1
                if step < 16:
                    step *= 2
                current += step
            elif sturm_method(poly, prev, current) > 1:
                step /= -2
                current += step
            else:
                ranges[k][0] = prev
                ranges[k][1] = current
                prev = current
                if step < 0:
                    step *= -1
                current += step
                break
    if ranges[root_num - 1][1] < right:
        ranges[root_num - 1][1] = right
    return ranges

def f0(x):
    return x**3 + 2.65804*x**2 - 28.0640*x + 21.9032

def f1(x):
    return -2.32*x**2 - 6.29*x + 9.12

def f2(x):
    return 1.3*x**2 - 2.2*x

def bisection(left, right, eps, iters, f):
    iters += 1
    mid = (right + left)/2
    value = f(mid)
    if abs(value) < eps:
        #print("{} итераций".format(iters))
        return mid
    elif f(left)*value > 0:
        return bisection(mid, right, eps, iters, f)
    elif f(right)*value > 0:
        return bisection(left, mid, eps, iters, f)

def chord(left, right, eps, f):
    if f(left)*f(right) > 0:
        print("err")
        return None
    a = left
    b = right
    iters = 1
    c_prev = 0
    while True:
        c = a - f(a) / (f(b)-f(a)) * (b-a)
        fa = f(a)
        fb = f(b)
        fc = f(c)
        #print(f"fa {fa}; fb {fb}; fc {fc}")
        if fa*fc < 0:
            b = c
        elif fc*fb < 0:
            a = c
        else:
            print("err")
            return None
        if abs(c - c_prev) <= eps and iters > 1:
            break
        c_prev = c
        iters += 1
    #print("{} итераций".format(iters))
    return c

def newton(x, eps, f):
    iters = 1
    while True:
        xp = x - f(x) / der(f, x)
        if abs(xp - x) <= eps:
            break
        x = xp
        iters += 1
    #print("{} итераций".format(iters))
    return x


if __name__ == '__main__':
    main()
