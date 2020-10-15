import matplotlib.pyplot as plt
import numpy as np

from math import exp, sqrt, pi
from functools import reduce, partial
from collections import deque
from itertools import accumulate


def runge_kutta(f, x0, y0, h, n):
    """
    Example:
        xs, ys = runge_kutta(lambda x, y: x+y, 0.0, 0.0, 0.1, 10)
    """
    def step(xsys, v):
        xs, ys = xsys
        xv = x0 + v * h
        yv = ys[0]
        k1 = f(xv, yv)
        k2 = f(xv + h/2, yv + k1 * h/2)
        k3 = f(xv + h/2, yv + k2 * h/2)
        k4 = f(xv + h, yv + h * k3)
        kv = h/6 * (k1 + 2 * (k2 + k3) + k4)
        y = yv + kv
        xs.appendleft(xv)
        ys.appendleft(y)
        return(xs, ys)
    return reduce(step, range(1, n+1), (deque([x0]), deque([y0])))


def runge_kutta_alt(f, x0, y0, h, n):
    def step(xy, v):
        _, yv = xy
        x = x0 + v * h
        k1 = f(x, yv)
        k2 = f(x + h/2, yv + k1 * h/2)
        k3 = f(x + h/2, yv + k2 * h/2)
        k4 = f(x + h, yv + h * k3)
        kv = h/6 * (k1 + 2 * (k2 + k3) + k4)
        y = yv + kv
        return (x, y)
    yield from accumulate(range(1, n+1), step, initial=(x0, y0))


def runge_kutta_alt2(f, x0, y0, end, n):
    h = end / n
    yield from runge_kutta_alt(f, x0, y0, h, n)


def f(kappa, t, h):
    """System of ODEs describing a round object moving through a liquid subject to gravity
    h(t) = (x position, y position, x velocity, y velocity)
    result(t) = (x velocity, y velocity, x acceleration, y acceleration)
    """
    g = 9.81
    vx, vy = h[2:]
    a = -kappa*(vx**2 + vy**2)
    b = np.arctan(vy/vx)
    return np.array([vx, vy, a * np.cos(b), a * np.sin(b) - g])


r = 0.145  # radius in meters
c_w = 0.4  # air resistance coefficient
m = 0.420  # weight of football in kilograms
A = r ** 2 * pi

rho_air = 1.2041  # density of air at 20°C in kg/m³
kappa = (c_w * rho_air * A) / (2 * m)

ts, hs = zip(
    *runge_kutta_alt2(partial(f, kappa), 0.0, np.array([0.0, 0.0, 28.0, 18.0]), 3.0, 1000))
xs, ys, vxs, vys = np.transpose(np.array(hs))
ys[ys < 0] = 0
print(ys)
plt.subplot(2, 1, 1)
plt.plot(xs, ys, label="y(x)")
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(ts, xs, label="$x(t)$")
plt.plot(ts, ys, label="$y(t)$")
plt.plot(ts, vxs, label=r"$v_x(t)$ in $\frac{m}{s}$")
plt.plot(ts, vys, label=r"$v_y(t)$ in $\frac{m}{s}$")
plt.xlabel("$t$ in $s$")
plt.legend()
plt.show()
