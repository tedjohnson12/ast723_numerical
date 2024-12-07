"""
Functions that are not figure-specific
"""
from typing import Tuple, Callable
import numpy as np
from scipy.optimize import bisect

def psi_max(gamma: float)-> float:
    return np.arcsin(1/gamma)

def alpha_sq(
    m1_sq: float,
    gamma: float,
)-> float:
    return m1_sq * (gamma-1)/(gamma+1) + 2/(gamma+1)

def y_sq(
    x: float,
    m1_sq: float,
    gamma: float,
)-> float:
    
    return (1-x)**2 * (m1_sq*x - alpha_sq(m1_sq, gamma))/(alpha_sq(m1_sq, gamma) + 2/(gamma+1) * m1_sq - m1_sq*x)

def y(
    x: float,
    m1_sq: float,
    gamma: float,
)-> Tuple[float, float]:
    _y_sq = y_sq(x, m1_sq, gamma)
    return _y_sq**0.5, -_y_sq**0.5

def dy_dx(
    x: float,
    m1_sq: float,
    gamma: float,
    eps: float = 1e-6,
)-> float:
    return (np.sqrt(y_sq(x+eps, m1_sq, gamma)) - np.sqrt(y_sq(x-eps, m1_sq, gamma))) / (2*eps)

def get_psi_max_func(
    m1_sq: float,
    gamma: float,
    eps: float = 1e-6,
)-> Callable[[float], float]:
    def func(x: float)-> float:
        lhs = dy_dx(x, m1_sq, gamma, eps)
        rhs = np.sqrt(y_sq(x, m1_sq, gamma))/x
        return lhs - rhs
    return func

def get_psi_max(
    m1_sq: float,
    gamma: float,
    eps: float = 1e-6,
)-> float:
    func = get_psi_max_func(m1_sq, gamma, eps)
    try:
        xbest = bisect(func, a=alpha_sq(m1_sq, gamma)/m1_sq+1e-5, b=1-1e-5)
        ybest = np.sqrt(y_sq(xbest, m1_sq, gamma))
        res = (np.arctan(ybest/xbest), np.sqrt(xbest**2 + ybest**2))
    except RuntimeError:
        res = (np.nan, np.nan)
    # if (res < 0) or res > 1:
    #     res = np.nan
    return res


def get_eta_func(
    psi: float,
    m1_sq: float,
    gamma: float,
)-> Callable[[float], float]:
    def func(eta: float)-> float:
        _alpha_sq = alpha_sq(m1_sq, gamma)
        _sinpsi = np.sin(psi)
        _cospsi = np.cos(psi)
        lhs = eta**2 * _sinpsi**2
        rhs = (1 - eta * _cospsi)**2 * (m1_sq * eta * _cospsi - _alpha_sq) / (_alpha_sq + 2 / (gamma + 1) * m1_sq - m1_sq * eta * _cospsi)
        return lhs - rhs
    return func

def eta_weak(
    psi: float,
    psi_max: float,
    eta_max:float,
    m1_sq: float,
    gamma: float,
)-> float:
    if psi > psi_max:
        return np.nan
    func = get_eta_func(psi, m1_sq, gamma)
    xmin = eta_max*np.cos(psi_max)/np.cos(psi)
    xmax = 1/np.cos(psi)
    return bisect(func, a=xmin, b=xmax)

def eta_strong(
    psi: float,
    psi_max: float,
    eta_max:float,
    m1_sq: float,
    gamma: float,
)-> float:
    if psi > psi_max:
        return np.nan
    func = get_eta_func(psi, m1_sq, gamma)
    xmin = alpha_sq(m1_sq, gamma)/m1_sq
    xmax = eta_max
    return bisect(func, a=xmin, b=xmax)
    
def tan_phi_weak(
    psi: float,
    m1_sq: float,
    gamma: float,
)-> float:
    eta = eta_weak(psi, m1_sq, gamma)
    return (1-eta*np.cos(psi)) / (eta*np.sin(psi))

def tan_phi_strong(
    psi: float,
    m1_sq: float,
    gamma: float,
)-> float:
    eta = eta_strong(psi, m1_sq, gamma)
    return (1-eta*np.cos(psi)) / (eta*np.sin(psi))

if __name__ == "__main__":
    
    m1_sq = 20000
    gamma = 5/3
    psi = 0.64
    print(psi_max(gamma))
    print(get_psi_max(m1_sq, gamma))
    # print(eta_weak(psi, m1_sq, gamma))
    # print(eta_strong(psi, m1_sq, gamma))