"""
Functions that are not figure-specific
"""
from typing import Tuple, Callable
import numpy as np
from scipy.optimize import bisect

def psi_max(gamma: float)-> float:
    """
    Maximum deflection angle in the limit of large Mach number.
    
    Parameters
    ----------
    gamma : float
        Adiabatic index
        
    Returns
    -------
    psi_max : float
    """
    return np.arcsin(1/gamma)

def alpha_sq(
    m1_sq: float,
    gamma: float,
)-> float:
    """
    Dimensionless critical speed squared
    
    Equivalent to $c_*^2/a_1^2$
    
    Parameters
    ----------
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    
    Returns
    -------
    alpha_sq : float
    """
    return m1_sq * (gamma-1)/(gamma+1) + 2/(gamma+1)

def y_sq(
    x: float,
    m1_sq: float,
    gamma: float,
)-> float:
    """
    $(u_2/u_1 \\sin\\psi)^2$
    
    Parameters
    ----------
    x : float
        $u_2/u_1 \\cos\\psi$
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    
    Returns
    -------
    y_sq : float
    """
    return (1-x)**2 * (m1_sq*x - alpha_sq(m1_sq, gamma))/(alpha_sq(m1_sq, gamma) + 2/(gamma+1) * m1_sq - m1_sq*x)

def y(
    x: float,
    m1_sq: float,
    gamma: float,
)-> Tuple[float, float]:
    """
    $(u_2/u_1 \\sin\\psi, -u_2/u_1 \\sin\\psi)$
    
    Parameters
    ----------
    x : float
        $u_2/u_1 \\cos\\psi$
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    
    Returns
    -------
    y_plus : float
        The positive solution
    y_minus : float
        The negative solution
    """
    _y_sq = y_sq(x, m1_sq, gamma)
    return _y_sq**0.5, -_y_sq**0.5

def dy_dx(
    x: float,
    m1_sq: float,
    gamma: float,
    eps: float = 1e-6,
)-> float:
    """
    Compute the deriviative of the shock polar using central difference.
    
    Parameters
    ----------
    x : float
        $u_2/u_1 \\cos\\psi$
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    eps : float
        Step size
    
    Returns
    -------
    dy_dx : float
    """
    return (np.sqrt(y_sq(x+eps, m1_sq, gamma)) - np.sqrt(y_sq(x-eps, m1_sq, gamma))) / (2*eps)

def get_psi_max_func(
    m1_sq: float,
    gamma: float,
    eps: float = 1e-6,
)-> Callable[[float], float]:
    """
    Get a function, the roots of which give the value of $u_2/u_1 \\cos\\psi$ for which $\\psi$ is maximum.
    
    Parameters
    ----------
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    eps : float
        Step size for central difference
    
    Returns
    -------
    psi_max_func : Callable[[float], float]
    """
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
    """
    Get the maximum deflection angle for general Mach number using the bisection method.
    
    Parameters
    ----------
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    eps : float
        Step size for central difference
    
    Returns
    -------
    psi_max : float
    """
    func = get_psi_max_func(m1_sq, gamma, eps)
    try:
        xbest = bisect(func, a=alpha_sq(m1_sq, gamma)/m1_sq+1e-5, b=1-1e-5)
        ybest = np.sqrt(y_sq(xbest, m1_sq, gamma))
        res = (np.arctan(ybest/xbest), np.sqrt(xbest**2 + ybest**2))
    except RuntimeError:
        res = (np.nan, np.nan)
    return res


def get_eta_func(
    psi: float,
    m1_sq: float,
    gamma: float,
)-> Callable[[float], float]:
    """
    Return a function, the roots of which are the values of $\\eta$ which are allowed for a particular deflection angle and Mach number.
    
    Parameters
    ----------
    psi : float
        Deflection angle
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    
    Returns
    -------
    eta_func : Callable[[float], float]
    """
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
    """
    Find $\\eta$ for a weak shock using the bisection method.
    
    Parameters
    ----------
    psi : float
        Deflection angle
    psi_max : float
        Maximum deflection angle
    eta_max : float
        Value of $\\eta$ at the maximum deflection angle
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    
    Returns
    -------
    eta : float
    """
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
    """
    Find $\\eta$ for a strong shock using the bisection method.
    
    Parameters
    ----------
    psi : float
        Deflection angle
    psi_max : float
        Maximum deflection angle
    eta_max : float
        Value of $\\eta$ at the maximum deflection angle
    m1_sq : float
        Mach number squared
    gamma : float
        Adiabatic index
    
    Returns
    -------
    eta : float
    """
    if psi > psi_max:
        return np.nan
    func = get_eta_func(psi, m1_sq, gamma)
    xmin = alpha_sq(m1_sq, gamma)/m1_sq
    xmax = eta_max
    return bisect(func, a=xmin, b=xmax)
