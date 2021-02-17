"""Main module."""
import numpy as np
from scipy import optimize
from scipy.special import digamma
from scipy.special import polygamma
from numpy import linalg as LA

def trigamma(x):
    """Trigamma function.

    Parameters
    ----------
    x: float
    """
    return polygamma(1, x)

def _digammainv(y):
    # Inverse of the digamma function (real positive arguments only).
    # This function is used in the `fit` method of `gamma_gen`.
    # The function uses either optimize.fsolve or optimize.newton
    # to solve `sc.digamma(x) - y = 0`.  There is probably room for
    # improvement, but currently it works over a wide range of y:
    #    >>> y = 64*np.random.randn(1000000)
    #    >>> y.min(), y.max()
    #    (-311.43592651416662, 351.77388222276869)
    #    x = [_digammainv(t) for t in y]
    #    np.abs(sc.digamma(x) - y).max()
    #    1.1368683772161603e-13
    #
    _em = 0.5772156649015328606065120
    func = lambda x: digamma(x) - y
    if y > -0.125:
        x0 = np.exp(y) + 0.5
        if y < 10:
            # Some experimentation shows that newton reliably converges
            # must faster than fsolve in this y range.  For larger y,
            # newton sometimes fails to converge.
            value = optimize.newton(func, x0, tol=1e-10)
            return value
    elif y > -3:
        x0 = np.exp(y/2.332) + 0.08661
    else:
        x0 = 1.0 / (-y - _em)

    value, info, ier, mesg = optimize.fsolve(func, x0, xtol=1e-11,
                                             full_output=True)
    if ier != 1:
        raise RuntimeError("_digammainv: fsolve failed, y = %r" % y)

    return value[0]

def inv_digamma(x, tol=1e-3, max_iterations=10, init_strategy="batir"):

    """Solve for inverse digamma. Psi(x) = 0.

    Parameters
    ----------

    x: float

    tol: float
         Tolerance criterion for stopping
         iterations when difference between sucessive estimates
         drops below this value.

    max_iterations: int
                    Maximum number of iterations.
    init_strategy: str
                   One of ["minka", "batir"]. Default: "batir"
                   For Minka initalization see

    References
    ----------

    1. http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/
    2. https://arxiv.org/pdf/1705.06547.pdf (Batir et al. initialization)
    """
    gamma = -0.5772156649015329  # digamma(1)

    x_old = np.piecewise(
        x,
        [x >= -2.22, x < -2.22],
        [(lambda z: np.exp(z) + 0.5), (lambda z: -1 / (z + gamma))],
    )
    for i in range(max_iterations):
        x_new = x_old - (digamma(x_old) - x) / trigamma(x_old)
        if LA.norm(x_new - x_old) < tol:
            return x_new, i + 1
        x_old = x_new
    raise RuntimeError("Failed to converge inv_digamma")


def fastfit_gamma(observed, s=None, tol=1e-3, max_iterations=10):
    """Estimate parameters of a Gamma distribution.

    Parameters
    ----------
    observed: array_like
              Observed values

    s: array_like
       Sufficient statistics. Default: None (inferred from the observations)

    tol: float
         Tolerance criterion for stopping
         iterations when difference between sucessive estimates
         drops below this value.

    max_iterations: int
                    Maximum number of iterations.

    Returns
    -------
    a, b: float
          alpha, beta parameters of Gamma Distribution
          https://en.wikipedia.org/wiki/Gamma_distribution#Characterization_using_shape_%CE%B1_and_rate_%CE%B2
    iterations: int
                Number of iterations before convergence
    """
    xbar = np.mean(observed)
    log_xbar = np.log(xbar)
    bar_logx = np.mean(np.log(observed))
    if s is None:
        s = log_xbar - bar_logx
    a = 0.5 / s

    a_old = np.inf
    iteration = 0
    while (np.abs(a_old - a) > tol) and (iteration < max_iterations):
        a_old = a
        #a, _ = inv_digamma(np.log(a) - s)
        sa = np.log(a) - s
        a = _digammainv(sa)# for x in sa]
        iteration += 1

    iteration = 0
    while (np.abs(a_old - a) > tol) and (iteration < max_iterations):
        a_old = a
        num = np.log(a) - s - digamma(a)
        denom = 1 / a - trigamma(a)
        a = 1 / (1 / a + num / (a ** 2 * denom))
        iteration += 1
    b = xbar / a
    return a, b, iteration


def fastfit_negbin(observed, tol=1e-4, max_iterations=100):
    """Fit Negatove Binomial.

    Parameters
    ----------
    observed: array_like
              Observed values

    tol: float
         Tolerance criterion for stopping
         iterations when difference between sucessive estimates
         drops below this value.

    max_iterations: int
                    Maximum number of iterations.

    Returns
    -------
    param_dict: dict
                Dict with inferred parameters and number of iterations
    """

    xmean = np.mean(observed)

    xvar = np.var(observed)
    b = xmean / xvar - 1

    if b < 0:
        b = 1e-3

    a = xmean / b
    a_old = np.inf
    iteration = 0
    while (np.abs(a_old - a) > tol) and (iteration < max_iterations):
        a_old = a
        p = b / (b + 1)
        # sufficient stat
        obsa = observed + a
        xmean = np.mean((obsa) * p)
        s = np.log(xmean) - np.mean(digamma(obsa) + np.log(p))
        a, b, _ = fastfit_gamma(xmean, s, tol=tol)
        iteration += 1
    return {
        "theta": a,
        "dispersion": 1/a,
        "mu": a*b,
        "iterations": iteration
    }
