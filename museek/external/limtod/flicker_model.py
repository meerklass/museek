import numpy as np
from mpmath import gammainc
from scipy.linalg import toeplitz


def aux_int(mu, u):
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi / 2 * mu
        return float(aux.real) * np.cos(ang) + float(aux.imag) * np.sin(ang)
    except ValueError as e:
        print(f"Error in aux_int with mu={mu}, u={u}: {e}")
        return np.inf  # or some other default value


def flicker_corr(tau, f0, fc, alpha, var_w=0.0):
    """
    Note that f0 and fc are in unit of angular frequency, differently from that of FFT frequency convention by a factor of 2pi.
    """
    if tau == 0:
        return fc / np.pi * (f0 / fc) ** alpha / (alpha - 1) + var_w
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1 / (np.pi * tau)
    mu = 1 - alpha
    result = theta_0**alpha * aux_int(mu, theta_c)
    return result * norm


def sim_noise(f0, fc, alpha, time_list, n_samples=1, white_n_variance=5e-6):
    lags = time_list - time_list[0]
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    covmat = toeplitz(corr_list)
    if n_samples == 1:
        return np.random.multivariate_normal(np.zeros_like(time_list), covmat).reshape(
            1, -1
        )
    else:
        return np.random.multivariate_normal(
            np.zeros_like(time_list), covmat, n_samples
        )
