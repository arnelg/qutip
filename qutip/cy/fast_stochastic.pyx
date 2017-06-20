import numpy as np
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.spmatfuncs cimport (spmv_csr,
                                  cy_expect_rho_vec_csr, cy_expect_psi_csr)

include "parameters.pxi"


def _rhs_rho_euler_homodyne_fast(complex[::1] rho_t, complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, int A_size
                                np.ndarray[double, ndim=2] ddW):
    """
    Fast Euler-Maruyama for homodyne detection.
    """

    dW = ddW[:, 0]

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    e = d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2)

    drho_t = d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    drho_t += (1.0 - np.inner(np.real(e), dW)) * rho_t
    return drho_t


def _rhs_rho_milstein_homodyne_single_fast(complex[::1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, int A_size
                                np.ndarray[double, ndim=2] ddW):
    """
    fast Milstein for homodyne detection with 1 stochastic operator
    """
    dW = np.copy(ddW[:, 0])

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))
    e[1] -= 2.0 * e[0] * e[0]

    drho_t = - np.inner(e, dW) * rho_t
    dW[0] -= 2.0 * e[0] * dW[1]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t


def _rhs_rho_milstein_homodyne_two_fast(complex[::1] rho_t, complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, int A_size
                                np.ndarray[double, ndim=2] ddW):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    dW = np.copy(ddW[:, 0])

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))
    d_vec[-2] -= np.dot(e[:2][::-1], d_vec[:2])

    e[2:4] -= 2.0 * e[:2] * e[:2]
    e[4] -= 2.0 * e[1] * e[0]

    drho_t = - np.inner(e, dW) * rho_t
    dW[:2] -= 2.0 * e[:2] * dW[2:4]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t


def _rhs_rho_milstein_homodyne_fast(complex[::1] rho_t, complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, int A_size
                                np.ndarray[double, ndim=2] ddW):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    dW = np.copy(ddW[:, 0])
    sc_len = len(A)
    sc2_len = 2 * sc_len

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    e = np.real(d_vec[:-1].reshape(
        -1, A_size, A_size).trace(axis1=1, axis2=2))
    d_vec[sc2_len:-1] -= np.array(
        [e[m] * d_vec[n] + e[n] * d_vec[m]
         for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    e[sc_len:sc2_len] -= 2.0 * e[:sc_len] * e[:sc_len]
    e[sc2_len:] -= 2.0 * np.array(
        [e[n] * e[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    drho_t = - np.inner(e, dW) * rho_t
    dW[:sc_len] -= 2.0 * e[:sc_len] * dW[sc_len:sc2_len]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t
