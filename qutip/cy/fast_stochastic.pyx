import numpy as np
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.spmatfuncs cimport (spmv_csr)

include "parameters.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def _smesolve_single_trajectory(n, sso):
    """
    Internal function. See smesolve.
    """
    dt = sso.dt
    times = sso.times
    d1, d2 = sso.d1, sso.d2
    d2_len = sso.d2_len
    L_data = sso.L.data
    N_substeps = sso.N_substeps
    N_store = sso.N_store
    A_ops = sso.A_ops

    rho_t = mat2vec(sso.state0.full()).ravel()
    dims = sso.state0.dims

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * np.random.randint(0, 4294967295 // (sso.ntraj+1)))

    if sso.noise is None:
        if sso.generate_noise:
            dW = sso.generate_noise(len(A_ops), N_store, N_substeps,
                                    sso.d2_len, dt)
        elif sso.homogeneous:
            #if sso.distribution == 'normal':
            dW = np.sqrt(dt) * np.random.randn(len(A_ops), N_store,
                                               N_substeps, d2_len)
            #else:
            #    raise TypeError('Unsupported increment distribution for ' +
            #                    'homogeneous process.')
        else:
            #if sso.distribution != 'poisson':
            #    raise TypeError('Unsupported increment distribution for ' +
            #                    'inhomogeneous process.')

            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = sso.noise[n]

    states_list = []
    measurements = np.zeros((len(times), len(sso.s_m_ops), d2_len),
                            dtype=complex)

    for t_idx, t in enumerate(times):

        if sso.s_e_ops:
            for e_idx, e in enumerate(sso.s_e_ops):
                s = cy_expect_rho_vec(e.data, rho_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2

        if sso.store_states or not sso.s_e_ops:
            states_list.append(Qobj(vec2mat(rho_t), dims=dims))

        rho_prev = np.copy(rho_t)

        for j in range(N_substeps):

            if sso.noise is None and not sso.homogeneous:
                for a_idx, A in enumerate(A_ops):
                    dw_expect = cy_expect_rho_vec(A[4], rho_t, 1) * dt
                    if dw_expect > 0:
                        dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect,
                                                                   d2_len)
                    else:
                        dW[a_idx, t_idx, j, :] = np.zeros(d2_len)

            rho_t = sso.rhs(L_data, rho_t, t + dt * j,
                            A_ops, dt, dW[:, t_idx, j, :], d1, d2, sso.args)

        if sso.store_measurement:
            for m_idx, m in enumerate(sso.s_m_ops):
                for dW_idx, dW_factor in enumerate(sso.dW_factors):
                    if m[dW_idx]:
                        m_expt = cy_expect_rho_vec(m[dW_idx].data, rho_prev, 0)
                    else:
                        m_expt = 0
                    measurements[t_idx, m_idx, dW_idx] = m_expt + dW_factor * \
                        dW[m_idx, t_idx, :, dW_idx].sum() / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(2))

    return states_list, dW, measurements, expect, ss



@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rhs_rho_euler_homodyne_fast(complex[::1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, 
                                int A_size, int A_len,
                                np.ndarray[double, ndim=2] ddW):
    """
    Fast Euler-Maruyama for homodyne detection.
    """

    np.ndarray[double, ndim=1] dW = ddW[:, 0]

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    e = d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2)

    drho_t = d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    drho_t += (1.0 - np.inner(np.real(e), dW)) * rho_t
    return drho_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_single_fast(complex[::1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, 
                                int A_size, int A_len,
                                np.ndarray[double, ndim=2] ddW):
    """
    Fast Milstein for homodyne detection with 1 stochastic operator
    """
    np.ndarray[double, ndim=1] dW = np.copy(ddW[:, 0])

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))
    e[1] -= 2.0 * e[0] * e[0]

    drho_t = - np.inner(e, dW) * rho_t
    dW[0] -= 2.0 * e[0] * dW[1]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_two_fast(complex[::1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, 
                                int A_size, int A_len,
                                np.ndarray[double, ndim=2] ddW):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    np.ndarray[double, ndim=1] dW = np.copy(ddW[:, 0])

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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_fast(complex[::1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr, 
                                int A_size, int sc_len,
                                np.ndarray[double, ndim=2] ddW):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    cdef np.ndarray[double, ndim=1] dW = np.copy(ddW[:, 0])
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
