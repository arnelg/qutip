import numpy as np
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.spmatfuncs cimport (spmv_csr)

include "parameters.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def _smesolve_single_trajectory(int n, object sso):
    """
    Internal function. See smesolve.
    """
    cdef double dt = sso.dt
    cdef np.ndarray[double, ndim=2] times = sso.times
    cdef int d2_len = sso.d2_len
    cdef int N_substeps = sso.N_substeps
    cdef int N_store = sso.N_store
    cdef complex s
    cdef object A_ops = sso.A_ops

    cdef np.ndarray[complex, ndim=1] rho_t = mat2vec(sso.state0.full()).ravel()
    dims = sso.state0.dims

    cdef np.ndarray[complex, ndim=2] expect = np.zeros((len(sso.e_ops),
                                            sso.N_store), dtype=complex)
    cdef np.ndarray[complex, ndim=2] ss = np.zeros((len(sso.e_ops),
                                            sso.N_store), dtype=complex)

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * np.random.randint(0, 4294967295 // (sso.ntraj+1)))

    if sso.noise is None:
        if sso.generate_noise:
            dW = sso.generate_noise(len(A_ops), N_store, N_substeps,
                                    sso.d2_len, dt)
        elif sso.homogeneous:
            dW = np.sqrt(dt) * np.random.randn(len(A_ops), N_store,
                                               N_substeps, d2_len)
        else:
            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = sso.noise[n]

    states_list = []
    cdef np.ndarray[complex, ndim=3] measurements = np.zeros((len(times),
                            len(sso.s_m_ops), d2_len), dtype=complex)

    if(sso.rhs in (10, 20, 21, 22)):
        A_data = A[0][0].data
        A_ind = A[0][0].indices
        A_ptr = A[0][0].indptr
        A_size = A[0][1]
        A_len = len(A)


    for t_idx, t in enumerate(times):

        if sso.s_e_ops:
            for e_idx, e in enumerate(sso.s_e_ops):
                s = cy_expect_rho_vec(e.data, rho_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2

        if sso.store_states or not sso.s_e_ops:
            states_list.append(Qobj(vec2mat(rho_t), dims=dims))

        cdef np.ndarray[complex, ndim=1] rho_prev = np.copy(rho_t)

        if sso.rhs == 10:
            for j in range(N_substeps):
                rho_t = _rhs_rho_euler_homodyne_fast(rho_t, A_data,
                                A_ind, A_ptr, A_size,
                                dW[:, t_idx, j, 0])
        elif sso.rhs == 20:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_single_fast(rho_t, A_data,
                                A_ind, A_ptr, A_size,
                                dW[:, t_idx, j, 0])
        elif sso.rhs == 21:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_two_fast(rho_t, A_data,
                                A_ind, A_ptr, A_size,
                                dW[:, t_idx, j, 0])
        elif sso.rhs == 22:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_fast(rho_t, A_data,
                                A_ind, A_ptr, A_size, A_len,
                                dW[:, t_idx, j, 0])




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
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    Fast Euler-Maruyama for homodyne detection.
    """
    cdef np.ndarray[complex, ndim=2] d_vec =
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e =
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2)

    drho_t = d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    drho_t += (1.0 - np.inner(np.real(e), dW)) * rho_t
    return drho_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_single_fast(complex[::1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    Fast Milstein for homodyne detection with 1 stochastic operator
    """
    cdef np.ndarray[complex, ndim=2] _rhs_rho_milstein_homodyne_two_fastd_vec =
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(
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
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    cdef np.ndarray[complex, ndim=2] d_vec =
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(
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
                                np.ndarray[double, ndim=1] dW):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    cdef int sc2_len = 2 * sc_len

    cdef np.ndarray[complex, ndim=2] d_vec =
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(d_vec[:-1].reshape(
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
