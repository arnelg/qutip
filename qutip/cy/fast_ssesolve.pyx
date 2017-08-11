import numpy as np
import scipy.sparse as sp
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.spmatfuncs cimport (spmv_csr)
import time

from qutip.qobj import Qobj
from qutip.td_qobj import td_Qobj

include "parameters.pxi"

from qutip.cy.spmatfuncs import cy_expect_rho_vec_csr, cy_expect_rho_vec, cy_expect_psi_csr, cy_expect_psi
#from qutip.cy.fast_stochastic import _generate_noise_Milstein, _generate_noise_Taylor_15
from qutip.superoperator import mat2vec, vec2mat

from scipy.linalg.cython_blas cimport dznrm2 as raw_dznrm2

cdef int ONE=1;

cdef double dznrm2(complex[::1] psi, int l):
    return raw_dznrm2(&l,<complex*>&psi[0],&ONE)

cdef extern from "Python.h":
    void* PyLong_AsVoidPtr(object)

def cy_ssesolve_single_trajectory(n, sso):
    """
    Internal function. See ssesolve.
    """
    cdef double dt = sso.dt, t
    cdef int d2_len = sso.d2_len, N_store = sso.N_store
    cdef int i, t_idx, j, N_substeps = sso.nsubsteps
    cdef double[:] times = sso.times
    cdef double[:,:,::1] dW
    e_ops = sso.e_ops

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    psi_t = sso.state0.full().ravel()
    dims = sso.state0.dims

    cdef int psi_len = len(psi_t)

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * (sso.ntraj + 11) +
                   np.random.randint(0, 4294967295 // (sso.ntraj + 1)))

    poisson = False
    if sso.noise is not None:
        dW = sso.noise[n]
    elif sso.generate_noise:
        dW = sso.generate_noise(N_store, N_substeps, d2_len, 1, dt)

    elif sso.homogeneous:
        dW = np.sqrt(dt) * np.random.randn(N_store, N_substeps, d2_len)
    else:
        poisson = True
        dW = np.zeros((N_store, N_substeps, d2_len))

    states_list = []
    measurements = np.zeros((len(times), len(sso.m_ops)),
                            dtype=complex)

    cdef void* rhs_ptr
    cdef long[:] expect_ptr = np.empty(d2_len, dtype=long)
    use_ptr = False
    if sso.rhs in (10, 30): #Euler-maruyama
      if use_ptr:
        rhs_ptr = PyLong_AsVoidPtr(sso.A_td_ops[0].get_rhs_ptr())
        for i, sc in enumerate(sso.A_td_ops[1]):
            expect_ptr[i] = (sc.get_rhs_ptr())
      else:
        A_rhs = sso.A_td_ops[0].get_rhs_func()
        A_expect = []
        for i, sc in enumerate(sso.A_td_ops[1]):
            A_expect += [sc.get_expect_func()]


    for t_idx in range(N_store):
        t = times[t_idx]
        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_psi_csr(e.data.data,
                                      e.data.indices,
                                      e.data.indptr, psi_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t, dims=dims))



        if sso.rhs == 10: #Euler-maruyama
            for j in range(N_substeps):
                cy_sse_euler_homodyne(psi_t, t, dt, dW[t_idx, j, :],
                                              A_rhs, A_expect,
                                              d2_len, psi_len)

        elif sso.rhs == 30: #Euler-maruyama
            for j in range(N_substeps):
                cy_sse_platen_homodyne(psi_t, t, dt, dW[t_idx, j, :],
                                              A_rhs, A_expect,
                                              d2_len, psi_len)
        else:
            raise NotImplementedError("")


        if sso.normalize:
            normalize(psi_t, psi_len)
            #psi_t = psi_t / dznrm2(psi_t, psi_len)
        #print(psi_t)

        if sso.store_measurement:
            for idx, (m, dW_factor) in enumerate(zip(sso.m_ops,
                                                     sso.dW_factors)):
                if m:
                    if sso.td[1]:
                        m_expt = cy_expect_psi(m(t).data, psi_t, 0)
                    else:
                        m_expt = cy_expect_psi(m.data, psi_t, 0)
                else:
                    m_expt = 0
                measurements[t_idx, idx] = m_expt + dW_factor * \
                        np.sum(dW[t_idx, :, idx]) / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(1))

    return states_list, dW, measurements, expect, ss

"""
cdef void cy_sse_euler_homodyne_ptr(complex[::1] psi, double t, double dt,
                                double [::1] dW, void* A_prod,
                                void* [:] A_expect, int N_sc_ops, int psi_len):

    #cdef void (*f_c)(double, complex*, complex*)
    #cdef complex (*f_c)(double, complex*, int)
    cdef complex e1, f0, fpsi
    cdef int i, j
    cdef complex[:,:] chi = A_prod(t, psi).reshape(psi_len, 2*N_sc_ops+1)
    cdef complex[:] d_psi = np.empty(psi_len, dtype=complex)

    for i in range(psi_len):
        d_psi[i] = chi[i,2*N_sc_ops]

    for j in range(N_sc_ops):
        e1 = A_expect[j](t,psi)
        f0 = e1*0.5*dt + dW[j]
        fpsi = e1*e1*1.25*dt + e1*0.5*dW[j]
        for i in range(psi_len):
            d_psi[i] += chi[i,2*j]*f0-chi[i,2*j+1]-psi[i]*fpsi

    for i in range(psi_len):
        psi[i] += d_psi[i]"""

# Cython version of the solvers
cdef void cy_sse_euler_homodyne(np.ndarray[complex, ndim=1] psi, double t,
                                double dt, double [::1] dW, object A_prod,
                                object A_expect, int N_sc_ops, int psi_len):

    cdef complex e1, f0, fpsi
    cdef int i, j
    cdef complex[:,:] chi = A_prod(t, psi).reshape(2*N_sc_ops+1, psi_len)
    cdef complex[:] d_psi = np.empty(psi_len, dtype=complex)

    for i in range(psi_len):
        d_psi[i] = chi[2*N_sc_ops,i]

    for j in range(N_sc_ops):
        e1 = A_expect[j](t,psi,0)
        f0 = e1*0.5*dt + dW[j]
        fpsi = e1*e1*0.125*dt + e1*0.5*dW[j]
        #print(psi, chi[0,2*j+1], chi[1,2*j+1])
        for i in range(psi_len):
            d_psi[i] += (chi[2*j,i]*f0-chi[2*j+1,i]*dt*.5-psi[i]*fpsi)

    for i in range(psi_len):
        psi[i] += d_psi[i]


cdef void cy_sse_platen_homodyne(np.ndarray[complex, ndim=1] psi, double t,
                                 double dt, double [::1] dW, object A_prod,
                                 object A_expect, int N_sc_ops, int psi_len):
    cdef complex dt_2, sqrt_dt, d1, d2 ,dWW
    cdef int i, j

    cdef complex[:,:] chi = A_prod(t, psi).reshape(2*N_sc_ops+1, psi_len)
    cdef complex[:] d_psi = np.empty(psi_len, dtype=complex)

    cdef complex[:] dpsi_t = np.zeros(psi_len, dtype=complex)
    cdef np.ndarray[complex, ndim=1] psi_t = np.zeros(psi_len, dtype=complex)
    cdef np.ndarray[complex, ndim=1] psi_p = np.zeros(psi_len, dtype=complex)
    cdef np.ndarray[complex, ndim=1] psi_m = np.zeros(psi_len, dtype=complex)

    cdef complex[:] e1 = np.empty(N_sc_ops, dtype=complex)
    cdef complex[:] f0 = np.empty(N_sc_ops, dtype=complex)
    cdef complex[:] fpsi = np.empty(N_sc_ops, dtype=complex)

    dt_2 = dt*.5
    sqrt_dt = np.sqrt(dt)

    for j in range(N_sc_ops):
        e1[j] = A_expect[j](t,psi,0) * 0.5
        f0[j] = e1[j] * dt
        fpsi[j] = e1[j]**2 * dt_2

    for i in range(psi_len):
        d1 = chi[2*N_sc_ops,i]
        for j in range(N_sc_ops):
            d1 += chi[2*j,i]*f0[j] - chi[2*j+1,i]*dt_2 - psi[i]*fpsi[j]
            d2 = chi[2*j,i] - psi[i]*e1[j]
            psi_t[i] = d2 * dW[j]
            psi_p[i] = d2 * sqrt_dt
            psi_m[i] = -d2 * sqrt_dt
        d_psi[i] = (d1 + psi_t[i]) * 0.5
        psi_p[i] += d1 + psi[i]
        psi_m[i] += d1 + psi[i]
        psi_t[i] += d1 + psi[i]

    chi = A_prod(t, psi_t).reshape(2*N_sc_ops+1, psi_len)
    dt_2 = dt*0.25

    for i in range(psi_len):
        d_psi[i] += chi[2*N_sc_ops,i]*0.5
        
    for j in range(N_sc_ops):
        e1[j] = A_expect[j](t,psi_t,0) * 0.5
        f0[j] = e1[j] * dt * 0.5
        fpsi[j] = e1[j]**2 * dt_2
        for i in range(psi_len):
            d_psi[i] += chi[2*j,i]*f0[j] - chi[2*j+1,i]*dt_2 - psi_t[i]*fpsi[j]

    chi = A_prod(t, psi_p).reshape(2*N_sc_ops+1, psi_len)
    for j in range(N_sc_ops):
        e1[j] = A_expect[j](t,psi_t,0) * 0.5
        dWW = dW[j]*.25 + (dW[j]*dW[j]-dt)*0.25/sqrt_dt
        for i in range(psi_len):
            d_psi[i] += (chi[2*j,i] - psi_p[i]*e1[j]) * dWW

    chi = A_prod(t, psi_m).reshape(2*N_sc_ops+1, psi_len)
    for j in range(N_sc_ops):
        e1[j] = A_expect[j](t,psi_t,0) * 0.5
        dWW = dW[j]*.25 - (dW[j]*dW[j]-dt)*0.25/sqrt_dt
        for i in range(psi_len):
            d_psi[i] += (chi[2*j,i] - psi_m[i]*e1[j]) * dWW

    for i in range(psi_len):
        psi[i] += d_psi[i]



cdef void normalize(complex [::1] psi, int psi_len):
    cdef double norm = 0.
    cdef int i
    norm = dznrm2(psi,psi_len)
    for i in range(psi_len):
        psi[i] = psi[i]/norm
