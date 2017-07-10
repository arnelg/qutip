def _rhs_deterministic(LH, rho_t, t, dt, args):
    """
    Deterministic contribution to the density matrix change
    LH : Louivillian or -iH
    """
    drho_t = spmv(LH, rho_t) * dt

    return drho_t

def _rhs_euler_maruyama(LH, sc_ops, vec_t, t, dt, dW, d1, d2, N_d2, args):
    """
    Euler-Maruyama rhs function for both master eq and schrodinger eq.

    dV = -iH*V*dt + d1*dt + d2_i*dW_i
    """
    dvec_t = _rhs_deterministic(LH, vec_t, t, dt, args)
    dvec_t += d1(t, vec_t, A, args) * dt 
    d2_vec = d2(t, vec_t, sc_ops, args)
    dvec_t += np.dot(dW, d2_vec)
    return vec_t + dvec_t

def _rhs_milstein(LH, sc_ops, vec_t, t, dt, dW, d1, d2, N_d2, args):
    """
    Milstein rhs function for both master eq and schrodinger eq.

    Slow but should be valid for non-commuting operators since computing 
        both i x j and j x i.

    dV = -iH*V*dt + d1*dt + d2_i*dW_i 
         + 0.5*d2_i(d2_j(V))*(dW_i*dw_j -dt*delta_ij)
    """

    #Euler part
    dvec_t = _rhs_deterministic(LH, vec_t, t, dt, args)
    dvec_t += d1(t, vec_t, sc_ops, args) * dt 
    d2_vec = d2(t, vec_t, sc_ops, args)
    dvec_t += np.dot(dW, d2_vec)

    #Milstein terms
    for ind, d2v in enumerate(d2_vec):
        dW2 = dW*dW[ind]*0.5
        dW2[ind] -= dt*0.5
        d22_vec = d2(t, d2v, sc_ops, args)
        dvec_t += np.dot(dW2, d22_vec)

    return vec_t + dvec_t

# -----------------------------------------------------------------------------
# Platen rhs functions for the stochastic master equation
#
def _rhs_platen(LH, sc_ops, vec_t, t, dt, dW, d1, d2, N_d2, args):
    """
    Platen rhs function for both master eq and schrodinger eq.
    
    dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))*2 * dt
         + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
         + (d2_i(V+)+d2_i(V-)) * (dW_i**2 -dt) * dt**(-.5)

    Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
    V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5

    Not validated for time-dependent operators
    """

    sqrt_dt = np.sqrt(dt)

    #Build Vt, V+, V-
    dv_H1 = _rhs_deterministic(LH, vec_t, t, dt, args)
    dv_H1 += d1(t, vec_t, sc_ops, args) * dt 
    d2_vec = d2(t, vec_t, sc_ops, args)
    Vp = vec_t + dv_H1 + np.sum(d2_vec,axis=0)*sqrt_dt
    Vm = vec_t + dv_H1 - np.sum(d2_vec,axis=0)*sqrt_dt
    Vt = vec_t + dv_H1 + np.dot(dW, d2_vec)

    # Platen dV
    dvt_H1 = _rhs_deterministic(LH, Vt, t+dt, dt, args)
    dvt_H1 += d1(t+dt, Vt, sc_ops, args) * dt 
    dvec_t = 0.50 * (dv_H1 + dvt_H1)

    d2_vp += d2(t+dt, Vp, sc_ops, args)
    d2_vm += d2(t+dt, Vm, sc_ops, args)
    dvec_t += np.dot(dW, 0.25*(2 * d2_vec + d2_vp + d2_vm))
    dW2 = (dW**2 - dt) / sqrt_dt
    dvec_t += np.dot(dW, 0.25*(d2_vp - d2_vm))

    return vec_t + dvec_t