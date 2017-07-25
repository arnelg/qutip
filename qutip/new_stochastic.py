# -*- coding: utf-8 -*-
#
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#    Significant parts of this code were contributed by Denis Vasilyev.
#
###############################################################################
"""
This module contains functions for solving stochastic schrodinger and master
equations. The API should not be considered stable, and is subject to change
when we work more on optimizing this module for performance and features.
"""

__all__ = ['new_ssesolve', 'new_smesolve']

import numpy as np
import scipy.sparse as sp
from scipy.linalg.blas import get_blas_funcs
try:
    norm = get_blas_funcs("znrm2", dtype=np.float64)
except:
    from scipy.linalg import norm

from numpy.random import RandomState

from qutip.qobj import Qobj, isket
from qutip.states import ket2dm
from qutip.solver import Result
from qutip.expect import expect, expect_rho_vec
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian, lindblad_dissipator)
from qutip.cy.spmatfuncs import cy_expect_psi_csr, spmv, cy_expect_rho_vec, \
                                cy_expect_psi
from qutip.cy.stochastic import (cy_d1_rho_photocurrent,
                                 cy_d2_rho_photocurrent)
from qutip.cy.fast_stochastic import cy_smesolve_fast_single_trajectory

from qutip.parallel import serial_map
from qutip.ui.progressbar import TextProgressBar
from qutip.solver import Options, _solver_safety_check
from qutip.settings import debug
from qutip.td_qobj import td_liouvillian, td_Qobj


if debug:
    import qutip.logging_utils
    import inspect
    logger = qutip.logging_utils.get_logger()


class StochasticSolverOptions:
    """Class of options for stochastic solvers such as
    :func:`qutip.stochastic.ssesolve`, :func:`qutip.stochastic.smesolve`, etc.
    Options can be specified either as arguments to the constructor::

        sso = StochasticSolverOptions(nsubsteps=100, ...)

    or by changing the class attributes after creation::

        sso = StochasticSolverOptions()
        sso.nsubsteps = 1000

    The stochastic solvers :func:`qutip.stochastic.ssesolve`,
    :func:`qutip.stochastic.smesolve`, :func:`qutip.stochastic.ssepdpsolve` and
    :func:`qutip.stochastic.smepdpsolve` all take the same keyword arguments as
    the constructor of these class, and internally they use these arguments to
    construct an instance of this class, so it is rarely needed to explicitly
    create an instance of this class.

    Attributes
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    state0 : :class:`qutip.Qobj`
        Initial state vector (ket) or density matrix.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        List of deterministic collapse operators.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    m_ops : list of :class:`qutip.Qobj`
        List of operators representing the measurement operators. The expected
        format is a nested list with one measurement operator for each
        stochastic increament, for each stochastic collapse operator.

    args : dict / list
        List of dictionary of additional problem-specific parameters.
        Implicit methods can adjust tolerance via args = {'tol':value}

    ntraj : int
        Number of trajectors.

    nsubsteps : int
        Number of sub steps between each time-spep given in `times`.

    d1 : function
        Function for calculating the operator-valued coefficient to the
        deterministic increment dt.

    d2 : function
        Function for calculating the operator-valued coefficient to the
        stochastic increment(s) dW_n, where n is in [0, d2_len[.

    d2_len : int (default 1)
        The number of stochastic increments in the process.

    dW_factors : array
        Array of length d2_len, containing scaling factors for each
        measurement operator in m_ops.

    rhs : function
        Function for calculating the deterministic and stochastic contributions
        to the right-hand side of the stochastic differential equation. This
        only needs to be specified when implementing a custom SDE solver.

    generate_A_ops : function
        Function that generates a list of pre-computed operators or super-
        operators. These precomputed operators are used in some d1 and d2
        functions.

    generate_noise : function
        Function for generate an array of pre-computed noise signal.

    homogeneous : bool (True)
        Wheter or not the stochastic process is homogenous. Inhomogenous
        processes are only supported for poisson distributions.

    solver : string
        Name of the solver method to use for solving the stochastic
        equations. Valid values are:
        1/2 order algorithms: 'euler-maruyama', 'fast-euler-maruyama',
        'pc-euler' is a predictor-corrector method which is more
        stable than explicit methods,
        1 order algorithms: 'milstein', 'fast-milstein', 'platen',
        'milstein-imp' is semi-implicit Milstein method,
        3/2 order algorithms: 'taylor15',
        'taylor15-imp' is semi-implicit Taylor 1.5 method.
        Implicit methods can adjust tolerance via args = {'tol':value},
        default is {'tol':1e-6}

    method : string ('homodyne', 'heterodyne', 'photocurrent')
        The name of the type of measurement process that give rise to the
        stochastic equation to solve. Specifying a method with this keyword
        argument is a short-hand notation for using pre-defined d1 and d2
        functions for the corresponding stochastic processes.

    distribution : string ('normal', 'poisson')
        The name of the distribution used for the stochastic increments.

    store_measurements : bool (default False)
        Whether or not to store the measurement results in the
        :class:`qutip.solver.SolverResult` instance returned by the solver.

    noise : array
        Vector specifying the noise.

    normalize : bool (default True)
        Whether or not to normalize the wave function during the evolution.

    options : :class:`qutip.solver.Options`
        Generic solver options.

    map_func: function
        A map function or managing the calls to single-trajactory solvers.

    map_kwargs: dictionary
        Optional keyword arguments to the map_func function function.

    progress_bar : :class:`qutip.ui.BaseProgressBar`
        Optional progress bar class instance.

    """
    def __init__(self, H=None, state0=None, times=None, c_ops=[], sc_ops=[],
                 e_ops=[], m_ops=None, args=None, ntraj=1, nsubsteps=1,
                 d1=None, d2=None, d2_len=1, dW_factors=None, rhs=None,
                 generate_A_ops=None, generate_noise=None, homogeneous=True,
                 solver=None, method=None, distribution='normal',
                 store_measurement=False, noise=None, normalize=True,
                 options=None, progress_bar=None, map_func=None,
                 map_kwargs=None):

        if options is None:
            options = Options()

        if progress_bar is None:
            progress_bar = TextProgressBar()

        self.H = H
        self.d1 = d1
        self.d2 = d2
        self.d2_len = d2_len
        self.dW_factors = dW_factors# if dW_factors else np.ones(d2_len)
        self.state0 = state0
        self.times = times
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops

        #if m_ops is None:
        #    self.m_ops = [[c for _ in range(d2_len)] for c in sc_ops]
        #else:
        #    self.m_ops = m_ops

        self.m_ops = m_ops

        self.ntraj = ntraj
        self.nsubsteps = nsubsteps
        self.solver = solver
        self.method = method
        self.distribution = distribution
        self.homogeneous = homogeneous
        self.rhs = rhs
        self.options = options
        self.progress_bar = progress_bar
        self.store_measurement = store_measurement
        self.store_states = options.store_states
        self.noise = noise
        self.args = args
        self.normalize = normalize

        self.generate_noise = generate_noise
        self.generate_A_ops = generate_A_ops

        if self.ntraj > 1 and map_func:
            self.map_func = map_func
        else:
            self.map_func = serial_map

        self.map_kwargs = map_kwargs if map_kwargs is not None else {}

"""        #Does any operator depend on time?
        self.td = False
        if not isinstance(H, Qobj):
            self.td = True
        for ops in c_ops:
            if not isinstance(ops, Qobj):
                self.td = True
        for ops in sc_ops:
            if not isinstance(ops, Qobj):
                self.td = True"""

def make_d1d2_se(sso):
    if not sso.method in [None, 'homodyne', 'heterodyne', 'photocurrent']:
        raise Exception("The method should be one of "+\
                            "[None, 'homodyne', 'heterodyne', 'photocurrent']")
    if sso.method == 'homodyne' or sso.method is None:
        sso.LH, sso.A_ops = prep_sc_ops_homodyne_psi(sso.H,
                                    sso.c_ops, sso.sc_ops, sso.dt, sso.td)
        sso.d1 = d1_psi_homodyne
        sso.d2 = d2_psi_homodyne
        sso.d2_len = len(sso.sc_ops)
        sso.homogeneous = True
        sso.distribution = 'normal'
        if not sso.dW_factors:
            sso.dW_factors = np.array([1.]*sso.d2_len)
        if not sso.m_ops:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [(c + c.dag())]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [(td_c + td_c.dag())]

    elif sso.method == 'heterodyne':
        sso.LH, sso.A_ops = prep_sc_ops_heterodyne_psi(sso.H,
                                    sso.c_ops, sso.sc_ops, sso.dt, sso.td)
        sso.d1 = d1_psi_heterodyne
        sso.d2 = d2_psi_heterodyne
        sso.d2_len = 2*len(sso.sc_ops)
        sso.homogeneous = True
        sso.distribution = 'normal'
        sc_ops_heterodyne = []
        for sc in iter(sso.sc_ops):
            if isinstance(sc, Qobj):
                sc_ops_heterodyne += [sc / np.sqrt(2), -1.0j * sc / np.sqrt(2)]
            elif isinstance(sc, list):
                sc_ops_heterodyne += [[sc[0] / np.sqrt(2), sc[1]],
                                      [-1.0j * sc[0] / np.sqrt(2), sc[1]]]

        if not sso.dW_factors:
            sso.dW_factors = np.array([np.sqrt(2)]*sso.d2_len)
        else:
            if len(sso.dW_factors) == len(sso.sc_ops):
                dwf = []
                for f in dW_factors:
                    dwf += [f*np.sqrt(2), f*np.sqrt(2)]
                sso.dW_factors = np.array(dwf)

        if not sso.m_ops:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [(c + c.dag()), -1j * (c - c.dag()) ]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [(td_c + td_c.dag()),
                                  -1j * (td_c - td_c.dag()) ]

        sso.sc_ops = sc_ops_heterodyne

    elif sso.method == 'photocurrent':
        sso.LH, sso.A_ops =  prep_sc_ops_photocurrent_psi(sso.H,
                                    sso.c_ops, sso.sc_ops, sso.dt, sso.td)
        if any(sso.td):
            sso.d1 = d1_psi_photocurrent
            sso.d2 = d2_psi_photocurrent
        sso.d2_len = len(sso.sc_ops)
        sso.homogeneous = False
        sso.distribution = 'poisson'
        if not sso.dW_factors:
            sso.dW_factors = np.array([1.]*sso.d2_len)
        if not sso.m_ops:
            sso.m_ops = [None for c in sso.sc_ops]
        if not sso.m_ops:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [c]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [td_c]

def make_d1d2_me(sso):
    if not sso.method in [None, 'homodyne', 'heterodyne', 'photocurrent']:
        raise Exception("The method should be one of "+\
                            "[None, 'homodyne', 'heterodyne', 'photocurrent']")
    if sso.method == 'homodyne' or sso.method is None:
        sso.LH, sso.A_ops = prep_sc_ops_homodyne_rho(sso.H,
                                    sso.c_ops, sso.sc_ops, sso.dt, sso.td)
        sso.d1 = d1_rho
        sso.d2 = d2_rho
        sso.d2_len = len(sso.sc_ops)
        sso.homogeneous = True
        sso.distribution = 'normal'
        if not sso.dW_factors:
            sso.dW_factors = np.array([1.]*sso.d2_len)
        if not sso.m_ops:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [spre(c + c.dag())]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [(td_c + td_c.dag()).apply(spre)]

    elif sso.method == 'heterodyne':
        sso.LH, sso.A_ops = prep_sc_ops_heterodyne_rho(sso.H,
                                    sso.c_ops, sso.sc_ops, sso.dt, sso.td)
        sso.d1 = d1_rho
        sso.d2 = d2_rho
        sso.d2_len = 2*len(sso.sc_ops)
        sso.homogeneous = True
        sso.distribution = 'normal'
        sc_ops_heterodyne = []
        for sc in iter(sso.sc_ops):
            sc_ops_heterodyne += [sc / np.sqrt(2), -1.0j * sc / np.sqrt(2)]
        if not sso.dW_factors:
            sso.dW_factors = np.array([np.sqrt(2)]*sso.d2_len)
        else:
            if len(sso.dW_factors) == len(sso.sc_ops):
                dwf = []
                for f in dW_factors:
                    dwf += [f*np.sqrt(2), f*np.sqrt(2)]
                sso.dW_factors = np.array(dwf)

        if not sso.m_ops:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [spre(c + c.dag()), -1j * spre(c - c.dag()) ]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [(td_c + td_c.dag()).apply(spre),
                                  -1j * (td_c - td_c.dag()).apply(spre) ]

        sso.sc_ops = sc_ops_heterodyne

    elif sso.method == 'photocurrent':
        sso.LH, sso.A_ops =  prep_sc_ops_photocurrent_rho(sso.H,
                                    sso.c_ops, sso.sc_ops, sso.dt, sso.td)
        if any(sso.td):
            sso.d1 = d1_rho_photocurrent
            sso.d2 = d2_rho_photocurrent
        else :
            sso.d1 = cy_d1_rho_photocurrent
            sso.d2 = cy_d2_rho_photocurrent
        sso.d2_len = len(sso.sc_ops)
        sso.homogeneous = False
        sso.distribution = 'poisson'
        if not sso.dW_factors:
            sso.dW_factors = np.array([1.]*sso.d2_len)
        if not sso.m_ops:
            sso.m_ops = [None for c in sso.sc_ops]
        if not sso.m_ops:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [c]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [td_c.apply(spre)]

def new_ssesolve(H, psi0, times, sc_ops=[], e_ops=[],
                 _safe_mode=True, **kwargs):
    """
    Solve the stochastic Schrödinger equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`
        An instance of the class :class:`qutip.solver.SolverResult`.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if _safe_mode:
        _solver_safety_check(H, psi0, sc_ops, e_ops)

    sso = StochasticSolverOptions(H=H, state0=psi0, times=times,
                                  sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    sso.me = False
    sso.dt = (times[1] - times[0]) / sso.nsubsteps

    #Is any of the rhs, (d1,d2), noise supplied?
    sso.custom = [False, False, False, False]
    if sso.rhs:
        sso.custom[0] = True
    if sso.d1 or sso.d2:
        if sso.d1 and sso.d2:
            sso.custom[1] = True
        else:
            raise Exception("Must define both d1 and d2 or none of them")
    if sso.generate_noise:
        sso.custom[2] = True
    if sso.noise:
        sso.custom[3] = True

    sso.td = [False, False]
    if not isinstance(H, Qobj):
        sso.td[0] = True
    for ops in sc_ops:
        if not isinstance(ops, Qobj):
            sso.td[1] = True

    if not sso.custom[1]:
        make_d1d2_se(sso)
    else:
        sso.A_ops = sc_ops
        sso.d2_len = len(sso.sc_ops)
        if any(sso.td):
            sso.LH = td_Qobj(H)
        else:
            sso.LH = H
            #sso.H_rhs = _rhs_deterministic
        if sso.dW_factors is None:
            sso.dW_factors =  np.ones(d2_len)
        if sso.m_ops is None:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [c]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [td_c]

    if sso.solver == 'euler-maruyama' or sso.solver is None:
        sso.rhs = _rhs_euler_maruyama

    elif sso.method == 'photocurrent':
        raise Exception("Only euler-maruyama supports photocurrent")

    elif sso.solver == 'platen':
        sso.rhs = _rhs_platen

    elif sso.method == 'heterodyne':
        raise Exception("Milstein do not supports heterodyne")

    elif sso.solver == 'milstein':
        sso.rhs = _rhs_milstein

    else:
        raise Exception("Unrecognized solver '%s'." % sso.solver)

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res

def new_smesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[], _safe_mode=True, debug=False, **kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.

    TODO
    ----
        Add checks for commuting jump operators in Milstein method.
    """

    if debug:
        logger.debug(inspect.stack()[0][3])

    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if _safe_mode:
        _solver_safety_check(H, rho0, c_ops+sc_ops, e_ops)

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times, c_ops=c_ops,
                                  sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    sso.me = True
    sso.dt = (times[1] - times[0]) / sso.nsubsteps

    #Is any of the rhs, (d1,d2), noise supplied?
    sso.custom = [False, False, False, False]
    if sso.rhs:
        sso.custom[0] = True
    if sso.d1 or sso.d2:
        if sso.d1 and sso.d2:
            sso.custom[1] = True
        else:
            raise Exception("Must define both d1 and d2 or none of them")
    if sso.generate_noise:
        sso.custom[2] = True
    if sso.noise:
        sso.custom[3] = True

    #Does any operator depend on time?
    #Now sc_ops = [[a],[a]] would be td
    sso.td = [False, False]
    if not isinstance(H, Qobj):
        sso.td[0] = True
    for ops in c_ops:
        if not isinstance(ops, Qobj):
            sso.td[0] = True
    for ops in sc_ops:
        if not isinstance(ops, Qobj):
            sso.td[1] = True
    #Set default d1,d2 if not supplied
    if not sso.custom[1]:
        make_d1d2_me(sso)
    else:
        sso.A_ops = sc_ops
        sso.d2_len = len(sso.sc_ops)
        if any(sso.td):
            sso.LH = td_liouvillian(H, c_ops = c_ops)
        else:
            sso.LH = liouvillian(H, c_ops = c_ops)
            #sso.H_rhs = _rhs_deterministic
        if sso.dW_factors is None:
            sso.dW_factors =  np.ones(d2_len)
        if sso.m_ops is None:
            sso.m_ops = []
            if not sso.td[1]:
                for c in sso.sc_ops:
                    sso.m_ops += [spre(c)]
            else:
                for c in sso.sc_ops:
                    td_c = td_Qobj(c)
                    sso.m_ops += [td_c.apply(spre)]

    # Priority for the noise: sso.generate_noise > sso.distribution
    # Make sure sso.distribution is always present and valid
    if sso.distribution in [None, 'normal']:
        sso.distribution = 'normal'
        sso.homogeneous = True
    elif sso.distribution == 'poisson':
        sso.homogeneous = False
    else:
        raise Exception("The distribution should be one of "+\
                        "[None, 'normal', 'poisson']")

    fast = not (any(sso.custom[:3]) or any(sso.td) or
                sso.distribution == 'poisson')

    #Set the sso.rhs based on the method
    #sso.rhs is an int for fast (cython) code
    if sso.rhs:
        #user has it own rhs
        pass

    else:
        if sso.solver == 'euler-maruyama' or sso.solver is None:
            if fast and sso.method == 'homodyne':
                sso.rhs = 10
                sso.generate_A_ops = _generate_A_ops_Euler
            else :
                sso.rhs = _rhs_euler_maruyama

        elif sso.method == 'photocurrent':
            raise Exception("Only 'euler-maruyama' supports 'photocurrent'")

        elif sso.solver == 'platen':
            sso.rhs = _rhs_platen

        elif sso.custom[1]: # Custom d1, d2
            raise Exception("Only 'euler-maruyama' and " +
                            "'platen' support custom d1,d2")

        elif sso.solver == 'milstein':
            if fast:
                sso.generate_A_ops = _generate_A_ops_Milstein
                sso.generate_noise = 20
                if sso.method == 'homodyne':
                    if len(sc_ops) == 1:
                        sso.rhs = 20
                    elif len(sc_ops) == 2:
                        sso.rhs = 21
                    else:
                        sso.rhs = 22

                elif sso.method == 'heterodyne':
                    sso.d2_len = 1
                    sso.sc_ops = []
                    for sc in iter(sc_ops):
                        sso.sc_ops += [sc / np.sqrt(2), -1.0j * sc / np.sqrt(2)]
                    if len(sc_ops) == 1:
                        sso.rhs = 21
                    else:
                        sso.rhs = 22
            else :
                sso.rhs = _rhs_milstein
                sso.d2 = d2_rho_milstein

        elif sso.td:
            raise Exception("Only 'euler-maruyama', 'milstein' and 'platen' support time dependant cases")
        elif sso.custom[2]: # Custom noise function
            raise Exception("Only 'euler-maruyama', 'milstein' and 'platen' support custom noise function")
        elif not sso.method == 'homodyne': # Not yet done
            raise Exception("Only 'euler-maruyama', 'milstein' and 'platen' support heterodyne")

        elif sso.solver == 'milstein-imp':
            sso.generate_A_ops = _generate_A_ops_implicit
            sso.generate_noise = 20
            if sso.args == None or 'tol' in sso.args:
                sso.args = {'tol':1e-6}
            if sso.method == 'homodyne':
                if len(sc_ops) == 1:
                    sso.rhs = 25
                else:
                    raise Exception("'milstein-imp' : Only one stochastic operator is supported")
            else:
                raise Exception("'milstein-imp' : Only homodyne is available")

        elif sso.solver == 'taylor15':
            sso.generate_A_ops = _generate_A_ops_simple
            sso.generate_noise = 30
            if sso.method == 'homodyne':
                if len(sc_ops) == 1:
                    sso.rhs = 30
                else:
                    raise Exception("'taylor15' : Only one stochastic operator is supported")
            else:
                raise Exception("'taylor15' : Only homodyne is available")

        elif sso.solver == 'taylor15-imp':
            sso.generate_A_ops = _generate_A_ops_implicit
            sso.generate_noise = 30
            if sso.args == None or 'tol' in sso.args:
                sso.args = {'tol':1e-6}
            if sso.method == 'homodyne':
                if len(sc_ops) == 1:
                    sso.rhs = 35
                else:
                    raise Exception("'taylor15-imp' : Only one stochastic operator is supported")
            else:
                raise Exception("'taylor15-imp' : Only homodyne is available")

        elif sso.solver == 'pc-euler':
            sso.generate_A_ops = _generate_A_ops_Milstein
            sso.generate_noise = 20
            if sso.method == 'homodyne':
                if len(sc_ops) == 1:
                    sso.rhs = 40
                else:
                    raise Exception("'pc-euler' : Only one stochastic operator is supported")
            else:
                raise Exception("'pc-euler' : Only homodyne is available")

        if sso.rhs is None:
            raise Exception("The solver should be one of "+\
                            "[None, 'euler-maruyama', "+\
                            "'milstein', 'platen', 'taylor15', "+\
                            "'milstein-imp', 'taylor15-imp', 'pc-euler']")

    if debug:
        return sso
    if isinstance(sso.rhs, int):
        res = _smesolve_fast(sso, sso.options, sso.progress_bar)
    else:
        res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def _sesolve_generic(sso, options, progress_bar):
    """
    Internal function. See smesolve.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    sso.N_store = len(sso.times)
    sso.dt = (sso.times[1] - sso.times[0]) / sso.nsubsteps
    nt = sso.ntraj
    sso.s_m_ops = sso.m_ops

    data = Result()
    data.times = sso.times
    data.expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    data.ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    data.noise = []
    data.measurement = []

    if sso.me:
        data.solver = "smesolve"
        # Master equation
        task = _smesolve_single_trajectory
        # use .data instead of Qobj ?
        sso.s_e_ops = [spre(e) for e in sso.e_ops]

    else:
        data.solver = "ssesolve"
        # Schrodinger equation
        task = _ssesolve_single_trajectory

    map_kwargs = {'progress_bar': progress_bar}
    map_kwargs.update(sso.map_kwargs)

    task_args = (sso,)
    task_kwargs = {}
    results = sso.map_func(task, list(range(sso.ntraj)),
                           task_args, task_kwargs, **map_kwargs)

    for result in results:
        states_list, dW, m, expect, ss = result
        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)
        data.expect += expect
        data.ss += ss

    if sso.me:
        # average density matrices
        if options.average_states and np.any(data.states):
            data.states = [sum([data.states[mm][n] for mm in range(nt)]).unit()
                           for n in range(len(data.times))]
    else:
        if options.average_states and np.any(data.states):
            data.states = [sum([ket2dm(data.states[mm][n])
                                for mm in range(nt)]).unit()
                           for n in range(len(data.times))]

    # average
    data.expect = data.expect / nt

    # standard error
    if nt > 1:
        data.se = (data.ss - nt * (data.expect ** 2)) / (nt * (nt - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :])
                   if e.isherm else data.expect[n, :]
                   for n, e in enumerate(sso.e_ops)]

    return data

def _smesolve_single_trajectory(n, sso):
    """
    Internal function. See smesolve.
    """
    dt = sso.dt
    times = sso.times
    d1, d2 = sso.d1, sso.d2
    d2_len = sso.d2_len
    L = sso.LH
    N_substeps = sso.nsubsteps
    N_store = sso.N_store
    A_ops = sso.A_ops

    rho_t = mat2vec(sso.state0.full()).ravel()
    dims = sso.state0.dims

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * (sso.ntraj + 11) +
                   np.random.randint(0, 4294967295 // (sso.ntraj + 1)))

    poisson = False
    if sso.noise is not None:
        dW = sso.noise[n]
    elif sso.generate_noise:
        dW = sso.generate_noise(N_store, N_substeps, d2_len, n, dt)
    elif sso.homogeneous:
        dW = np.sqrt(dt) * np.random.randn(N_store, N_substeps, d2_len)
    else:
        poisson = True
        dW = np.zeros((N_store, N_substeps, d2_len))


    states_list = []
    measurements = np.zeros((len(times), d2_len), dtype=complex)

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
            if poisson:
                dW_poisson = np.zeros(d2_len)
                for a_idx, A in enumerate(A_ops):
                    if not sso.td[1]:
                        dw_expect = cy_expect_rho_vec(A[2], rho_t, 1) * dt
                    else:
                        dw_expect = cy_expect_rho_vec(A[2](t + dt * j).data,
                                                      rho_t, 1) * dt
                    if dw_expect > 0:
                        dW_poisson[a_idx] = np.random.poisson(dw_expect, d2_len)
                dW[t_idx, j, :] = dW_poisson

            rho_t = sso.rhs(L, rho_t, t + dt * j, dt, A_ops,
                            dW[t_idx, j, :], d1, d2, sso.args, sso.td)

        if sso.store_measurement:
            for idx, (m, dW_factor) in enumerate(zip(sso.s_m_ops,
                                                     sso.dW_factors)):
                if m:
                    if sso.td[1]:
                        m_expt = cy_expect_rho_vec(m(t).data, rho_prev, 0)
                    else:
                        m_expt = cy_expect_rho_vec(m.data, rho_prev, 0)
                else:
                    m_expt = 0
                measurements[t_idx, idx] = m_expt + dW_factor * \
                        dW[t_idx, :, idx].sum() / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(1))

    return states_list, dW, measurements, expect, ss

def _ssesolve_single_trajectory(n, sso):
    """
    Internal function. See ssesolve.
    """
    dt = sso.dt
    times = sso.times
    d1, d2 = sso.d1, sso.d2
    d2_len = sso.d2_len
    e_ops = sso.e_ops
    H = sso.LH
    N_substeps = sso.nsubsteps
    A_ops = sso.A_ops
    N_store = sso.N_store

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    psi_t = sso.state0.full().ravel()
    dims = sso.state0.dims

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * (sso.ntraj + 11) +
                   np.random.randint(0, 4294967295 // (sso.ntraj + 1)))

    poisson = False
    if sso.noise is not None:
        dW = sso.noise[n]
    elif sso.generate_noise:
        dW = sso.generate_noise(N_store, N_substeps, d2_len, n, dt)
    elif sso.homogeneous:
        dW = np.sqrt(dt) * np.random.randn(N_store, N_substeps, d2_len)
    else:
        poisson = True
        dW = np.zeros((N_store, N_substeps, d2_len))

    states_list = []
    measurements = np.zeros((len(times), len(sso.m_ops)),
                            dtype=complex)

    for t_idx, t in enumerate(times):
        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_psi_csr(e.data.data,
                                      e.data.indices,
                                      e.data.indptr, psi_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t, dims=dims))

        for j in range(N_substeps):
            if poisson:
                dW_poisson = np.zeros(d2_len)
                for a_idx, A in enumerate(A_ops):
                    if not sso.td[1]:
                        dw_expect = cy_expect_psi_vec(A[1], rho_t, 1) * dt
                    else:
                        dw_expect = cy_expect_psi_vec(A[1](t + dt * j).data,
                                                      rho_t, 1) * dt
                    if dw_expect > 0:
                        dW_poisson[a_idx] = np.random.poisson(dw_expect, d2_len)
                dW[t_idx, j, :] = dW_poisson

            psi_t = sso.rhs(H, psi_t, t + dt * j, dt, A_ops,
                            dW[t_idx, j, :], d1, d2, sso.args, sso.td)
            # optionally renormalize the wave function
            if sso.normalize:
                psi_t /= norm(psi_t)

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
                        dW[t_idx, :, idx].sum() / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(1))

    return states_list, dW, measurements, expect, ss

#
#   Hamiltonian deterministic evolution
#
def _rhs_deterministic(H, vec_t, t, dt, args, td):
    """
    Deterministic contribution to the density matrix / Hamiltonian for time-dependent cases
    """
    if td:
        return spmv(H(t).data, vec_t)
    else:
        return spmv(H, vec_t)


#
#   d1 and d2 functions for common schemes (Master Eq)
#
def prep_sc_ops_homodyne_rho(H, c_ops, sc_ops, dt, td):
    if not any(td):
        # No time-dependance
        L = liouvillian(H, c_ops=c_ops).data * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc, data_only=True) * dt
            A += [[spre(sc).data + spost(sc.dag()).data]]

    elif not td[1]:
        # sc_ops do not depend on time
        L = td_liouvillian(H, c_ops=c_ops) * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc) * dt
            A += [[spre(sc).data + spost(sc.dag()).data]]

    else:
        td[0] = True
        L = td_liouvillian(H, c_ops=c_ops) * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc, data_only=True) * dt
            td_sc = td_Qobj(sc)
            A += [[td_sc.apply(spre) + td_sc.dag().apply(spost)]]
    return L, A

def prep_sc_ops_heterodyne_rho(H, c_ops, sc_ops, dt, td):
    if not any(td):
        # No time-dependance
        L = liouvillian(H, c_ops=c_ops).data * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc, data_only=True) * dt
            A += [[ 1.0  / np.sqrt(2) * (spre(sc).data + spost(sc.dag()).data)]]
            A += [[-1.0j / np.sqrt(2) * (spre(sc).data - spost(sc.dag()).data)]]

    elif not td[1]:
        # sc_ops do not depend on time
        L = td_liouvillian(H, c_ops=c_ops) * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc) * dt
            A += [[ 1.0  / np.sqrt(2) * (spre(sc).data + spost(sc.dag()).data)]]
            A += [[-1.0j / np.sqrt(2) * (spre(sc).data - spost(sc.dag()).data)]]

    else:
        td[0] = True
        L = td_liouvillian(H, c_ops=c_ops) * dt
        A = []
        for sc in sc_ops:
            L += td_lindblad_dissipator(sc) * dt
            td_sc = td_Qobj(sc)
            A += [[ 1.0  / np.sqrt(2) * (td_sc.apply(spre) + td_sc.dag().apply(spost))]]
            A += [[-1.0j / np.sqrt(2) * (td_sc.apply(spre) - td_sc.dag().apply(spost))]]
    return L, A

def prep_sc_ops_photocurrent_rho(H, c_ops, sc_ops, dt, td):
    if not any(td):
        # No time-dependance
        L = liouvillian(H, c_ops=c_ops).data * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc, data_only=True) * dt
            n = sc.dag() * sc
            A += [[spre(n).data + spost(n).data,
                   (spre(sc) * spost(sc.dag())).data,
                   spre(n).data]]

    elif not td[1]:
        # sc_ops do not depend on time
        L = td_liouvillian(H, c_ops=c_ops) * dt
        A = []
        for sc in sc_ops:
            L += lindblad_dissipator(sc) * dt
            n = sc.dag() * sc
            A += [[spre(n).data + spost(n).data,
                   (spre(sc) * spost(sc.dag())).data,
                   spre(n).data]]

    else:
        def _cdc(c):
            return c*c.dag()
        def _cdc2(c):
            return spre(c) * spost(c.dag())

        td[0] = True
        L = td_liouvillian(H, c_ops=c_ops) * dt
        A = []
        for sc in sc_ops:
            L += td_lindblad_dissipator(sc) * dt
            td_sc = td_Qobj(sc)
            n = td_sc.apply(_cdc)._f_norm2()
            A += [[n.apply(spre) + n.apply(spost),
                   td_sc.apply(_cdc2)._f_norm2(),
                   n.apply(spre)]]
    return L, A


def d1_rho(t, rho_vec, sc_ops, args, td):
    """
    D1[a] rho = lindblad_dissipator(a) * rho
    Included in the Liouvilllian

    Homodyne + Heterodyne
    """
    return np.zeros(len(rho_vec))#

def d2_rho(t, rho_vec, sc_ops, args, td):
    """
    D2[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger]
              = (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec]

    Homodyne + Heterodyne
    """
    d2 = []
    if td:
        sc_t = [sc[0](t).data for sc in sc_ops]
    else:
        sc_t = [sc[0] for sc in sc_ops]
    for sc in sc_t:
        e1 = cy_expect_rho_vec(sc, rho_vec, 0)
        d2 += [spmv(sc, rho_vec) - e1 * rho_vec]
    return np.vstack(d2)

def d2_rho_milstein(t, rho_vec, sc_ops, args, td):
    """
    D2[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger]
              = (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec]

    Homodyne + Heterodyne
    """
    d2 = []
    if td:
        sc_t = [sc[0](t).data for sc in sc_ops]
    else:
        sc_t = [sc[0] for sc in sc_ops]
    for sc in sc_t:
        e1 = cy_expect_rho_vec(sc, rho_vec, 0)
        d2 += [spmv(sc, rho_vec) - e1 * rho_vec]
    d2_vec = np.vstack(d2)

    dd2 = []
    len_sc = len(sc_ops)
    for sc in sc_t:
        e1 = cy_expect_rho_vec(sc, rho_vec, 0)
        for d2 in d2_vec:
            e2 = cy_expect_rho_vec(sc, d2, 0)
            dd2 += [spmv(sc, d2) - e2 * rho_vec - e1 * d2]
    dd2 = np.vstack(dd2)
    return d2_vec, dd2.reshape((len_sc,len_sc,-1))

def d1_rho_photocurrent(t, rho_vec, A, args, td):
    d1 = np.zeros(len(rho_vec))
    if td:
        sc_t = [sc[0](t).data for sc in A]
    else:
        sc_t = [sc[0] for sc in A]
    for sc in sc_t:
        e1 = cy_expect_rho_vec(sc[0], rho_vec, 0)
        d1 += 0.5 * (e1 * rho_vec - spmv(sc[0], rho_vec))
    return d1

def d2_rho_photocurrent(t, rho_vec, A, args, td):
    d2 =[]
    if td:
        sc_t = [sc[1](t).data for sc in A]
    else:
        sc_t = [sc[1] for sc in A]
    for sc in sc_t:
        e1 = cy_expect_rho_vec(sc, rho_vec, 0)
        if e1.real > 1e-15:
            d2 += [spmv(sc, rho_vec) / e1 - rho_vec]
        else:
            d2 += [-rho_vec]
    return np.vstack(d2)


#
#   d1 and d2 functions for common schemes (Schrodinger)
#
def prep_sc_ops_homodyne_psi(H, c_ops, sc_ops, dt, td):
    if not td[0]:
        H = -1.0j*H.data * dt
    else:
        H = td_Qobj(H)*-1.0j * dt

    A = []
    if not td[1]:
        for sc in sc_ops:
            A += [[sc.data, (sc + sc.dag()).data, (sc.dag() * sc).data]]
    else:
        def _cdc(c):
            return c.dag()*c

        sc_t = [td_Qobj(sc) for sc in sc_ops]
        for sc in sc_t:
            n = sc.apply(_cdc)._f_norm2()
            A += [[sc, (sc + sc.dag()), n]]
    return H,A

def prep_sc_ops_heterodyne_psi(H, c_ops, sc_ops, dt, td):
    if not td[0]:
        H = -1.0j*H.data * dt
    else:
        H = td_Qobj(H)*-1.0j * dt

    A = []
    if not td[1]:
        for sc in sc_ops:
            A.append([sc.data, sc.dag().data, (sc + sc.dag()).data,
                     (sc - sc.dag()).data, (sc.dag() * sc).data])
    else:
        def _cdc(c):
            return c.dag()*c

        sc_t = [td_Qobj(sc) for sc in sc_ops]
        for sc in sc_t:
            cd = sc.dag()
            n = sc.apply(_cdc)._f_norm2()
            A.append([sc, cd, sc + cd,
                     sc - cd, n])
    return H,A

def prep_sc_ops_photocurrent_psi(H, c_ops, sc_ops, dt, td):
    if not td[0]:
        H = H.data * -1.0j * dt
    else:
        H = td_Qobj(H)*-1.0j * dt

    A = []
    if not td[1]:
        for sc in sc_ops:
            A += [[sc.data, (sc.dag() * sc).data]]
    else:
        def _cdc(c):
            return c.dag()*c
        sc_t = [td_Qobj(sc) for sc in sc_ops]
        for sc in sc_t:
            n = sc.apply(_cdc)._f_norm2()
            A += [[sc, n]]

    return H,A


def d1_psi_homodyne(t, psi, A, args, td):
    """
    .. math::
        D_1(C, \psi) = \\frac{1}{2}(\\langle C + C^\\dagger\\rangle\\C psi -
        C^\\dagger C\\psi - \\frac{1}{4}\\langle C + C^\\dagger\\rangle^2\\psi)
    """
    d1 = np.zeros(len(psi), dtype=complex)
    if td:
        sc_t = [[sc[0](t).data,sc[1](t).data,sc[2](t).data] for sc in A]
    else:
        sc_t = A
    for sc in sc_t:
        e1 = cy_expect_psi(sc[1], psi, 0)
        d1 += 0.5 * (e1 * spmv(sc[0], psi) -
                    spmv(sc[2], psi) -
                    0.25 * e1 ** 2 * psi)
    return d1

def d2_psi_homodyne(t, psi, A, args, td):
    """
    .. math::
        D_2(\psi, t) = (C - \\frac{1}{2}\\langle C + C^\\dagger\\rangle)\\psi
    """
    d2 = []
    if td:
        sc_t = [[sc[0](t).data,sc[1](t).data] for sc in A]
    else:
        sc_t = A
    for sc in sc_t:
        e1 = cy_expect_psi(sc[1], psi, 0)
        d2 += [spmv(sc[0], psi) - 0.5 * e1 * psi]
    return np.vstack(d2)

def d1_psi_heterodyne(t, psi, A, args, td):
    """
    .. math::
        D_1(\psi, t) = -\\frac{1}{2}(C^\\dagger C -
        \\langle C^\\dagger \\rangle C +
        \\frac{1}{2}\\langle C \\rangle\\langle C^\\dagger \\rangle))\psi
    """
    d1 = np.zeros(len(psi), dtype=complex)
    if td:
        sc_t = [[sc[0](t).data, sc[1](t).data, sc[4](t).data] for sc in A]
    else:
        sc_t = A
    for sc in sc_t:
        e_C = cy_expect_psi(sc[0], psi, 0)
        e_Cd = cy_expect_psi(sc[1], psi, 0)
        d1 += (-0.5 * spmv(sc[2], psi) +
                0.5 * e_Cd * spmv(sc[0], psi) -
                0.25 * e_C * e_Cd * psi)
    return d1

def d2_psi_heterodyne(t, psi, A, args, td):
    """
        X = \\frac{1}{2}(C + C^\\dagger)
        Y = \\frac{1}{2}(C - C^\\dagger)
        D_{2,1}(\psi, t) = \\sqrt(1/2) (C - \\langle X \\rangle) \\psi
        D_{2,2}(\psi, t) = -i\\sqrt(1/2) (C - \\langle Y \\rangle) \\psi
    """
    d2 = []
    if td:
        sc_t = [[sc[0](t).data, sc[2](t).data, sc[3](t).data] for sc in A]
    else:
        sc_t = [[sc[0], sc[2], sc[3]] for sc in A]
    for sc in sc_t:
        X = 0.5 * cy_expect_psi(sc[1], psi, 0)
        Y = 0.5 * cy_expect_psi(sc[2], psi, 0)
        d2 += [np.sqrt(0.5) * (spmv(sc[0], psi) - X * psi)]
        d2 += [-1.0j * np.sqrt(0.5) * (spmv(sc[0], psi) - Y * psi)]
    return np.vstack(d2)

def d1_psi_photocurrent(t, psi, A, args, td):
    """
    Note: requires poisson increments
    .. math::
        D_1(\psi, t) = - \\frac{1}{2}(C^\dagger C \psi - ||C\psi||^2 \psi)
    """
    d1 = np.zeros(len(psi))
    for sc in A:
        d1 += (-0.5 * (spmv(A[1], psi)
                - norm(spmv(A[0], psi)) ** 2 * psi))
    return d1

def d2_psi_photocurrent(t, psi, A, args, td):
    """
    Note: requires poisson increments
    .. math::
        D_2(\psi, t) = C\psi / ||C\psi|| - \psi
    """
    d2 = []
    for sc in A:
        psi_1 = spmv(A[0], psi)
        n1 = norm(psi_1)
        if n1 != 0:
            d2 += [psi_1 / n1 - psi]
        else:
            d2 += [- psi]
    return np.vstack(d2)


#
#   Stochastic schemes
#
def _rhs_euler_maruyama(H, vec_t, t, dt, sc_ops, dW, d1, d2, args, td):
    """
    Euler-Maruyama rhs function for both master eq and schrodinger eq.

    dV = -iH*V*dt + d1*dt + d2_i*dW_i
    """
    dvec_t = _rhs_deterministic(H, vec_t, t, dt, args, td[0])
    dvec_t += d1(t, vec_t, sc_ops, args, td[1]) * dt
    d2_vec = d2(t, vec_t, sc_ops, args, td[1])
    dvec_t += np.dot(dW, d2_vec)
    return vec_t + dvec_t

def _rhs_milstein(H, vec_t, t, dt, sc_ops, dW, d1, d2, args, td):
    """
    Milstein rhs function for both master eq and schrodinger eq.

    Slow but should be valid for non-commuting operators since computing
        both i x j and j x i.

    dV = -iH*V*dt + d1*dt + d2_i*dW_i
         + 0.5*d2_i(d2_j(V))*(dW_i*dw_j -dt*delta_ij)
    """

    #Euler part
    dvec_t = _rhs_deterministic(H, vec_t, t, dt, args, td[0])
    dvec_t += d1(t, vec_t, sc_ops, args, td[1]) * dt
    d2_vec, d22_vec = d2(t, vec_t, sc_ops, args, td[1])
    #d2_vec = d2_[0,:,:]
    #d22_vec = d2_[1:,:,:]
    dvec_t += np.dot(dW, d2_vec)

    #Milstein terms
    for ind, d2v in enumerate(d22_vec):
        dW2 = dW*dW[ind]*0.5
        dW2[ind] -= dt*0.5
        dvec_t += np.dot(dW2, d2v)

    return vec_t + dvec_t

def _rhs_platen(H, vec_t, t, dt, sc_ops, dW, d1, d2, args, td):
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
    dv_H1 = _rhs_deterministic(H, vec_t, t, dt, args, td[0])
    dv_H1 += d1(t, vec_t, sc_ops, args, td[1]) * dt
    d2_vec = d2(t, vec_t, sc_ops, args, td[1])
    Vp = vec_t + dv_H1 + np.sum(d2_vec,axis=0)*sqrt_dt
    Vm = vec_t + dv_H1 - np.sum(d2_vec,axis=0)*sqrt_dt
    Vt = vec_t + dv_H1 + np.dot(dW, d2_vec)

    # Platen dV
    dvt_H1 = _rhs_deterministic(H, Vt, t+dt, dt, args, td[0])
    dvt_H1 += d1(t+dt, Vt, sc_ops, args, td[1]) * dt
    dvec_t = 0.50 * (dv_H1 + dvt_H1)

    d2_vp = d2(t+dt, Vp, sc_ops, args, td[1])
    d2_vm = d2(t+dt, Vm, sc_ops, args, td[1])
    dvec_t += np.dot(dW, 0.25*(2 * d2_vec + d2_vp + d2_vm))
    dW2 = (dW**2 - dt) / sqrt_dt
    dvec_t += np.dot(dW, 0.25*(d2_vp - d2_vm))

    return vec_t + dvec_t


#
#   Preparation of operator for the fast version
#
def _generate_A_ops_simple(H, c_ops, sc_ops, dt):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic master equations
    """
    L = liouvillian(H,c_ops=c_ops).data
    A_len = len(sc)
    temp = [spre(c).data + spost(c.dag()).data for c in sc]
    tempL = (L + np.sum([lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) # Lagrangian

    out = []
    out += temp
    out += [tempL]

    out1 = [out]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]

    return L, out1

def _generate_A_ops_Euler(H, c_ops, sc_ops, dt):
    """
    combine precomputed operators in one long operator for the Euler method
    """
    L = liouvillian(H,c_ops=c_ops).data
    A_len = len(sc)
    out = []
    out += [spre(c).data + spost(c.dag()).data for c in sc]
    out += [(L + np.sum(
        [lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) * dt]
    out1 = [[sp.vstack(out).tocsr(), sc[0].shape[0]]]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]

    # XXX: fix this!
    out1[0][0].indices = np.array(out1[0][0].indices, dtype=np.int32)
    out1[0][0].indptr = np.array(out1[0][0].indptr, dtype=np.int32)

    return L, out1

def _generate_A_ops_Milstein(H, c_ops, sc_ops, dt):
    """
    combine precomputed operators in one long operator for the Milstein method
    with commuting stochastic jump operators.
    """
    L = liouvillian(H,c_ops=c_ops).data
    A_len = len(sc)
    temp = [spre(c).data + spost(c.dag()).data for c in sc]
    out = []
    out += temp
    out += [temp[n] * temp[n] for n in range(A_len)]
    out += [temp[n] * temp[m] for (n, m) in np.ndindex(A_len, A_len) if n > m]
    out += [(L + np.sum(
        [lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) * dt]
    out1 = [[sp.vstack(out).tocsr(), sc[0].shape[0]]]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]

    # XXX: fix this!
    out1[0][0].indices = np.array(out1[0][0].indices, dtype=np.int32)
    out1[0][0].indptr = np.array(out1[0][0].indptr, dtype=np.int32)

    return L, out1

def _generate_A_ops_implicit(H, c_ops, sc_ops, dt):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic master equations
    """
    L = liouvillian(H,c_ops=c_ops).data
    A_len = len(sc)
    temp = [spre(c).data + spost(c.dag()).data for c in sc]
    tempL = (L + np.sum([lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) # Lagrangian

    out = []
    out += temp
    out += [sp.eye(L.shape[0], format='csr') - 0.5*dt*tempL]
    out += [tempL]

    out1 = [out]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]

    return L, out1
