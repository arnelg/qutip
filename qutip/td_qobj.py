from qutip import Qobj
from qutip.interpolate import Cubic_Spline
from functools import partial
from types import FunctionType,BuiltinFunctionType
import numpy as np
from numbers import Number

class td_Qobj:


    def __init__(self, operator=[], args={}, tlist=None):
        self.const = False
        self.args = args

        op_type = self._td_format_check_single(operator, tlist)
        self.ops = []

        if isinstance(op_type, int):
            if op_type == 0:
                self.cte = operator
                self.const = True
                if operator.issuper :
                    self.issuper = True
                self.op_call = self._evaluate_const

            elif op_type == 1: #a function, no test to see if the function does return a Qobj.
                self.op_call = operator


            elif op_type == -1: #a function, no test to see if the function does return a Qobj.
                self.op_call = self._evaluate_func
        else:


        if op_type<10:
            if op_type == 0:
                self.cte = operator
                self.const = True
                if operator.issuper :
                    self.issuper = True
                self.op_call = self._evaluate_const

            if op_type == 1: #a function, no test to see if the function does return a Qobj.
                self.op_call = operator

        else:
            compile_list = []
            compile_count = 0
            for type_, op in zip(op_type, operator):
                if type_ == 0:
                    if self.cte == None:
                        self.cte = op
                    else:
                        self.cte += op
                elif type_ == 1:
                    self.ops.append([op[0], op[1], op[1], 1])
                elif type_ == 2:
                    self.ops.append([op[0], None, op[1], 2])
                    compile_list.append((op[1], compile_count))
                    compile_count += 1
                elif type_ == 3:
                    self.ops.append([op[0],
                            lambda t,*args: 
                            (0 if (t > tlist[-1]) 
                            else op[1][int(round((len(tlist)-1) * (t/tlist[-1])))]),
                            op[1], 3])
                else:
                    raise Exception("Should never be here")

            if compile_count:
                str_funcs = _compile_str_single(compile_list)
                count = 0
                for op in self.ops:
                    if op[3] == 2:
                        op[1] = str_funcs[count]
                        count += 1
            
            self.op_call = self._evaluate_func
            if not self.cte:
                self.cte = self.ops[0][0]
                for op in self.ops[1:]:
                    self.cte += op[0]
                self.cte *= 0.

            if not self.ops:
                self.const = True



    # Different function to get the state
    def __call__(self, t):
        return self.op_call(t,**(self.args))

    def _evaluate_const(self, t):
        return self.cte

    def _evaluate_list(self, t, **kw_args):
        op_t = self.cte
        for part in self.op_func:
            op_t += part[0]*part[1](t,**kw_args)
        return op_t



    def _td_array_to_str(self, op_np2, times):
        """
        Wrap numpy-array based time-dependence in the string-based time dependence
        format
        """
        n = 0
        str_op = []
        np_args = {}

        for op in op_np2:
            td_array_name = "_td_array_%d" % n
            H_td_str = '(0 if (t > %f) else %s[int(round(%d * (t/%f)))])' %\
                (times[-1], td_array_name, len(times) - 1, times[-1])
            np_args[td_array_name] = op[1]
            str_op.append([op[0], H_td_str])
            n += 1

        return str_op,np_args

    def _td_format_check_single(self, operator, tlist=None):
        op_type = []

        if isinstance(operator, Qobj):
            op_type = 0
        elif isinstance(operator, (FunctionType, BuiltinFunctionType, partial)):
            op_type = 1 
        elif isinstance(operator, list):
            if (len(operator) == 0 ):
                op_type = -1
            for op_k in operator:
                if isinstance(op_k, Qobj):
                    op_type.append(0)
                elif isinstance(op_k, list):
                    if not isinstance(op_k[0], Qobj):
                        raise TypeError("Incorrect operator specification")
                    elif len(op_k) == 2:
                        if isinstance(op_k[1], (FunctionType,
                                               BuiltinFunctionType, partial)):
                            op_type.append(1)
                        elif isinstance(op_k[1], str):
                            op_type.append(2)
                        elif isinstance(op_k[1], np.ndarray):
                            if not isinstance(tlist, np.ndarray) or not len(op_k[1]) == len(tlist):
                                raise TypeError("Time list do not match")
                            op_type.append(3)
                        elif isinstance(op_k[1], Cubic_Spline):
                            raise NotImplementedError("Cubic_Spline not supported")
                        #    h_obj.append(k)
                        else:
                            raise TypeError("Incorrect operator specification")
                    else:
                        raise TypeError("Incorrect operator specification")



        else:
            raise TypeError("Incorrect operator specification")

        return op_type

    def _generate_op(self):

        compiled_str_coeff = self._compile_str_()

        if( len(self.args) == 0 ):
            def str_func_with_np(t):
                return compiled_str_coeff(t)
        else:
            def str_func_with_np(t):
                return compiled_str_coeff(t,*(list(zip(*self.args))[1]))

        return compiled_str_coeff
        

    def _compile_str_single(self, compile_list):

        import os
        _cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

        all_str = ""
        for op in compile_list:
            all_str += op[1]

        filename = "td_Qobj_"+str(hash(all_str))[1:]

        Code = """
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

include """+_include_string+"\n"

        for str_coeff in compile_list:
            Code_header += _str_2_code(str_coeff)

        file = open(filename+".pyx", "w")
        file.writelines(Code)
        file.close()


        str_func = []
        imp = ' import '
        for i in range(len(compile_count))
            func_name = '_str_factor_'+ str(i)
            import_code = compile('from ' + filename + ' import ' + func_name +
                                  "\n str_func.append(" + func_name + ")",
                                  '<string>', 'exec')
            exec(import_code, globals())

        try:
            os.remove(filename+".pyx")
        except:
            pass

        return str_func

    def _str_2_code(self,str_coeff):

        func_name = '_str_factor_'+ str(str_coeff[1])

        Code = """

@cython.boundscheck(False)
@cython.wraparound(False)

def """ + func_name +"(double t"
        Code += self._get_arg_str()
        Code += "):\n"
        Code += "return " + str_coeff[0] + "\n"

        return Code


    def _get_arg_str(self):
        if len(self.args) == 0:
            return ''

        ret = ''
        for name, value in self.args.items():
            if isinstance(value, np.ndarray):
                ret += ",\n        np.ndarray[np.%s_t, ndim=1] %s" % \
                    (value.dtype.name, name)
            else:
                if isinstance(value, (int, np.int32, np.int64)):
                    kind = 'int'
                elif isinstance(value, (float, np.float32, np.float64)):
                    kind = 'float'
                elif isinstance(value, (complex, np.complex128)):
                    kind = 'complex'
                ret += ",\n        " + kind + " " + name
        return ret


    def copy(self):
        new = td_Qobj(self.cte.copy())
        for op in self.ops:
            new.ops.append(op)
            new.ops[-1][0] = op[0].copy()

    def __add__(self, other):
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        res = self.copy()
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, td_Qobj):
            self.cte += other.cte
            self.ops += other.ops
        else:
            self.cte += other

    def __sub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __isub__(self, other):
        self += (-other)

    def __mul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, Number):
            self.cte *= other
            for op in enumerate(ops):
                op[0] *= other
            return res
        else:
            raise TypeError("td_qobj can only be multiplied with Qobj or numbers")

    def __div__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            res = self.copy()
            res *= other**(-1)
            return res
        else:
            raise TypeError('Incompatible object for division')

    def __idiv__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            self *= other**(-1)
        else:
            raise TypeError('Incompatible object for division')

    def __truediv__(self, other):
        return self.__div__(other)

    def __pow__(self, n, m=None):
        res = self.copy()
        res.cte = self.cte.__pow__(n,m)
        for op in res.ops:
            op[0] = op[0].__pow__(n,m)
        return res

    def __ipow__(self, n, m=None):
        self.cte = self.cte.__pow__(n,m)
        for op in self.ops:
            op[0] = op[0].__pow__(n,m)

    def __neg__(self):
        res = self.copy()
        res.cte = -res.cte
        for op in res.ops:
            op[0] = -op[0]
        return res

    def __abs__(self):
        res = self.copy()
        res.cte = abs(res.cte)
        for op in res.ops:
            op[0] = abs(op[0])
        return res

    def trans(self):
        res = self.copy()
        res.cte = trans(res.cte)
        for op in res.ops:
            op[0] = trans(op[0])
        return res

    def sinm(self):
        res = self.copy()
        res.cte = sinm(res.cte)
        for op in res.ops:
            op[0] = sinm(op[0])
        return res

    def cosm(self):
        res = self.copy()
        res.cte = cosm(res.cte)
        for op in res.ops:
            op[0] = cosm(op[0])
        return res

    def conj(self):
        res = self.copy()
        res.cte = conj(res.cte)
        for op in res.ops:
            op[0] = conj(op[0])
        return res       

    def dag(self):
        res = self.copy()
        res.cte = dag(res.cte)
        for op in res.ops:
            op[0] = dag(op[0])
        return res     

    def sqrtm(self, sparse=False, tol=0, maxiter=100000):
        res = self.copy()
        res.cte = sqrtm(res.cte, sparse, tol, maxiter)
        for op in res.ops:
            op[0] = sqrtm(op[0], sparse, tol, maxiter)
        return res     

    def apply(self, function, *args, **kw_args):
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op[0] = function(op[0], *args, **kw_args)
        return res 

#return bool
#def check_herm(self):





liouvillian(H, c_ops=[], data_only=False, chi=None)

def td_liouvillian(H, c_ops=[],  chi=None):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators. Accept time dependant 
    operator and return a td_qobj

    Parameters
    ----------
    H : qobj
        System Hamiltonian.

    c_ops : array_like
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : td_qobj
        Liouvillian superoperator.

    """
    L = None

    if chi and len(chi) != len(c_ops):
        raise ValueError('chi must be a list with same length as c_ops')

    if H is not None:
        if not isinstance(H,td_Qobj):
            L = td_Qobj(H)
            L.apply(liouvillian, chi=chi)


    if isinstance(c_ops, list) and len(c_ops) > 0:
        def liouvillian_c(c_ops,chi):
            return liouvillian(c_ops=c_ops,chi=chi)
        for c in c_ops:
            if not isinstance(c,td_Qobj):
                cL = td_Qobj(c)
                cL.apply(liouvillian_c,chi=chi)
                if L is None:
                    L = cL
                else:
                    L += cL

    return L
