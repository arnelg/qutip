from qutip import Qobj
from qutip.interpolate import Cubic_Spline
from functools import partial
from types import FunctionType,BuiltinFunctionType
import numpy as np


class td_Qobj:




    def __init__(self, operator, issuper=False, args={}, tlist=None):
        self.const = False
        self.issuper = issuper
        self.args = args

        op_type, self.op_const, self.op_str, self.op_func, self.op_np2 =\
            self._td_format_check_single(operator, tlist)
        print(op_type)
        self.op_type = op_type;

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

            if self.op_const:
                self.cte = sum(self.op_const)

            if op_type == 10: #list of const
                self.const = True
                if self.cte.issuper :
                    self.issuper = True
                self.op_call = self._evaluate_const

            elif op_type == 11: #list of str
                sum_op = sum(list(zip(*self.op_str))[0])
                if self.op_const:
                    sum_op += self.cte
                else:
                    self.cte = sum_op*0. ##---------------------------------------------------------------------
                if sum_op.issuper :
                    self.issuper = True
                self.op_call = self._evaluate_str
                self.str_compute = self._generate_op()

            elif op_type == 12: #list of func
                sum_op = sum(list(zip(*self.op_func))[0])
                if self.op_const:
                    sum_op += self.cte
                else:
                    self.cte = sum_op*0. ##---------------------------------------------------------------------
                if sum_op.issuper :
                    self.issuper = True
                self.op_call = self._evaluate_func

            elif op_type == 14: #list of numpy array
                self.t_max = tlist[-1]
                self.num_t = len(tlist)-1
                sum_op = sum(list(zip(*self.op_np2))[0])
                if self.op_const:
                    sum_op += self.cte
                else:
                    self.cte = sum_op*0. ##---------------------------------------------------------------------
                if sum_op.issuper :
                    self.issuper = True
                self.op_call = self._evaluate_numpy

            elif op_type == 15: 
                #list of str + numpy
                #numpy array transformed to string
                sum_op = sum(list(zip(*self.op_str))[0])
                sum_op += sum(list(zip(*self.op_np2))[0])
                if self.op_const:
                    sum_op += self.cte
                else:
                    self.cte = sum_op*0. ##---------------------------------------------------------------------
                if sum_op.issuper :
                    self.issuper = True
                np_str_op, args_new = self._td_array_to_str(self.op_np2, tlist)
                self.op_str += np_str_op
                args.update(args_new)
                self.op_call = self._evaluate_str
                self.str_compute = self._generate_op()

            elif op_type == 16: 
                #list of func + numpy
                #numpy array transformed to func
                sum_op = sum(list(zip(*self.op_func))[0])
                sum_op += sum(list(zip(*self.op_np2))[0])
                if self.op_const:
                    sum_op += self.cte
                else:
                    self.cte = sum_op*0. ##---------------------------------------------------------------------
                if sum_op.issuper :
                    self.issuper = True
                for op in self.op_np2:
                    self.op_func += [[op[0],
                            lambda t,*args: 
                            (0 if (t > tlist[-1]) 
                            else op[1][int(round((len(tlist)-1) * (t/tlist[-1])))]) ]]
                self.op_call = self._evaluate_func


            else:
                print(op_type)
                raise("Format not supported (string and function)")



    # Different function to get the state
    def __call__(self, t):
        return self.op_call(t,**(self.args))

    def _evaluate_const(self, t):
        return self.cte

    def _evaluate_str(self, t, **kw_args):
        op_t = self.cte
        coeff = self.str_compute(t,**kw_args)
        for i,part in enumerate( self.op_str):
            op_t += part[0]*coeff[i]
        return op_t

    def _evaluate_func(self, t, **kw_args):
        op_t = self.cte
        for part in self.op_func:
            op_t += part[0]*part[1](t,**kw_args)
        return op_t

    def _evaluate_numpy(self, t, **kw_args):
        op_t = self.cte
        for part in self.op_np2:
            op_t += part[0]*(0 if (t > self.t_max) else part[1][int(round(self.num_t * (t/self.t_max)))])
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
        op_const = []
        op_func = []
        op_str = []
        op_np2 = []
        #op_obj = []

        if isinstance(operator, Qobj):
            op_type = 0
        elif isinstance(operator, (FunctionType, BuiltinFunctionType, partial)):
            op_type = 1 

        elif isinstance(operator, list):
            for k, op_k in enumerate(operator):
                if isinstance(op_k, Qobj):
                    op_const.append(op_k)
                elif isinstance(op_k, list):
                    if not isinstance(op_k[0], Qobj):
                        raise TypeError("Incorrect operator specification")
                    elif len(op_k) == 2:
                        if isinstance(op_k[1], (FunctionType,
                                               BuiltinFunctionType, partial)):
                            op_func.append(op_k)
                        elif isinstance(op_k[1], str):
                            op_str.append(op_k)
                        elif isinstance(op_k[1], Cubic_Spline):
                            raise TypeError("Cubic_Spline not supported")
                        #    h_obj.append(k)
                        elif isinstance(op_k[1], np.ndarray):
                            if not isinstance(tlist, np.ndarray) or not len(op_k[1]) == len(tlist):
                                raise TypeError("Time list do not match")
                            op_np2.append(op_k)
                        else:
                            raise TypeError("Incorrect operator specification")
                    else:
                        raise TypeError("Incorrect operator specification")

            if (len(operator) == 0 ): #+ len(op_np3)
                op_type = -1
                raise TypeError(
                    "The operator cannot be an empty list")

            op_type = 10

            if len(op_str)  > 0:
                op_type += 1

            if len(op_func) > 0:
                op_type += 2

            if len(op_np2) > 0:
                op_type += 4


        else:
            raise TypeError("Incorrect operator specification")

        return op_type, op_const, op_str, op_func, op_np2#, op_np3

    def _generate_op(self):

        compiled_str_coeff = self._compile_str_()

        if( len(self.args) == 0 ):
            def str_func_with_np(t):
                return compiled_str_coeff(t)
        else:
            def str_func_with_np(t):
                return compiled_str_coeff(t,*(list(zip(*self.args))[1]))

        return compiled_str_coeff
        
    def _compile_str_(self):

        import os
        _cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

        all_str = ""
        for op in self.op_str:
            all_str += op[1]

        filename = "td_Qobj_"+str(hash(all_str))[1:]

        n_str = len(self.op_str)


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

include """+_include_string+"""

@cython.boundscheck(False)
@cython.wraparound(False)

def cy_td_str_factor(double t"""
        Code += self._get_arg_str()
        Code += """):
    cdef double complex * out = <complex *>PyDataMem_NEW_ZEROED("""+str(n_str)+",sizeof(complex))\n"

        op_index = 0
        for op in self.op_str:
            Code += "\n    out["+str(op_index)+"] = " + op[1]
            op_index += 1

        Code += "\n    cdef np.npy_intp dims = " + str(n_str)
        Code += "\n    cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_COMPLEX128, out)"
        Code += "\n    PyArray_ENABLEFLAGS(arr_out, np.NPY_OWNDATA)"
        Code += "\n    return arr_out"

        file = open(filename+".pyx", "w")
        file.writelines(Code)
        file.close()

        import_code = compile('from ' + filename +
                       ' import cy_td_str_factor', '<string>', 'exec')
        exec(import_code, globals())

        try:
            os.remove(filename+".pyx")
        except:
            pass

        return cy_td_str_factor


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




