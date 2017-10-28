import mxnet as mx
import numpy as np

var_registrar = {}
def get_variable(name, shape=None, init=None):
    global var_registrar
    if name not in var_registrar:
        var_registrar[name] = mx.sym.Variable(name, shape=shape, init=init, dtype=np.float32)
    return var_registrar[name]