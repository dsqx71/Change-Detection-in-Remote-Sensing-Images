import mxnet as mx
import numpy as np

var_registrar = {}

def get_variable(name, shape=None, init=None):
    global var_registrar
    if name not in var_registrar:
        var_registrar[name] = mx.sym.Variable(name, shape=shape, init=init, dtype=np.float32)
    return var_registrar[name]


class SparseRegressionLoss(mx.operator.CustomOp):
    """
        SparseRegressionLoss will ignore labels with values of NaN
    """
    def __init__(self,loss_scale, is_l1, ratio_neg):
        # due to mxnet serialization problem
        super(SparseRegressionLoss, self).__init__()

        self.loss_scale = float(loss_scale)
        self.is_l1 = bool(is_l1)
        self.ratio_neg = float(ratio_neg)

    def forward(self, is_train, req, in_data, out_data, aux):

        x = in_data[0]
        y = out_data[0]
        self.assign(y, req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        label = in_data[1].asnumpy()
        y = out_data[0].asnumpy()

        # find negative samples
        mask_neg = (label == 0)

        # find invalid labels
        mask_nan = (label != label)

        # total number of valid points
        normalize_coeff = (~mask_nan[:, 0, :, :]).sum()
        if self.is_l1:
            tmp = np.sign(y - label) * self.loss_scale / float(normalize_coeff)
        else:
            tmp = (y - label) * self.loss_scale / float(normalize_coeff)

        # ignore NaN
        tmp[mask_nan] = 0
        if normalize_coeff == 0:
            tmp[:] = 0

        tmp[mask_neg] = self.ratio_neg * tmp[mask_neg]
        self.assign(in_grad[0], req[0], mx.nd.array(tmp))


@mx.operator.register("SparseRegressionLoss")
class SparseRegressionLossProp(mx.operator.CustomOpProp):

    def __init__(self, loss_scale, is_l1, ratio_neg=1):
        super(SparseRegressionLossProp, self).__init__(False)
        self.loss_scale = loss_scale
        self.is_l1 = is_l1
        self.ratio_neg = ratio_neg

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]

        return [data_shape, label_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):

        return SparseRegressionLoss(self.loss_scale, self.is_l1, self.ratio_neg)
