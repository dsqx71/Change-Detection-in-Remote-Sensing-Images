from .sym_util import *

def get_conv(name, data, num_filter, kernel, stride=(1,1), pad=(1,1), dilate=(1, 1), with_relu=True):

    weight = get_variable(name=name+'_weight')
    bias = get_variable(name=name+'_bias')

    conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel,
                                 weight=weight, bias=bias,
                                 stride=stride, pad=pad, dilate=dilate, no_bias=False)


    return mx.sym.Activation(conv, act_type="relu") if with_relu else conv

def siamese(data):

    # conv1
    conv1_1 = get_conv(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    conv1 = conv1_2 = get_conv(data=conv1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    conv1_2 = mx.sym.Pooling(data=conv1_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")

    # conv2s
    conv2_1 = get_conv(data=conv1_2, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1", dilate=(1, 1))
    conv2 = conv2_2 = get_conv(data=conv2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    conv2_2 = mx.sym.Pooling(data=conv2_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")

    # # conv3
    conv3_1 = get_conv(data=conv2_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1", dilate=(1, 1))
    conv3_2 = get_conv(data=conv3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2", dilate=(1 ,1))
    conv3 = conv3_3 = get_conv(data=conv3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    # conv3_3 = mx.sym.Pooling(data=conv3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")
    #
    # # conv4
    # conv4_1 = get_conv(data=conv3_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1", dilate=(1, 1))
    # conv4_2 = get_conv(data=conv4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2", dilate=(1, 1))
    # conv4 = conv4_3 = get_conv(data=conv4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    # conv4_3 = mx.sym.Pooling(data=conv4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
    #
    # # conv5
    # conv5_1 = get_conv(data=conv4_3, kernel=(3, 3), pad=(16, 16), num_filter=512, name="conv5_1", dilate=(16, 16))
    # conv5_2 = get_conv(data=conv5_1, kernel=(3, 3), pad=(16, 16), num_filter=512, name="conv5_2", dilate=(16, 16))
    # conv5_3 = get_conv(data=conv5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    #
    # b5 = mx.sym.UpSampling(data=conv5_3, scale=16, num_filter=512, num_args=1, sample_type='bilinear', workspace=2048)
    # b4 = mx.sym.UpSampling(data=conv4, scale=8, num_filter=512, num_args=1, sample_type='bilinear', workspace=2048)
    b3 = mx.sym.UpSampling(data=conv3, scale=4, num_filter=256, num_args=1, sample_type='bilinear', workspace=2048)
    b2 = mx.sym.UpSampling(data=conv2, scale=2, num_filter=128, num_args=1, sample_type='bilinear', workspace=2048)
    #
    # ret = mx.sym.concat(conv1, b2, b3, b4, b5)
    ret = mx.sym.concat(conv1, b2, b3)
    return ret



def vgg16(ratio_neg=50, metric='cosine'):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    label = mx.sym.Variable('label')

    out1 = siamese(img1)
    out2 = siamese(img2)


    if metric == 'l2_distance':
        data = mx.sym.sqrt(mx.sym.sum(mx.sym.square(out1 - out2), axis=1, keepdims=True))

    elif metric == 'cosine':
        data = mx.sym.sum(out1 * out2, axis=1, keepdims=True)
        norm1 = mx.sym.sqrt(mx.sym.sum(mx.sym.square(out1), axis=1, keepdims=True))
        norm2 = mx.sym.sqrt(mx.sym.sum(mx.sym.square(out2), axis=1, keepdims=True))
        data = data / norm1
        data = data / norm2
        data = 1 - data

    loss = mx.symbol.Custom(data=data, label=label, name='L1_sparse', loss_scale=1.0, is_l1=False, adaptive=True, ratio_neg=ratio_neg, op_type='SparseRegressionLoss')

    return loss
