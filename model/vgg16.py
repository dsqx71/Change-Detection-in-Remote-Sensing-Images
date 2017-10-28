from .sym_util import  *

def siamese(data, name):


    # Sharing weights
    w1_1 = get_variable(name="conv1_1")
    w1_2 = get_variable(name="conv1_2")
    w2_1 = get_variable(name="conv2_1")
    w2_2 = get_variable(name="conv2_2")
    w3_1 = get_variable(name="conv3_1")
    w3_2 = get_variable(name="conv3_2")
    w3_3 = get_variable(name="conv3_3")

    # conv1
    conv1_1 = mx.sym.Convolution(data=data, weight=w1_1, kernel=(3, 3), pad=(1, 1), num_filter=64)
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.sym.Convolution(data=relu1_1, weight=w1_2, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", name="relu1_2")

    # conv2
    conv2_1 = mx.sym.Convolution(data=relu1_2, weight=w2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.sym.Convolution(data=relu2_1, weight=w2_2, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", name="relu2_2")

    # conv3
    conv3_1 = mx.sym.Convolution(data=relu2_2, weight=w3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, weight=w3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, weight=w3_3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")

    return relu3_3


def Vgg16_siamese():

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    out1 = siamese(img1, 'img1')
    out2 = siamese(img2, 'img2')
    
    net = mx.sym.sum(out1 * out2, axis=1)

    return net
