from config import cfg
from utils import io, misc
import matplotlib.pyplot as plt
from collections import namedtuple
from model import vgg16
import mxnet as mx
import numpy as np

plt.rcParams['figure.figsize'] = (10, 20)


# Load data
print ("Loading data...")
pca_img2015, pca_img2017, label = io.read_data()
pca_img2015 = pca_img2015 - cfg.data.mean
pca_img2017 = pca_img2017 - cfg.data.mean

# Build network
print ("Building symbol...")
net = vgg16.Vgg16_siamese(ratio_neg=10)
args, auxs = misc.load_pretrainModel('vgg16', net)

# Init executor
print ("Initializing executor...")
DataBatch = namedtuple('DataBatch', ['data', 'label'])
fix_params = [item for item in net.list_arguments() if 'upsampling' in item]
init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=8)
mod = mx.module.Module(symbol=net,
                       data_names=['img1', 'img2'],
                       label_names=['label', ],
                       context=mx.cpu(0),
                       fixed_param_names=fix_params)
mod.bind(data_shapes=[('img1', cfg.data.batch_shape), ('img2', cfg.data.batch_shape)],
         label_shapes=[('label', cfg.data.label_shape)],
         for_training=True, force_rebind=False)
mod.init_params(initializer=init, arg_params=args, aux_params=auxs,
                allow_missing=True, force_init=False)
mod.init_optimizer(kvstore='local',
                   optimizer='Adam',
                   optimizer_params=dict(learning_rate=1e-5,
                                         beta1=0.90,
                                         beta2=0.999,
                                         epsilon=1e-4,
                                         rescale_grad=1.0 / cfg.data.batch_shape[0],
                                         wd=0.0001,
                                         lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                                      factor=0.5,
                                                                                      stop_factor_lr=3.125E-6)))

# EM
print ("Start EM algorithm...")
for k in range(3):

    print("iteration:{}".format(k))
    print("E-step...")
    t1 = 2000
    t2 = 10000
    minm_p = np.inf
    maxm_p = -np.inf
    count_p = 0

    minm_n = np.inf
    maxm_n = -np.inf
    count_n = 0

    # E-step
    samples = io.sample_label(pca_img2015, pca_img2017, label, 20, 50)

    for i in range(len(samples)):
        dbatch = DataBatch(data=[mx.nd.array(np.expand_dims(samples[i][0], 0).transpose(0, 3, 1, 2)),
                                 mx.nd.array(np.expand_dims(samples[i][1], 0).transpose(0, 3, 1, 2))],
                           label=[mx.nd.array([np.expand_dims(samples[i][2], 0)])])

        mod.forward(dbatch)
        out = mod.get_outputs()[0].asnumpy()[0][0]

        if (samples[i][2] == samples[i][2]).any():

            if (out[samples[i][2] == 1].size > 0):
                minm_p = min(out[samples[i][2] == 1].min(), minm_p)
                maxm_p = max(out[samples[i][2] == 1].max(), maxm_p)
                count_p += 1
                print('minm_p:{}, maxm_p:{}'.format(minm_p, maxm_p))

            if (out[samples[i][2] == 0].size > 0):
                minm_n = min(out[samples[i][2] == 0].min(), minm_n)
                maxm_n = max(out[samples[i][2] == 0].max(), maxm_n)
                count_n += 1
                print('minm_n:{}, maxm_n:{}'.format(minm_n, maxm_n))

        # Label those data with negative class
        if count_p >= 10:
            mask1 = (out < minm_p - t1) & (samples[i][2] != samples[i][2])
            samples[i][2][mask1] = 0

        # Label those data with positive class
        if count_n >= 10:
            mask2 = (out > maxm_n + t2) & (samples[i][2] != samples[i][2])
            samples[i][2][mask2] = 1

    print("positive class: {}".format((label == 1).sum()))
    print("negative class: {}".format((label == 0).sum()))

    # M-step
    print ("M-step")
    for i in range(len(samples)):
        if ((samples[i][2] == samples[i][2]).any() and (count_p > 0 or count_n > 0)):
            dbatch = DataBatch(data=[mx.nd.array(np.expand_dims(samples[i][0], 0).transpose(0, 3, 1, 2)),
                                     mx.nd.array(np.expand_dims(samples[i][1], 0).transpose(0, 3, 1, 2))],
                               label=[mx.nd.array([np.expand_dims(samples[i][2], 0) * maxm_p])])
            mod.forward_backward(dbatch)
            mod.update()

