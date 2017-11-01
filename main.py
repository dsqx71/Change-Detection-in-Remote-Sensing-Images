import mxnet as mx
import numpy as np
import argparse
import logging
import os

from config import cfg
from utils import io, misc
import matplotlib.pyplot as plt
from collections import namedtuple
from model import vgg16
DataBatch = namedtuple('DataBatch', ['data', 'label'])

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pretrain model', choices=['vgg16'])
    parser.add_argument('--epoch', type=int, help='continue training', default=0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--t1', type=float, default=0.010)
    parser.add_argument('--t2', type=float, default=0.999)
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'Adam'])
    args = parser.parse_args()

    # logging
    exp_name = '_'.join([args.model, str(args.opt), str(args.t1), str(args.t2)])
    log_file = os.path.join(cfg.dirs.log_prefix, exp_name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', filename=log_file, filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(args)

    # Load data
    pca_img2015, pca_img2017, label = io.read_data()
    pca_img2015 = pca_img2015 - cfg.data.mean
    pca_img2017 = pca_img2017 - cfg.data.mean
    gt = io.tiff.imread('./data/answer_complete 2.tif')
    gt[gt > 0] = 1

    # create dir
    checkpoint_prefix = os.path.join(cfg.dirs.checkpoint, exp_name)
    checkpoint_path = os.path.join(checkpoint_prefix, exp_name)
    if os.path.isdir(checkpoint_prefix) == False:
        os.makedirs(checkpoint_prefix)

    # Init executor
    net = eval('{}.{}'.format(args.model, args.model))(ratio_neg=1)
    fix_params = [item for item in net.list_arguments() if 'upsampling' in item]
    if args.epoch == 0:
        # Load pretrain model
        init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=8)
        mod = mx.module.Module(symbol=net,
                               data_names=['img1', 'img2'],
                               label_names=['label', ],
                               context=mx.gpu(0),
                               fixed_param_names=fix_params)
        mod.bind(data_shapes=[('img1', cfg.data.batch_shape), ('img2', cfg.data.batch_shape)],
                 label_shapes=[('label', cfg.data.label_shape)],
                 for_training=True, force_rebind=False)
        pretrain_args, pretrain_auxs = misc.load_pretrainModel(args.model, net)
        mod.init_params(initializer=init, arg_params=pretrain_args, aux_params=pretrain_auxs    ,
                        allow_missing=True, force_init=False)
    else:
        mod = mx.module.Module.load(prefix=checkpoint_path,
                                    epoch=args.epoch,
                                    load_optimizer_states=True,
                                    data_names=['img1', 'img2'],
                                    label_names=['label', ],
                                    context=mx.gpu(0),
                                    fixed_param_names=fix_params)
        mod.bind(data_shapes=[('img1', cfg.data.batch_shape), ('img2', cfg.data.batch_shape)],
                 label_shapes=[('label', cfg.data.label_shape)],
                 for_training=True, force_rebind=False)
        label = np.load(checkpoint_path + '_predict_{}'.format(args.epoch))
        a,b = mod.get_params()
        for key in a:
            assert (a[key].asnumpy() !=0).any()


    if args.opt =='sgd':
        optimizer_params = dict(learning_rate=args.lr,
                                wd=0.0004,
                                momentum=0.90)
    else:
        optimizer_params = dict(learning_rate=args.lr,
                                beta1=0.90,
                                beta2=0.999,
                                epsilon=1e-4,
                                rescale_grad=1.0 / cfg.data.batch_shape[0],
                                wd=0.0004,
                                lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                             factor=0.5,
                                                                             stop_factor_lr=3.125E-6))
    mod.init_optimizer(kvstore='device',
                       optimizer=args.opt,
                       optimizer_params=optimizer_params)

    # EM
    print ("EM alogtihm")
    logging.info("Start EM algorithm...")
    for k in range(args.epoch+1, args.num_epoch):

        logging.info("-------------Epoch:{}----------------".format(k))
        logging.info("E-step..............")
        minm_p = np.inf
        maxm_p = -np.inf
        count_p = 0

        minm_n = np.inf
        maxm_n = -np.inf
        count_n = 0

        flag = False

        # E-step
        samples = io.sample_label(pca_img2015, pca_img2017, label, 1000, 2000)

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

                if (out[samples[i][2] == 0].size > 0):
                    minm_n = min(out[samples[i][2] == 0].min(), minm_n)
                    maxm_n = max(out[samples[i][2] == 0].max(), maxm_n)
                    count_n += 1

            if count_p >= 400:
                # Label those data with negative class
                mask1 = (out < args.t1 * minm_p) & (samples[i][2] != samples[i][2])
                samples[i][2][mask1] = 0

                # Label those data with positive class
                mask2 = (out > args.t2 * maxm_p) & (samples[i][2] != samples[i][2])
                samples[i][2][mask2] = 1
                if mask1.size > 0 or mask2.size > 0:
                    flag = True

            if i % 200 == 0:
                logging.info('minm_p:{}, maxm_p:{}, minm_n:{}, maxm_n:{}, positive class: {}, negative class: {}'.format(minm_p, maxm_p, minm_n, maxm_n, (label == 1).sum(), (label == 0).sum()))

        # M-step
        logging.info("M-step..........")
        for i in range(len(samples)):
            if ((samples[i][2] == samples[i][2]).any() and (count_p > 0 or count_n > 0)):
                dbatch = DataBatch(data=[mx.nd.array(np.expand_dims(samples[i][0], 0).transpose(0, 3, 1, 2)),
                                         mx.nd.array(np.expand_dims(samples[i][1], 0).transpose(0, 3, 1, 2))],
                                   label=[mx.nd.array([np.expand_dims(samples[i][2], 0)])])
                mod.forward_backward(dbatch)
                mod.update()

        # Save checkpoint and result
        mod.save_checkpoint(prefix=checkpoint_path, epoch=k, save_optimizer_states=True)
        np.save(checkpoint_path + '_predict_{}'.format(k), label)

        # Evaluation
        score, density = misc.F1_score(label, gt)
        acc = misc.accuracy(label, gt)
        logging.info("Epoch : %d, F1-score : %.4f, accuracy: %.4f, Density : %.4f" %(k, score, acc, density))

        if flag == False:
            args.t1 += 0.05
            args.t2 -= 0.01

