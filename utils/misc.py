import mxnet as mx
import subprocess
import os
from config import cfg
from sklearn.metrics import f1_score, accuracy_score

def load_checkpoint(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params,aux_params

def load_pretrainModel(name, net):
    # Load pretrained model
    arg_name = net.list_arguments()
    aux_name = net.list_auxiliary_states()
    pretrain_args, pretrain_auxs = load_checkpoint(os.path.join(cfg.dirs.pretrain_model,name), 0)

    args = {}
    auxs = {}
    for key in pretrain_args:
        if key in arg_name:
            args[key] = pretrain_args[key]

    for key in pretrain_auxs:
        if key in aux_name:
            auxs[key][:] = pretrain_auxs[key]

    return args, auxs

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))


def F1_score(pred, gt):

    mask = pred == pred
    val_pred = pred[mask]
    val_gt = gt[mask]
    score = f1_score(val_gt, val_pred, average='binary')
    density = mask.sum() / float(gt.size)

    return score, density


def accuracy(pred, gt):

    mask = pred == pred
    val_pred = pred[mask]
    val_gt = gt[mask]

    score = accuracy_score(val_gt, val_pred)

    return score