import easydict
import numpy as np
cfg = easydict.EasyDict()

### dirs
cfg.dirs = easydict.EasyDict()

# original data
cfg.dirs.FILE_2015 = './data/quickbird2015.tif'
cfg.dirs.FILE_2017 = './data/quickbird2017.tif'
cfg.dirs.FILE_cadastral2015 = './data/cadastral2015.tif'
cfg.dirs.FILE_tinysample = './data/tinysample.tif'

# labelled patch data
cfg.dirs.FILE_2015patch_img   = './data/2015/'
cfg.dirs.FILE_2017patch_img   = './data/2017/'
cfg.dirs.FILE_2015patch_label = './data/mylabel_2015/'
cfg.dirs.FILE_2017patch_label = './data/mylabel_2017/'

# Full labelled data
cfg.dirs.FILE_label2015 = './data/label/label2015.npy'
cfg.dirs.FILE_label2017 = './data/label/label2017.npy'
cfg.dirs.PCA_img2015 = './data/pca_img2015.npy'
cfg.dirs.PCA_img2017 = './data/pca_img2017.npy'
cfg.dirs.tiny_label = './data/tiny_label.npy'

# checkpoint
cfg.dirs.pretrain_model ='./pretrain_model/'

### data setting
cfg.data = easydict.EasyDict()
cfg.data.data_shape = 5106, 15106
cfg.data.batch_shape = (1, 3, 128, 128)
cfg.data.label_shape = (1, 128, 128)
cfg.data.r = 64
cfg.data.mean = np.array([103.939, 116.779, 123.68])


