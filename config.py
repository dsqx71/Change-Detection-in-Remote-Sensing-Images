import easydict
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


### data setting
cfg.data = easydict.EasyDict()
cfg.data.data_shape = tuple((5106, 15106))


