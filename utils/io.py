import numpy as np
import tifffile as tiff
from config import cfg

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

def read_rawdata(FILE_2015=cfg.dirs.FILE_2015,
                 FILE_2017=cfg.dirs.FILE_2017,
                 FILE_tinysample=cfg.dirs.FILE_tinysample,
                 FILE_cadastral2015=cfg.dirs.FILE_cadastral2015):

    # Read raw data
    im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])

    im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])

    im_tiny = tiff.imread(FILE_tinysample)

    im_cada = tiff.imread(FILE_cadastral2015)

    assert im_2015.shape[:2] == cfg.data.data_shape
    assert im_2015.shape[:2] == cfg.data.data_shape
    assert im_2015.shape[:2] == cfg.data.data_shape

    return im_2015, im_2017, im_tiny, im_cada

