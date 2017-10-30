import numpy as np
import random
from config import cfg
import tifffile as tiff

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

    assert im_2015.shape[:2] == tuple(cfg.data.data_shape)
    assert im_2015.shape[:2] == tuple(cfg.data.data_shape)
    assert im_2015.shape[:2] == tuple(cfg.data.data_shape)

    return im_2015, im_2017, im_tiny, im_cada

def read_data():

    pca_img2015 = np.load(cfg.dirs.PCA_img2015)
    pca_img2017 = np.load(cfg.dirs.PCA_img2017)
    tiny_label = np.load(cfg.dirs.tiny_label)

    return pca_img2015, pca_img2017, tiny_label

def sample_label(pca_img2015, pca_img2017, label, num_pos, num_neg):

    data = []
    labeled_points = np.argwhere(label == label)
    explore = [(random.randint(cfg.data.r, label.shape[0] - cfg.data.r),
                random.randint(cfg.data.r, label.shape[1] - cfg.data.r)) for i in range(num_neg+100)]
    rnd_range = [i for i in range(labeled_points.shape[0])]
    random.shuffle(rnd_range)
    rnd_range = rnd_range[:num_pos] + explore
    count = 0

    for i in rnd_range:

        if type(i) == int:
            rnd_y = random.randint(-cfg.data.r + 20, cfg.data.r - 20)
            rnd_x = random.randint(-cfg.data.r + 20, cfg.data.r - 20)
            y, x = labeled_points[i]
            y += rnd_y
            x += rnd_x
        else:
            y, x = i

        img1_patch = pca_img2015[y - cfg.data.r:y + cfg.data.r, x - cfg.data.r:x + cfg.data.r]
        img2_patch = pca_img2017[y - cfg.data.r:y + cfg.data.r, x - cfg.data.r:x + cfg.data.r]
        label_patch = label[y - cfg.data.r:y + cfg.data.r, x - cfg.data.r:x + cfg.data.r]

        if img1_patch.size == 3 * cfg.data.batch_shape[2] ** 2:
            data.append([img1_patch, img2_patch, label_patch, y, x])
            count += 1

        if count == num_neg + num_pos:
            break

    return data