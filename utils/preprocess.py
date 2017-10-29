import cv2
import numpy as np
from . import io
from config import cfg
from sklearn.decomposition import PCA

def concat_patche():

    s = 960
    data2015 = np.zeros(cfg.data.data_shape + [3])
    data2017 = np.zeros(cfg.data.data_shape + [3])

    label2015 = np.zeros(cfg.data.data_shape)
    label2017 = np.zeros(cfg.data.data_shape)

    for i in range(6):
        for j in range(16):

            Patch2015 = cv2.imread(cfg.dirs.FILE_2015patch_img + '{}_{}_960_.jpg'.format(i, j), 1)
            Label2015 = cv2.imread(cfg.dirs.FILE_2015patch_label + '{}_{}_960_.jpg'.format(i, j), 0)

            h, w = Patch2015.shape[:2]
            if (i != 5 and j != 15):
                assert (h == s and w == s)

            _, binary = cv2.threshold(Label2015, 128, 1, cv2.THRESH_BINARY)

            data2015[s * i: s * i + h, s * j: s * j + w, :3] = Patch2015
            label2015[s * i: s * i + h, s * j: s * j + w] = binary

            Patch2017 = cv2.imread(cfg.dirs.FILE_2017patch_img + '{}_{}_960_.jpg'.format(i, j), 1)
            Label2017 = cv2.imread(cfg.dirs.FILE_2017patch_label + '{}_{}_960_.jpg'.format(i, j), 0)

            h, w = Patch2017.shape[:2]
            if (i != 5 and j != 15):
                assert (h == s and w == s)

            _, binary = cv2.threshold(Label2017, 128, 1, cv2.THRESH_BINARY)

            data2017[s * i: s * i + h, s * j: s * j + w, :3] = Patch2017
            label2017[s * i: s * i + h, s * j: s * j + w] = binary

            # The two images cannot be completely the same
            assert (np.abs(Patch2015 - Patch2017) > 0).any()
            if (Label2015 is not None):
                assert (np.abs(Label2015 - Label2017) > 0).any()

    assert data2015.shape[:2] == tuple(cfg.data.data_shape)
    assert data2017.shape[:2] == tuple(cfg.data.data_shape)
    assert label2015.shape == tuple(cfg.data.data_shape)
    assert label2017.shape == tuple(cfg.data.data_shape)
    assert np.abs(label2015 - label2017).any()

    return label2015, label2017

def pca_transform():

    im_2015, im_2017, im_tiny, im_cada = io.read_rawdata()
    randn = np.random.uniform(0, 1, size=(cfg.data.data_shape[0], cfg.data.data_shape[1]))
    mask = randn < 0.1
    tmp = np.r_[im_2015[mask], im_2017[mask]]
    pca = PCA(n_components=3)
    pca.fit(tmp)

    tmp2015 = pca.transform(im_2015.reshape((cfg.data.data_shape[0] * cfg.data.data_shape[1], 4))).reshape(
        (cfg.data.data_shape[0], cfg.data.data_shape[1], 3))
    tmp2017 = pca.transform(im_2017.reshape((cfg.data.data_shape[0] * cfg.data.data_shape[1], 4))).reshape(
        (cfg.data.data_shape[0], cfg.data.data_shape[1], 3))

    tmp2015 = io.scale_percentile(tmp2015) * 255
    tmp2017 = io.scale_percentile(tmp2017) * 255

    label = (im_tiny > 0).astype(float)
    label = label[:,:,0]
    label[label == 0] = np.nan

    return tmp2015, tmp2017, label
    
    
if __name__ == '__main__':

    label2015, label2017 = concat_patche()
    np.save(cfg.dirs.FILE_label2015, label2015)
    np.save(cfg.dirs.FILE_label2017, label2017)

    pca_img2015, pca_img2017, tiny_label = pca_transform()

    np.save(cfg.dirs.PCA_img2015, pca_img2015)
    np.save(cfg.dirs.PCA_img2017, pca_img2017)
    np.save(cfg.dirs.tiny_label, tiny_label)

    print('Positive sample in 2015 : %f' % ((label2015 == 1).sum() / float(label2015.size)))
    print('Positive sample in 2017 : %f' % ((label2017 == 1).sum() / float(label2017.size)))
