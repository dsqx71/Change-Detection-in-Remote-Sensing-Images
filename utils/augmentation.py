import cv2
import random
import numpy as np

from . import data_util
from easydict import EasyDict as edict
from math import *

def between(data, low, high):
    """
    within the boundary or not
    """
    if low <= data and data <= high:
        return True
    else:
        return False

def clip(data, low, high):
    """
    Clip RGB data
    """
    data[data<low] = low
    data[data>high] = high
    return data

class augmentation(object):
    """
    Data augmentation. The design purpose of this class is to decouple augmentation from dataloaders

    Parameters
    ---------
     max_num_tries: int
        Not all spatial transformation coefficients are valid,
        max_num_tries denotes the maximum times it will try to find valid spatial coefficients.
     cropped_width: int
        target width
     cropped_height,
        target height
     data_type : str,
        the allowable values are 'stereo' and 'flow'
     augment_ratio :
        the probability of performing augmentation.
     noise_std: float,
        standard deviation of noise
     interpolation_method : str,
        how to interpolate data,
        the allowable values are 'bilinear' and 'nearest',
     mirror_rate: float
        the probability of performing mirror transformation
     rotate_range:
     translate_range:
     zoom_range:
     squeeze_range:
     gamma_range:
     brightness_range:
     contrast_range:
     rgb_multiply_range:
        dict, there are two forms of the dict:
        {'method':'uniform','low': float, 'high': float}, samples are uniformly distributed over the interval [low, high)
        or
        {'method':'normal', 'mean': float, 'scale': float}, samples are drawed from a normal distribution


    Notes:
    - For stereo matching, introducing any rotation, or vertical shift would break the epipolar constraint
    - When data type is 'stereo', it will ignore rotate_range, and not perform rotation transformation.

    Examples
    ----------
    augment_pipeline = augmentation.augmentation(max_num_tries=50,
                                                 cropped_height=320,
                                                 cropped_width=768,
                                                 data_type='stereo',
                                                 augment_ratio=1.0,
                                                 noise_std=0.0004,
                                                 mirror_rate=0.5,
                                                 rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
                                                 translate_range={'method': 'uniform', 'low': -0.2, 'high': 0.2},
                                                 zoom_range={'method': 'uniform', 'low': 0.8, 'high': 1.5},
                                                 squeeze_range={'method': 'uniform', 'low': 0.75, 'high': 1.25},
                                                 gamma_range={'method': 'uniform', 'low': 0.9, 'high': 1.1},
                                                 brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.002},
                                                 contrast_range={'method': 'uniform', 'low': 0.8, 'high': 1.2},
                                                 rgb_multiply_range={'method': 'uniform', 'low': 0.75, 'high': 1.25},
                                                 interpolation_method='bilinear')

    img1, img2, label = augment_pipeline(img1, img2, label, discount_coeff=0.5)
    """
    def __init__(self,
                 max_num_tries,
                 cropped_width,
                 cropped_height,
                 data_type,
                 augment_ratio,
                 interpolation_method,
                 # spatial transformation
                 mirror_rate,
                 flip_rate,
                 rotate_range,
                 translate_range,
                 zoom_range,
                 squeeze_range,
                 # chromatic transformation
                 gamma_range,
                 brightness_range,
                 contrast_range,
                 rgb_multiply_range,
                 # eigenvector transformation
                 lmult_pow,
                 lmult_mult,
                 lmult_add,
                 sat_pow,
                 sat_mult,
                 sat_add,
                 col_pow,
                 col_mult,
                 col_add,
                 ladd_pow,
                 ladd_mult,
                 ladd_add,
                 col_rotate,
                 noise_range):

        self.augment_ratio = augment_ratio
        self.data_type = data_type
        self.max_num_tries = max_num_tries
        self.cropped_width = cropped_width
        self.cropped_height = cropped_height

        if 'bilinear' in interpolation_method:
            self.interpolation_method = cv2.INTER_LINEAR
        elif 'nearest' in interpolation_method:
            self.interpolation_method = cv2.INTER_NEAREST
        else:
            raise ValueError("Wrong interpolation method")

        # spatial transform
        self.flip_rate = flip_rate
        self.mirror_rate = mirror_rate
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.zoom_range = zoom_range
        self.squeeze_range = squeeze_range

        # chromatic transform
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rgb_multiply_range = rgb_multiply_range

        # eigvector tranform
        self.lmult_pow = lmult_pow
        self.lmult_mult = lmult_mult
        self.lmult_add =  lmult_add
        self.sat_pow = sat_pow
        self.sat_mult = sat_mult
        self.sat_add = sat_add
        self.col_pow = col_pow
        self.col_mult = col_mult
        self.col_add = col_add
        self.ladd_pow = ladd_pow
        self.ladd_mult = ladd_mult
        self.ladd_add = ladd_add
        self.col_rotate = col_rotate

        # noise
        self.noise_range = noise_range

        # eigen vector
        self.eigvec = np.array([[0.51, 0.56, 0.65],
                                [0.79, 0.01,-0.62],
                                [0.35,-0.83, 0.44]])

    def generate_random(self, random_range, size=(1,)):
        """
           random number generator
        """
        mean = random_range['mean']
        spread = random_range['spread'] * self.discount_coeff
        result = np.zeros(size)
        if random_range['method'] == 'uniform':
            cv2.randu(result, mean-spread, mean+spread)
        elif random_range['method'] == 'normal':
            cv2.randn(result, mean=mean, stddev=spread)
        else:
            raise ValueError("Wrong sampling method")
        result = np.exp(result) if random_range['exp'] else result

        if size == (1, ):
            result = result[0]
        return result

    def generate_spatial_coeffs(self):
        # order: mirror, rotate, translate, zoom, squeeze
        for i in range(self.max_num_tries):
            # identity matrix
            coeff = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

            # mirror
            if np.random.uniform(0, 1) < self.mirror_rate * self.discount_coeff:
                mirror = np.array([[-1, 0,  0.5*self.cropped_width],
                                   [ 0, 1, -0.5*self.cropped_height],
                                   [ 0, 0, 1]])
            else:
                # move the center to (0, 0)
                mirror = np.array([[1, 0, -0.5*self.cropped_width],
                                   [0, 1, -0.5*self.cropped_height],
                                   [0, 0, 1]])
            coeff = np.dot(mirror, coeff)

            # vertical flip
            if np.random.uniform(0, 1) < self.flip_rate * self.discount_coeff:
                flip = np.array([[1, 0, 0],
                                 [0,-1, 0],
                                 [0, 0, 1]])
                coeff = np.dot(flip, coeff)

            # rotate
            if self.data_type == 'flow':
                angle = self.generate_random(self.rotate_range) / 180.0 * pi
                rotate = np.array([[cos(angle), - sin(angle), 0],
                                   [sin(angle),   cos(angle), 0],
                                   [0, 0, 1]])
                coeff = np.dot(rotate, coeff)

            # translate
            dx = self.generate_random(self.translate_range)
            dy = self.generate_random(self.translate_range)
            translate = np.array([[1, 0, dx*self.cropped_width],
                                  [0, 1, dy*self.cropped_height],
                                  [0, 0, 1]])
            coeff = np.dot(translate, coeff)

            # zoom
            zoom_x = self.generate_random(self.zoom_range)
            zoom_y = zoom_x

            # squeeze
            squeeze_coeff = self.generate_random(self.squeeze_range)
            zoom_x *= squeeze_coeff
            zoom_y /= squeeze_coeff
            zoom = np.array([[1.0/zoom_x, 0, 0],
                             [0, 1.0/zoom_y, 0],
                             [0, 0, 1]])
            coeff = np.dot(zoom, coeff)

            # move_back
            move_back = np.array([[1, 0, self.width*0.5],
                                  [0, 1, self.height*0.5],
                                  [0, 0, 1]])
            coeff = np.dot(move_back, coeff)

            # Four corners should not exceed the boundaries of the origin
            flag = True
            for x in [0, self.cropped_width - 1]:
                for y in [0, self.cropped_height - 1]:
                    dest = np.array([x, y, 1])
                    src = np.dot(coeff, dest)
                    if between(src[0], 1, self.width - 2) == False or between(src[1], 1, self.height - 2) == False:
                        flag = False
                        break
            if flag:
                return coeff
        return None

    def spatial_transform(self, img1, img2, label):
        """
        coeff =  a1, a2, t1,
                 a3, a4, t2,
                 0,   0,  1
        a1, a2, a3, a4 : rotate, zoom, squeeze ; t1, t2 : crop and translate

        src_grid = np.dot(coeff, dst_grid)
        """
        coeff = self.generate_spatial_coeffs()
        if coeff is not None:
            grid = np.zeros((3, self.cropped_height, self.cropped_width))
            xv, yv = np.meshgrid(np.arange(self.cropped_height), np.arange(self.cropped_width))
            grid[0, :, :] = yv.T
            grid[1, :, :] = xv.T
            grid[2, :, :] = 1.0
            grid = grid.reshape(3, -1)
            grid = np.dot(coeff, grid).astype(np.float32)
            grid = grid.reshape((3, self.cropped_height, self.cropped_width))
            img1_result = cv2.remap(img1, map1 = grid[0], map2=grid[1], interpolation=self.interpolation_method,
                                    borderValue=0)
            img2_result = cv2.remap(img2, map1=grid[0], map2=grid[1], interpolation=self.interpolation_method,
                                    borderValue=0)
            if label is not None :
                label_result = cv2.remap(label, map1=grid[0], map2=grid[1], interpolation=self.interpolation_method,
                                         borderValue=np.nan)
                if self.data_type == 'stereo':
                    label_result /= coeff[0,0]
                elif self.data_type == 'flow':
                    label_result = np.dot(label_result.reshape(-1,2), np.linalg.inv(coeff[:2, :2]).T)
                    label_result = label_result.reshape((self.cropped_height, self.cropped_width, 2))
            else:
                label_result = None

            return img1_result, img2_result, label_result
        else:
            # print("Augmentation: Exceeded maximum tries in finding spatial coeffs.")
            img1_result, img2_result, label_result = data_util.crop(img1, img2, label, target_height=self.cropped_height,
                                                                                       target_width=self.cropped_width)
            return img1_result, img2_result, label_result

    def generate_chromatic_coeffs(self):

        coeff = edict()
        coeff.gamma = self.generate_random(self.gamma_range)
        coeff.brightness = self.generate_random(self.brightness_range)
        coeff.contrast = self.generate_random(self.contrast_range)
        coeff.rgb = np.array([self.generate_random(self.rgb_multiply_range) for i in range(3)])
        return coeff

    def apply_chromatic_transform(self, img, coeff):
        # normalize into [0, 1]
        img = img / 255.0
        # color change
        brightness_in = img.sum(axis=2)
        img = img * coeff.rgb
        brightness_out = img.sum(axis=2)
        brightness_coeff = brightness_in / (brightness_out + 1E-5)
        brightness_coeff = np.expand_dims(brightness_coeff, 2)
        brightness_coeff = np.concatenate([brightness_coeff for i in range(3)], axis=2)
        # compensate brightness
        img = img * brightness_coeff
        img = clip(img, 0, 1.0)
        # gamma change
        img = cv2.pow(img, coeff.gamma)
        # brightness change
        img = cv2.add(img, coeff.brightness)
        # contrast change
        img = 0.5 + (img-0.5) * coeff.contrast
        img = clip(img, 0.0, 1.0)
        img = img * 255
        return img

    def chromatic_transform(self, img1, img2):
        coeff = self.generate_chromatic_coeffs()
        if random.uniform(0,1) < 0.5:
            img2 = self.apply_chromatic_transform(img2, coeff)
        else:
            img1 = self.apply_chromatic_transform(img1, coeff)
        return img1, img2

    def generate_eigenvec_coeffs(self):

        coeff = edict()
        coeff.pow_nomean = np.array([self.generate_random(self.ladd_pow),
                                     self.generate_random(self.col_pow),
                                     self.generate_random(self.col_pow)])
        coeff.add_nomean = np.array([self.generate_random(self.ladd_add),
                                     self.generate_random(self.col_add),
                                     self.generate_random(self.col_add)])
        coeff.multi_nomean = np.array([self.generate_random(self.ladd_mult),
                                       self.generate_random(self.col_mult),
                                       self.generate_random(self.col_mult)])
        tmp = self.generate_random(self.sat_pow)
        coeff.pow_withmean = np.array([tmp, tmp])

        tmp = self.generate_random(self.sat_add)
        coeff.add_withmean = np.array([tmp, tmp])

        tmp = self.generate_random(self.sat_mult)
        coeff.mult_withmean = np.array([tmp, tmp])

        coeff.lmult_pow = self.generate_random(self.lmult_pow)
        coeff.lmult_mult = self.generate_random(self.lmult_mult)
        coeff.lmult_add = self.generate_random(self.lmult_add)
        coeff.col_angle = self.generate_random(self.col_rotate)

        return coeff

    def apply_eigenvec_transform(self, coeff, img):

        shape = img.shape
        img = img.reshape(-1, 3)
        mean_rgb = img.mean(axis=0)
        eig = np.dot(self.eigvec, img.T)
        max_abs_eig = np.abs(eig).max(axis=1)

        mean_eig = np.dot(self.eigvec, mean_rgb) / (max_abs_eig + 1E-7)
        max_length = np.linalg.norm(max_abs_eig, ord=2)

        #  doing the nomean stuff
        img = img - mean_rgb
        eig = np.dot(self.eigvec, img.T)
        eig = eig.T
        eig = eig / (max_abs_eig + 1E-7)
        eig = eig.reshape(shape)
        for i in range(3):
            eig[:, :, i] = cv2.pow(np.abs(eig[:, :, i]), coeff.pow_nomean[i]) * np.sign(eig[:, :, i])
        eig = eig + coeff.add_nomean
        eig = eig * coeff.multi_nomean

        # re-adding the mean
        eig = eig + mean_eig
        eig[:, :, 0] = cv2.pow(np.abs(eig[:, :, 0]), coeff.pow_withmean[0]) * np.sign(eig[:, :, 0])
        eig[:, :, 0] = eig[:, :, 0] + coeff.add_withmean[0]
        eig[:, :, 0] = eig[:, :, 0] * coeff.mult_withmean[0]

        # doing the withmean stuff
        s = np.sqrt(eig[:, :, 1] * eig[:, :, 1] + eig[:, :, 2] * eig[:, :, 2])
        s1 = s.copy()
        s1 = cv2.pow(s1, coeff.pow_withmean[1])
        s1 = s1 + coeff.add_withmean[1]
        s1[s1 < 0] = 0
        s1 = s1 * coeff.mult_withmean[1]

        if coeff.col_angle !=0 :
            tmp1 = cos(coeff.col_angle) * eig[:, :, 1] - sin(coeff.col_angle) * eig[:, :, 2]
            tmp2 = sin(coeff.col_angle) * eig[:, :, 1] + cos(coeff.col_angle) * eig[:, :, 2]
            eig[:, :, 1] = tmp1
            eig[:, :, 2] = tmp2

        eig = eig * max_abs_eig
        l1 = np.linalg.norm(eig, axis=2, ord=2)
        l1 = l1 / (max_length + 1E-7)

        for i in range(1, 3):
            eig[:, :, i] = eig[:, :, i] / (s + 1E-7) * s1

        l = np.linalg.norm(eig, ord=2, axis=2)
        l1 = cv2.pow(l1, coeff.lmult_pow)
        l1 = l1 + coeff.lmult_add
        l1[ l1 < 0] = 0
        l1 = l1 * coeff.lmult_mult
        l1 = l1 * max_length

        for i in range(3):
            eig[:, :, i] = eig[:, :, i] / (l + 1E-7) * l1
            eig[:, :, i] = np.where(eig[:, :, i] < max_abs_eig[i], eig[:, :, i], max_abs_eig[i])
        # convert to RGB
        rgb = np.dot(eig, self.eigvec)
        rgb = clip(rgb, 0, 255)
        rgb = rgb.reshape(shape)
        return rgb

    def eigenvec_transform(self, img1, img2):

        coeff = self.generate_eigenvec_coeffs()
        img1 = self.apply_eigenvec_transform(coeff, img1)
        img2 = self.apply_eigenvec_transform(coeff, img2)

        return img1, img2

    def __call__(self, img1, img2, label, discount_coeff):
        """
        Perform data augmentation

        Parameters
        ----------
        img1: numpy.ndarray
        img2: numpy.ndarray
        label: numpy.ndarray
        discount_coeff : float
            the discount of augmentation coefficients
        """
        self.height = img1.shape[0]
        self.width = img1.shape[1]
        self.discount_coeff = discount_coeff
        if np.random.uniform(0, 1) < self.augment_ratio * self.discount_coeff:
            img1, img2, label = self.spatial_transform(img1, img2, label)
            img1, img2 = self.eigenvec_transform(img1, img2)
            img1, img2 = self.chromatic_transform(img1, img2)
            noise = self.generate_random(self.noise_range)
            img1 += noise
            img2 += noise
        else:
            img1, img2, label = data_util.crop(img1, img2, label, self.cropped_height, self.cropped_width)
        return img1, img2, label
