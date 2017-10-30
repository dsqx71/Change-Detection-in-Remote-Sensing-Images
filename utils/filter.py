from guided_filter.core import filters
import numpy as np

def weight_median_filter(i, left, radius, epsilon, mask):
    """
    Constant Time Weighted Median Filtering for Stereo Matching and Beyond
    Parameters
    ----------
    i : ndarray
        disparity
    left : ndarray
        original image
    radius : int
    epsilon : float
    mask: ndarray of boolean
        indicate which need to be changed
    Returns
    -------
    dispout : ndarray
        filted disparity
    """
    dispin  = i.copy()
    dispout = dispin.copy()
    dispout[mask] = 0
    vecdisp = np.unique(dispin)

    tot = np.zeros(i.shape)
    imgaccum = np.zeros(i.shape)

    gf = filters.GuidedFilterColor(left.copy(), radius, epsilon)

    for d in vecdisp:
        if d<=0:
            continue
        ab = gf._computeCoefficients((dispin==d).astype(float))
        weight = gf._computeOutput(ab, gf._I)
        tot = tot + weight

    for d in vecdisp:
        if d<=0:
            continue
        ab = gf._computeCoefficients((dispin==d).astype(float))
        weight = gf._computeOutput(ab, gf._I)
        imgaccum = imgaccum + weight
        musk =  (imgaccum > 0.5*tot) & (dispout==0) & (mask) & (tot> 0.0001)
        dispout[musk] = d

    return dispout