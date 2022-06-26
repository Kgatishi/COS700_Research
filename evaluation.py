# Importing Necessary Libraries
# Displaying the sample image - Monochrome Format
from skimage import data
from skimage.color import rgb2gray

from itertools import combinations
import numpy as np

from otsu import otsu_get_variance
from kapur import kapur_get_regions_entropy

def evalution_function(gray_image, thresholds, otsu=0.5, kapur=0.5):
    # Histogran
    hist, _ = np.histogram(gray_image, bins=range(256), density=True)
    
    # Cumulative histogra
    c_hist = hist.cumsum()
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    # Extending histograms for convenience
    c_hist = np.append(c_hist, [0])
    cdf = np.append(cdf, [0])

    # Extending thresholds for convenience
    e_thresholds = [-1]
    e_thresholds.extend(thresholds)
    e_thresholds.extend([len(hist) - 1])

    # Computing variance for the current combination of thresholds
    regions_var = otsu_get_variance(hist, c_hist, cdf, e_thresholds)

    # Computing regions entropy for the current combination of thresholds
    regions_entropy = kapur_get_regions_entropy(hist, c_hist, e_thresholds)

    return (regions_var*otsu + regions_entropy*kapur)


# Sample Image of scikit-image package
if __name__ == __name__:
    coffee = data.coffee()
    gray_image = rgb2gray(coffee)
    print (type(gray_image))
    nthrs = 2
    thr_combinations = combinations(range(255), nthrs)

    
    max_eval_res = 0
    opt_thresholds = None

    for thresholds in thr_combinations:
        eva_res = evalution_function(gray_image,thresholds)

        if eva_res > max_eval_res:
            max_eval_res = eva_res
            opt_thresholds = thresholds