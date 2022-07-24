from skimage import data, img_as_ubyte
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from itertools import combinations
import numpy as np
from PIL import Image

from otsu import otsu_get_variance
from kapur import kapur_get_regions_entropy

class Evaluation:
    
    def __init__(self, histogram_ , otsu=0.5, kapur=0.5):
        self.otsu = otsu 
        self.kapur = kapur
        self.images_histograms  = histogram_


    def evalution_function(self, solution, solution_idx):
        
        hist = self.images_histograms 
        thresholds = np.sort(solution) 

        # Cumulative histogram
        c_hist = np.cumsum(hist)
        cdf = np.cumsum(np.arange(len(hist)) * hist)

        # Extending histograms for convenience
        c_hist = np.append(c_hist, [0])
        cdf = np.append(cdf, [0])

        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # OTSU: variance for the current combination of thresholds
        regions_var = otsu_get_variance(hist, c_hist, cdf, e_thresholds)
        # KAPUR: regions entropy for the current combination of thresholds
        regions_entropy = kapur_get_regions_entropy(hist, c_hist, e_thresholds)

        return (regions_var*self.otsu + regions_entropy*self.kapur)

    """Get the total entropy of regions for a given set of thresholds"""
    def otsu_get_variance(self, hist, c_hist, cdf, thresholds):
        variance = 0

        for i in range(len(thresholds) - 1):
            # Thresholds
            t1 = thresholds[i] + 1
            t2 = thresholds[i + 1]

            weight = c_hist[t2] - c_hist[t1 - 1]                # Cumulative histogram
            r_cdf = cdf[t2] - cdf[t1 - 1]                       # Region CDF
            r_mean = r_cdf / weight if weight != 0 else 0       # Region mean

            variance += weight * r_mean ** 2

        return variance
    
    """Get the total entropy of regions for a given set of thresholds"""
    def kapur_get_regions_entropy(self, hist, c_hist, thresholds):

        total_entropy = 0
        for i in range(len(thresholds) - 1):
            # Thresholds
            t1 = thresholds[i] + 1
            t2 = thresholds[i + 1]

            # print(thresholds, t1, t2)
            hc_val = c_hist[t2] - c_hist[t1 - 1]                        # Cumulative histogram
            h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1       # Normalized histogram
            entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()     # entropy

            total_entropy += entropy

        return total_entropy

# Sample Image of scikit-image package
if __name__ == __name__:
    # Getting image
    coffee = data.coffee()

    # Convert using skimage
    gray_image = img_as_ubyte(rgb2gray(coffee))
    print (type(gray_image))
    print (gray_image.shape)
    #print (gray_image)

    # Convert using numpy
    img = np.array(Image.fromarray(coffee).convert('L'))
    print (type(img))
    print (img.shape)
    #print (img)

    # Convert image to Histogran
    hist, _ = np.histogram(img, bins=range(256), density=True)
    
    # Generate Random threshold combinations for testing
    nthrs = 1
    thr_combinations = combinations(range(255), nthrs)

    # Best performing threshold combination
    max_eval_res = 0
    opt_thresholds = None

    # Evaluation threshold combination
    Eval = Evaluation(hist)
    for thresholds in thr_combinations:
        eva_res = Eval.evalution_function(thresholds, 0)

        if eva_res > max_eval_res:
            max_eval_res = eva_res
            opt_thresholds = thresholds

    # Print the Results, Best threshold combination
    print("Thresholds :" + str(opt_thresholds))
    print("MAX Evalution :" + str(max_eval_res))