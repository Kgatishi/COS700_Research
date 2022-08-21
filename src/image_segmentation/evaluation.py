from skimage import data, img_as_ubyte
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from itertools import combinations
import numpy as np
from PIL import Image

from .otsu import otsu_get_variance
from .kapur import kapur_get_regions_entropy


OTSU = 0.5 
KAPUR = 1- OTSU
image_histograms  = 0

def evalution_function(solution, solution_idx):
    
    global image_histograms
    hist = image_histograms 
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

    return (regions_var*OTSU + regions_entropy*KAPUR)


# Sample Image of scikit-image package
def Main():
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
    global image_histograms 
    image_histograms = hist
    
    for thresholds in thr_combinations:
        eva_res = evalution_function(thresholds, 0)

        if eva_res > max_eval_res:
            max_eval_res = eva_res
            opt_thresholds = thresholds

    # Print the Results, Best threshold combination
    print("Thresholds :" + str(opt_thresholds))
    print("MAX Evalution :" + str(max_eval_res))

if __name__ == "__main__":
    Main()
