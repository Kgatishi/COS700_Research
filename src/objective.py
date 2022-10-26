from math import log10, sqrt
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from skimage import data, io, img_as_float
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# Peak Signal-to-Noise Ratio
def PSNR(original, segmented):
    mse = np.mean((original - segmented) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# Structural similarity index
def SSIM(original, segmented):
    ssim_val = ssim(original, segmented,data_range=segmented.max() - segmented.min())
    return ssim_val

def algo(original, segmented):pass

def algo2(original, segmented):pass

def algorithm_thresholds(num_thresh,algorithm):
    pass

def matrix(im):
    img = io.imread(im)
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    algo = ["GA","GE","SA","TS"]
    thres = [1,3,5,7]
    for t in thres:
        for a in algo:
            thres_v = algorithm_thresholds(t,a)
            # Apply image segmentation to image based on this thresholds
            img_converted = 0

            # get matrices
            mse = mean_squared_error(img, img)
            ssim =SSIM(img,img_converted)
            psnr =PSNR(img,img_converted)


def main():
    img = img_as_float(data.camera())
    rows, cols = img.shape

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    rng = np.random.default_rng()
    noise[rng.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + noise
    img_const = img + abs(noise)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),sharex=True, sharey=True)
    ax = axes.ravel()
    


    mse_none = mean_squared_error(img, img)
    ssim_none = SSIM(img, img)
    psnr_none = PSNR(img, img)
    print (img.max())
    print (img.min())

    mse_noise = mean_squared_error(img, img_noise)
    ssim_noise = SSIM(img, img_noise)
    psnr_noise = PSNR(img, img_noise)
    print (img_noise.max())
    print (img_noise.min())

    mse_const = mean_squared_error(img, img_const)
    ssim_const = SSIM(img, img_const)
    psnr_const = PSNR(img, img_const)
    print (img_const.max())
    print (img_const.min())

    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(f'MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}, PSNR: {psnr_none:.2f}')
    ax[0].set_title('Original image')

    ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(f'MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f} , PSNR: {psnr_noise:.2f}')
    ax[1].set_title('Image with noise')

    ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[2].set_xlabel(f'MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}, PSNR: {psnr_const:.2f}')
    ax[2].set_title('Image plus constant')

    plt.tight_layout()
    plt.show()

    print(f'Original image (min:,max:)>> MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}, PSNR: {psnr_none:.2f}')
    print(f'Image with noise (min:,max:)>> MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}, PSNR: {psnr_noise:.2f}')
    print(f'Image plus constant (min:,max:)>> MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}, PSNR: {psnr_const:.2f}')

if __name__ == "__main__":
    main()