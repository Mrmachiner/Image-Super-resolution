from math import log10, sqrt 
import cv2 
import numpy as np 
from skimage.measure import compare_ssim, compare_psnr, compare_nrmse, compare_mse
def PSNR(original, compressed): # image original and image compressed
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel/sqrt(mse))
    return psnr
compare_psnr()