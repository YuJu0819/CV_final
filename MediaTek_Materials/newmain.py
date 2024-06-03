from utils_new import apply_gmc
import cv2
import os
import numpy as np
from skimage.exposure import match_histograms
from tqdm import tqdm

def psnr(img1, img2):
    mse = np.mean(np.square(img1 - img2))
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)

    return psnr

def main():
    frames = []
    folder_path = './gt/'
    for i in tqdm(range(129)):
        idx = str(i).zfill(3)
        img_pth = folder_path + idx + '.png'

        # print(f'{folder_path = }')
        img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
        else:
            print(f'Warning: {img_pth} could not be loaded.')

    target_frame = frames[5]
    prev_frame = frames[4]
    next_frame = frames[6]
    compensated_frame = apply_gmc(
        target_frame, prev_frame, next_frame)
        
    # cv2.imwrite('./output/5_nopar.png', compensated_frame)
    cv2.imwrite('./output/5_test.png', compensated_frame)

    PSNR = psnr(frames[5], compensated_frame)
    print(f'PSNR: {PSNR}')
    print('complete')

    # compensated_matched = match_histograms(compensated_frame, frames[16])

    # # Mean brightness correction
    # mean_brightness_diff = np.mean(frames[16]) - np.mean(compensated_matched)
    # compensated_corrected = np.clip(compensated_matched + mean_brightness_diff, 0, 255).astype(np.uint8)

    # # Smooth block boundaries
    # compensated_smoothed = cv2.bilateralFilter(compensated_corrected, d=9, sigmaColor=75, sigmaSpace=75)

    # # Blend with the previous frame
    # blending_weight = 0.5
    # compensated_final = (compensated_corrected * blending_weight + compensated_smoothed * (1 - blending_weight)).astype(np.uint8)

    # cv2.imwrite('./output/smoothen_16.png', compensated_final)
    

if __name__ == '__main__':
    main()