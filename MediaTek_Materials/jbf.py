import cv2
import numpy as np
import cv2.ximgproc as xip
import tqdm

ref_path = './gt/'
output_path = './output/'

# for i in range(129):
#         if not i%32:
#             continue
#         idx = str(i)
#         ref_idx = str(i).zfill(3)
#         img_pth = output_path + idx + '.png'
#         ref_img_pth = ref_path + ref_idx + '.png'

#         # print(f'{folder_path = }')
#         img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
#         ref_img = cv2.imread(ref_img_pth, cv2.IMREAD_GRAYSCALE)
        
#         img_filtered = xip.jointBilateralFilter(img, ref_img, 15, 5, 11)
#         cv2.imwrite(f'./filteroutput/d=15_filtered{idx}.png', img_filtered)

img = cv2.imread('./par_output/16.png', cv2.IMREAD_GRAYSCALE)
ref_img = cv2.imread('./16.png', cv2.IMREAD_GRAYSCALE)

img_filtered = xip.jointBilateralFilter(img, ref_img, 5, 11, 11)
cv2.imwrite('./jbf_par_16.png', img_filtered)