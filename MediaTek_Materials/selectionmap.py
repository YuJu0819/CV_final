import numpy as np
import cv2

# import os
output = './par_output/'
gt = './gt/'

def sum_blocks(matrix, block_size):
    h, w = matrix.shape
    assert h % block_size == 0 and w % block_size == 0, "The matrix dimensions must be divisible by the block size."
    
    # Reshape the matrix to separate each block
    # reshaped_matrix = matrix.reshape(h // block_size, block_size, -1, block_size)
    reshaped_matrix = matrix.reshape(block_size, block_size, h// block_size, -1)
    # reshaped_matrix = reshaped_matrix.swapaxes(1, 2)
    
    # Sum the elements within each block
    block_sums = reshaped_matrix.sum(axis=(0, 1)).squeeze()
    # block_sums = reshaped_matrix.sum(axis=(2, 3))
    
    return block_sums

for i in range(129):
    if i%32==0:
        continue
    # print(i, type(i))
    img_pth = output + str(i) + '.png'
    print(img_pth)
    gt_idx = str(i).zfill(3)
    gt_pth = gt + gt_idx + '.png'
    print(gt_pth)
    img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
    gt_img = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)

    se = np.square(img-gt_img)
    block_se = sum_blocks(se, 16)
    block_se = block_se.reshape((1,-1))
    print(block_se)
    best_se = np.argsort(block_se.squeeze())[:13000]
    print(best_se)

    # best_se = np.sort(np.argsort(block_se.reshape((1, -1)[0]))[:13000])
    # # selectionmap = np.zeros(np.prod(img.shape // 16), dtype=int)
    # print(len(selectionmap))
    # selectionmap[best_se] = 1

    select_path = './solution/s_' + str(i).zfill(3) + '.txt'


    with open(select_path, 'w') as file:
        for line in range(32400):
            if line in best_se:
                file.write('1\n')
            else:
                file.write('0\n')
        file.close()
    
        
    # np.savetxt(select_path, selectionmap, fmt='%d')

    