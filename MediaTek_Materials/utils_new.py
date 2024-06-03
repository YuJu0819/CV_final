import numpy as np
import cv2
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import img_as_float

# Example function to load YUV frames
def load_yuv_frames(filepath, width, height, num_frames):
    frames = []
    frame_size = width * height
    with open(filepath, 'rb') as f:
        for _ in range(num_frames):
            y = np.frombuffer(f.read(frame_size), dtype=np.uint8).reshape((height, width))
            frames.append(y)
    return frames

# Function to divide image into 16x16 blocks with boundary handling
def divide_into_blocks(image, block_size=16):
    blocks = []
    h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            blocks.append((i, j, block))
    return blocks

# Example function to generate a selection map (binary map indicating easy and difficult blocks)
def generate_selection_map(image, block_size=16):
    h, w = image.shape
    stds = []  # List to hold standard deviations and block coordinates

    # Calculate the standard deviation for each block and store it with its coordinates
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            std_dev = np.std(block)
            stds.append((i, j, block, std_dev))

    # Sort blocks by standard deviation in ascending order (easiest first)
    stds.sort(key=lambda x: x[3])
    stds = stds[::-1]

    # Create a selection map and mark the easiest blocks
    selection_map = np.zeros((h // block_size, w // block_size), dtype=np.uint8)
    for idx, (i, j, block, std_dev) in enumerate(stds):
        # Mark block as easy (1)
        selection_map[i // block_size, j // block_size] = 1

    return selection_map

# Compute Sum of Block Sums
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

def compute_sad(current_block, reference_block):
    return np.sum(np.abs(current_block - reference_block))

# Block matching using SAD
def block_matching(current_frame, reference_frame, block_size, search_range = 16, reftype = 'prev'):
    height, width = current_frame.shape
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)

    # for i in range(0, height, block_size):
    #     for j in range(0, width, block_size):
    #         current_block = current_frame[i:i + block_size, j:j + block_size]
    #         min_sad = float('inf')
    #         best_vector = (0, 0)
            
    #         for dy in range(-search_range, search_range + 1):
    #             for dx in range(-search_range, search_range + 1):
    #                 ref_i = i + dy
    #                 ref_j = j + dx
                    
    #                 if ref_i < 0 or ref_i + block_size > height or ref_j < 0 or ref_j + block_size > width:
    #                     continue
                    
    #                 reference_block = reference_frame[ref_i:ref_i + block_size, ref_j:ref_j + block_size]
    #                 sad = compute_sad(current_block, reference_block)
                    
    #                 if sad < min_sad:
    #                     min_sad = sad
    #                     best_vector = (dy, dx)
            
    #         motion_vectors[i // block_size, j // block_size] = best_vector
    padded_reference = cv2.copyMakeBorder(reference_frame, search_range, search_range, 2*search_range, 2*search_range, cv2.BORDER_CONSTANT, value=np.inf)
    # Initialize the best SAD to infinity and the best motion vectors to zero
    best_sad = np.ones((height // block_size, width // block_size))*np.inf
    best_dy = np.zeros((height // block_size, width // block_size), dtype=np.int32)
    best_dx = np.zeros((height // block_size, width // block_size), dtype=np.int32)

    for dy in range(-search_range, search_range + 1):
        rolled_frame_y = np.roll(padded_reference, dy, axis=0)
        for dx in range(-search_range, search_range + 1):
            rolled_frame_xy = np.roll(rolled_frame_y, dx, axis=1)
            # Extract the central part of the rolled frame that matches the size of the current frame
            rolled_frame_xy_cropped = rolled_frame_xy[search_range+dy:search_range+dy + height, dx+search_range:dx+search_range + width]
            sad = np.abs(current_frame - rolled_frame_xy_cropped)
            sad_sum = sum_blocks(sad, block_size)
            mask = sad_sum < best_sad
            best_sad[mask] = sad_sum[mask]
            best_dy[mask] = dy
            best_dx[mask] = dx

    # Apply the best motion vectors to each block
    motion_vectors[..., 0] = best_dy
    motion_vectors[..., 1] = best_dx

    return motion_vectors

# Apply global motion compensation
def apply_gmc(current_frame, previous_frame, next_frame, block_size=16, search_range=16):
    motion_vectors_prev = block_matching(current_frame, previous_frame, block_size, search_range, 'prev')
    motion_vectors_next = block_matching(current_frame, next_frame, block_size, search_range, 'next')
    height, width = current_frame.shape
    compensated_frame = np.zeros_like(current_frame)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            dy_prev, dx_prev = motion_vectors_prev[i // block_size, j // block_size]
            dy_next, dx_next = motion_vectors_next[i // block_size, j // block_size]
            
            ref_i_prev = i + dy_prev
            ref_j_prev = j + dx_prev
            ref_i_next = i + dy_next
            ref_j_next = j + dx_next
            
            if (ref_i_prev < 0 or ref_i_prev + block_size > height or ref_j_prev < 0 or ref_j_prev + block_size > width):
                block_prev = current_frame[i:i + block_size, j:j + block_size]
            else:
                block_prev = previous_frame[ref_i_prev:ref_i_prev + block_size, ref_j_prev:ref_j_prev + block_size]
                
            if (ref_i_next < 0 or ref_i_next + block_size > height or ref_j_next < 0 or ref_j_next + block_size > width):
                block_next = current_frame[i:i + block_size, j:j + block_size]
            else:
                block_next = next_frame[ref_i_next:ref_i_next + block_size, ref_j_next:ref_j_next + block_size]
            
            # Combine the two blocks
            
            compensated_frame[i:i + block_size, j:j + block_size] = block_prev // 2 + block_next // 2

    return compensated_frame