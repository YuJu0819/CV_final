import numpy as np
import cv2
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import AffineTransform
from skimage import img_as_float
from skimage.measure import ransac
from scipy.signal import convolve2d
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as transforms
from PIL import Image

BLOCK_SIZE = 16

# Function to divide image into 16x16 blocks with boundary handling
def divide_into_blocks(image, block_size=BLOCK_SIZE):
    blocks = []
    h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            blocks.append(((i, j), block))
    return blocks

# Example function to generate a selection map (binary map indicating easy and difficult blocks)
def generate_selection_map(image, block_size=BLOCK_SIZE):
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

    # Create a selection map and mark the easiest 13,000 blocks
    selection_map = np.zeros((h // block_size, w // block_size), dtype=np.uint8)
    for idx, (i, j, block, std_dev) in enumerate(stds[:13000]):
        selection_map[i // block_size, j // block_size] = 1  # Mark block as easy (1)

    return selection_map, stds[:13000]  # Also return the selected blocks for plotting

# Function to select 13,000 luma blocks using the selection map
def select_luma_blocks(blocks, target_frame, target_index, selection_map, std_blocks, num_blocks=13000):
    easy_blocks = [(i, j, block) for (i, j, block, std_dev) in std_blocks]
    selected_blocks = random.sample(easy_blocks, num_blocks)

    # Plot selected luma blocks
    fig, ax = plt.subplots(1)
    ax.imshow(target_frame, cmap='gray')

    # Highlight the selected blocks
    for (i, j, block) in selected_blocks:
        rect = patches.Rectangle((j, i), block.shape[1], block.shape[0], linewidth=0.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(f'./processed_output/selected_blocks/{target_index}.png')

    return selected_blocks

# Interpolation filter coefficients (from the table)
interpolation_filter = np.array([
    [0, 0, 0, 64, 0, 0, 0, 0],
    [0, 1, -3, 63, 4, -2, 1, 0],
    [-1, 2, -5, 62, 8, -3, 1, 0],
    [-1, 3, -8, 60, 13, -4, 1, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 52, 26, -8, 3, -1],
    [-1, 3, -9, 47, 31, -10, 4, -1],
    [-1, 4, -11, 45, 34, -10, 4, -1],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [-1, 4, -10, 34, 45, -11, 4, -1],
    [-1, 4, -10, 31, 47, -9, 3, -1],
    [-1, 3, -8, 26, 52, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
    [0, 1, -4, 13, 60, -8, 3, -1],
    [0, 1, -3, 8, 62, -5, 2, -1],
    [0, 1, -2, 4, 63, -3, 1, 0]
])

def apply_interpolation_filter(block, filter_index):
    h, w = block.shape
    filtered_block = np.zeros((h, w), dtype=np.float32)
    filter_coeffs = interpolation_filter[filter_index]

    for y in range(h):
        for x in range(w):
            sum_value = 0.0
            for k in range(-3, 5):
                pos = x + k
                if 0 <= pos < w:
                    sum_value += block[y, pos] * filter_coeffs[k + 3]
            filtered_block[y, x] = sum_value / 64.0

    return np.clip(filtered_block, 0, 255).astype(np.uint8)

def derive_affine_motion_model(ref_frame, tgt_frame):    
    # Enhance the frames to improve feature detection
    ref_frame = enhance_image(ref_frame)
    tgt_frame = enhance_image(tgt_frame)
    
    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors with AKAZE in both frames
    kp1, des1 = akaze.detectAndCompute(ref_frame, None)
    kp2, des2 = akaze.detectAndCompute(tgt_frame, None)

    # print(f"Detected {len(kp1)} keypoints in reference frame.")
    # print(f"Detected {len(kp2)} keypoints in target frame.")

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        # No descriptors found or not enough keypoints
        print("No descriptors found or not enough keypoints in one of the frames.")
        return None

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # print(f"Found {len(matches)} matches.")

    if len(matches) < 3:
        # If there are not enough matches, return None
        print("Not enough matches found.")
        return None

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Estimate the affine transformation using RANSAC
    model_robust, inliers = ransac((src_pts, dst_pts),
                                   AffineTransform,
                                   min_samples=3,
                                   residual_threshold=2,
                                   max_trials=100)
    # Assuming model_robust is an AffineTransform object from skimage
    if model_robust is not None:
        return model_robust.params[:2, :].astype(np.float32)  # Ensure it is (2, 3) and float32
    else:
        return None

def apply_affine_transform(block, affine_transform):
    # Ensure the affine_transform is a numpy array of shape (2, 3) and type float32
    matrix = np.array(affine_transform, dtype=np.float32)
    assert matrix.shape == (2, 3), f"Affine transform matrix has invalid shape: {matrix.shape}"

    # Apply the affine transformation using OpenCV's warpAffine function
    h, w = block.shape
    transformed_block = cv2.warpAffine(block, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return transformed_block

# apply the affine motion model to the entire frame
# def apply_affine_transform(frame, affine_transform):
#     # Convert AffineTransform to a format suitable for cv2.warpAffine
#     matrix = affine_transform.params[:2, :]  # Only take the first 2 rows for 2D transformation
    
#     # Apply the affine transformation using OpenCV's warpAffine function
#     h, w = frame.shape
#     transformed_frame = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
#     return transformed_frame

def derive_perspective_motion_model(ref_frame, tgt_frame):    
    # Enhance the frames to improve feature detection
    ref_frame = enhance_image(ref_frame)
    tgt_frame = enhance_image(tgt_frame)
    
    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors with AKAZE in both frames
    kp1, des1 = akaze.detectAndCompute(ref_frame, None)
    kp2, des2 = akaze.detectAndCompute(tgt_frame, None)

    # print(f"Detected {len(kp1)} keypoints in reference frame.")
    # print(f"Detected {len(kp2)} keypoints in target frame.")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # No descriptors found or not enough keypoints
        print("No descriptors found or not enough keypoints in one of the frames.")
        return None

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # print(f"Found {len(matches)} matches.")

    if len(matches) < 4:
        # If there are not enough matches, return None
        print("Not enough matches found.")
        return None

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Estimate the perspective transformation using RANSAC
    perspective_matrix, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Assuming perspective_matrix is the result of cv2.findHomography
    if perspective_matrix is not None:
        return perspective_matrix.astype(np.float32)  # Ensure it is float32
    else:
        return None

def apply_perspective_transform(block, perspective_matrix):
    # Apply the perspective transformation using OpenCV's warpPerspective function
    h, w = block.shape
    transformed_block = cv2.warpPerspective(block, perspective_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return transformed_block

def derive_projection_motion_model(ref_frame, tgt_frame):
    # Enhance the frames to improve feature detection
    ref_frame = enhance_image(ref_frame)
    tgt_frame = enhance_image(tgt_frame)
    
    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors with AKAZE in both frames
    kp1, des1 = akaze.detectAndCompute(ref_frame, None)
    kp2, des2 = akaze.detectAndCompute(tgt_frame, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # No descriptors found or not enough keypoints
        print("No descriptors found or not enough keypoints in one of the frames.")
        return None

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        # If there are not enough matches, return None
        print("Not enough matches found.")
        return None

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Estimate the homography (projection model) using RANSAC
    projection_matrix, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if projection_matrix is not None:
        return projection_matrix.astype(np.float32)  # Ensure it is float32
    else:
        print("RANSAC failed to estimate the projection transformation.")
        return None

def apply_projection_transform(block, projection_matrix):
    # Apply the projection transformation using OpenCV's warpPerspective function
    h, w = block.shape
    transformed_block = cv2.warpPerspective(block, projection_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return transformed_block

def enhance_image(image):
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    # Apply a slight Gaussian blur to reduce noise
    enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    # Apply a sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    return enhanced_image

def calculate_edge_strength(block):
    # Calculate gradient magnitudes using Sobel operator
    grad_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(grad_mag)

def adaptive_deblocking_filter(image, block_size=BLOCK_SIZE):
    h, w = image.shape
    filtered_image = np.copy(image)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            edge_strength = calculate_edge_strength(block)
            
            # Adjust filtering strength based on edge strength
            if edge_strength > 50:
                filter_strength = 0.5  # Weak filtering for strong edges
            else:
                filter_strength = 1.0  # Strong filtering for weak edges
            
            # Apply horizontal and vertical deblocking filter
            for k in range(block_size):
                # Ensure the indices are within bounds
                if i + k < h and j + block_size <= w:
                    filtered_image[i + k, j:j + block_size] = cv2.GaussianBlur(
                        image[i + k, j:j + block_size], (5, 5), filter_strength).reshape(-1)
                if i + block_size <= h and j + k < w:
                    filtered_image[i:i + block_size, j + k] = cv2.GaussianBlur(
                        image[i:i + block_size, j + k], (5, 5), filter_strength).reshape(-1)
    
    return filtered_image

# # Post-processing: De-blocking filter
# def deblocking_filter(image):
#     kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
#     return cv2.filter2D(image, -1, kernel)

# Post-processing: Illuminance compensation
def illuminance_compensation(ref_block, tgt_block):
    ref_mean = np.mean(ref_block)
    tgt_mean = np.mean(tgt_block)
    compensation_factor = tgt_mean / ref_mean if ref_mean != 0 else 1
    return np.clip(ref_block * compensation_factor, 0, 255).astype(np.uint8)

def derive_motion_models(ref_frame_0, ref_frame_1, tgt_frame):
    models = []

    # Compute affine models
    affine_model_0 = derive_affine_motion_model(ref_frame_0, tgt_frame)
    affine_model_1 = derive_affine_motion_model(ref_frame_1, tgt_frame)
    models.append(('affine', affine_model_0, affine_model_1))

    # Compute perspective models
    perspective_model_0 = derive_perspective_motion_model(ref_frame_0, tgt_frame)
    perspective_model_1 = derive_perspective_motion_model(ref_frame_1, tgt_frame)
    models.append(('perspective', perspective_model_0, perspective_model_1))

    # Compute projection models
    projection_model_0 = derive_projection_motion_model(ref_frame_0, tgt_frame)
    projection_model_1 = derive_projection_motion_model(ref_frame_1, tgt_frame)
    models.append(('projection', projection_model_0, projection_model_1))

    return models

def select_best_models(models, target_frame, ref_frame_0, ref_frame_1, num_models=12):
    evaluated_models = []

    for model_type, model_0, model_1 in models:
        if model_0 is not None and model_1 is not None:
            # Apply models and blend the results
            blended_frame = blend_models(target_frame, ref_frame_0, ref_frame_1, model_0, model_1)
            # Evaluate the model (e.g., compute residual error or PSNR)
            error = compute_residual_error(target_frame, blended_frame)
            evaluated_models.append((model_type, model_0, model_1, error))

    # Sort models by error and select the best 12 models
    evaluated_models.sort(key=lambda x: x[3])
    best_models = evaluated_models[:num_models]

    return best_models

def blend_models(target_frame, ref_frame_0, ref_frame_1, model_0, model_1, weight=0.5):

    # classify the model matrices based on their shape
    if isinstance(model_0, (AffineTransform, np.ndarray)) and model_0.shape == (2, 3):
        compensated_frame_0 = apply_affine_transform(ref_frame_0, model_0)
    elif isinstance(model_0, np.ndarray) and model_0.shape == (3, 3):
        compensated_frame_0 = apply_projection_transform(ref_frame_0, model_0)
    else:
        compensated_frame_0 = apply_perspective_transform(ref_frame_0, model_0)

    if isinstance(model_1, (AffineTransform, np.ndarray)) and model_1.shape == (2, 3):
        compensated_frame_1 = apply_affine_transform(ref_frame_1, model_1)
    elif isinstance(model_1, np.ndarray) and model_1.shape == (3, 3):
        compensated_frame_1 = apply_projection_transform(ref_frame_1, model_1)
    else:
        compensated_frame_1 = apply_perspective_transform(ref_frame_1, model_1)

    # Blend the compensated frames
    blended_frame = (compensated_frame_0 * weight + compensated_frame_1 * (1 - weight)).astype(np.uint8)

    return blended_frame

def compute_residual_error(block, compensated_block):
    # MSE
    error = np.mean((block.astype(np.float32) - compensated_block.astype(np.float32)) ** 2)
    return error

def assign_models_to_blocks(blocks, best_models, target_frame, ref_frame_0, ref_frame_1):
    assigned_models = []

    for (block_pos, block) in blocks:
        best_model = None
        best_error = float('inf')

        for model_type, model_0, model_1, error in best_models:
            ref_block_0 = ref_frame_0[block_pos[0]:block_pos[0]+block.shape[0], block_pos[1]:block_pos[1]+block.shape[1]]
            ref_block_1 = ref_frame_1[block_pos[0]:block_pos[0]+block.shape[0], block_pos[1]:block_pos[1]+block.shape[1]]

            blended_block = blend_models(block, ref_block_0, ref_block_1, model_0, model_1)
            block_error = compute_residual_error(block, blended_block)

            if block_error < best_error:
                best_error = block_error
                best_model = (model_type, model_0, model_1)

        assigned_models.append((block_pos, best_model))

    return assigned_models

def apply_obmc(compensated_frame, block_size=BLOCK_SIZE):
    h, w = compensated_frame.shape
    obmc_frame = np.zeros_like(compensated_frame, dtype=np.float32)
    weights = np.zeros_like(compensated_frame, dtype=np.float32)

    # Define the weight mask for blending
    weight_mask = np.ones((block_size, block_size), dtype=np.float32)
    weight_mask[1:-1, 1:-1] = 4
    weight_mask = weight_mask / weight_mask.sum()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = compensated_frame[i:i+block_size, j:j+block_size].astype(np.float32)
            obmc_frame[i:i+block_size, j:j+block_size] += block * weight_mask
            weights[i:i+block_size, j:j+block_size] += weight_mask

    # Normalize the OBMC frame
    obmc_frame /= weights
    obmc_frame = np.clip(obmc_frame, 0, 255).astype(np.uint8)

    return obmc_frame


def apply_gmc(target_frame, reference_frames, target_index):
    h, w = target_frame.shape
    compensated_frame = np.zeros_like(target_frame)

    # Derive motion models
    models = derive_motion_models(reference_frames[0], reference_frames[1], target_frame)
    best_models = select_best_models(models, target_frame, reference_frames[0], reference_frames[1])

    # Process each block in the frame
    blocks = divide_into_blocks(target_frame, block_size=BLOCK_SIZE)
    selection_map, std_blocks = generate_selection_map(target_frame, block_size=BLOCK_SIZE)

    assigned_models = assign_models_to_blocks(blocks, best_models, target_frame, reference_frames[0], reference_frames[1])

    for (block_pos, (model_type, model_0, model_1)) in tqdm(assigned_models, desc="Processing Blocks", unit="block"):
        i, j = block_pos
        block = target_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        ref_block_0 = reference_frames[0][i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        ref_block_1 = reference_frames[1][i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

        # Apply illuminance compensation
        ref_block_0 = illuminance_compensation(ref_block_0, block)
        ref_block_1 = illuminance_compensation(ref_block_1, block)

        # Apply the selected model to the block
        compensated_block = blend_models(block, ref_block_0, ref_block_1, model_0, model_1)
        # Update the compensated frame
        compensated_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = compensated_block

    # POST-PROCESSING below
    # Apply Overlapped Block Motion Compensation (OBMC)
    compensated_frame = apply_obmc(compensated_frame, block_size=BLOCK_SIZE)
    # Apply de-blocking filter to the entire compensated frame
    compensated_frame = adaptive_deblocking_filter(compensated_frame)

    return compensated_frame


# def apply_gmc(target_frame, reference_frames, target_index):
#     h, w = target_frame.shape
#     compensated_frame = np.zeros_like(target_frame, dtype=np.float32)

#     # Derive global motion models using the entire frame
#     affine_matrix_0 = derive_affine_motion_model(reference_frames[0], target_frame)
#     affine_matrix_1 = derive_affine_motion_model(reference_frames[1], target_frame)

#     # Apply the interpolation filter (post-processing)
#     reference_frames[0] = apply_interpolation_filter(reference_frames[0], filter_index=2)  # Example filter index
#     reference_frames[1] = apply_interpolation_filter(reference_frames[1], filter_index=2)  # Example filter index

#     # Apply illuminance compensation (post-processing)
#     reference_frames[0] = illuminance_compensation(reference_frames[0], target_frame)
#     reference_frames[1] = illuminance_compensation(reference_frames[1], target_frame)

#     if affine_matrix_0 is None or affine_matrix_1 is None:
#         raise ValueError("Failed to derive motion models for the entire frame.")

#     # Apply affine transform to the entire reference frames
#     compensated_frame_0 = apply_affine_transform(reference_frames[0], affine_matrix_0)
#     compensated_frame_1 = apply_affine_transform(reference_frames[1], affine_matrix_1)

#     # Blend the compensated frames
#     compensated_frame = (compensated_frame_0.astype(np.float32) + compensated_frame_1.astype(np.float32)) / 2

#     # Apply de-blocking filter to the entire compensated frame
#     compensated_frame = deblocking_filter(compensated_frame.astype(np.uint8))

#     # Apply Joint Bilateral Filter (JBF)
#     compensated_frame = cv2.ximgproc.jointBilateralFilter(
#         joint=compensated_frame,
#         src=target_frame,
#         d=1,
#         sigmaColor=75,
#         sigmaSpace=75
#     )

#     return compensated_frame
