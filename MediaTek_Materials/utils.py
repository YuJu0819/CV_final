import numpy as np
import cv2
from skimage.transform import AffineTransform
from skimage.measure import ransac
from tqdm import tqdm

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


def assign_models_to_blocks(blocks, models, target_frame, ref_frame_0, ref_frame_1):
    assigned_models = []

    for (block_pos, block) in blocks:
        # best_model = None
        # best_error = float('inf')

        # for model_type, model_0, model_1, error in best_models:
        #     ref_block_0 = ref_frame_0[block_pos[0]:block_pos[0]+block.shape[0], block_pos[1]:block_pos[1]+block.shape[1]]
        #     ref_block_1 = ref_frame_1[block_pos[0]:block_pos[0]+block.shape[0], block_pos[1]:block_pos[1]+block.shape[1]]

        #     blended_block = blend_models(block, ref_block_0, ref_block_1, model_0, model_1)
        #     block_error = compute_residual_error(block, blended_block)

        #     if block_error < best_error:
        #         best_error = block_error
        #         best_model = (model_type, model_0, model_1)

        assigned_models.append((block_pos, models[0]))

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

def interpolation(frame, current_block, i, j, search_range=16, block_size=BLOCK_SIZE):
    h, w = frame.shape
    min_sad = np.inf
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_i = i + dy
            ref_j = j + dx
            
            if ref_i < 0 or ref_i + block_size > h or ref_j < 0 or ref_j + block_size > w:
                continue
            
            reference_block = frame[ref_i:ref_i + block_size, ref_j:ref_j + block_size]
            sad = np.sum(np.abs(current_block - reference_block))
            
            if sad < min_sad:
                min_sad = sad
                best_vector = (dy, dx)
    new_i = i + best_vector[0]
    new_j = j + best_vector[1]
    return frame[new_i: new_i + block_size, new_j:new_j + block_size]
def apply_gmc(target_frame, reference_frames, target_index, modelmap):
    h, w = target_frame.shape
    compensated_frame = np.zeros_like(target_frame)

    # Derive motion models
    models = derive_motion_models(reference_frames[0], reference_frames[1], target_frame)
    # best_models = select_best_models(models, target_frame, reference_frames[0], reference_frames[1])

    # Process each block in the frame
    blocks = divide_into_blocks(target_frame, block_size=BLOCK_SIZE)

    assigned_models = assign_models_to_blocks(blocks, models, target_frame, reference_frames[0], reference_frames[1])

    for (block_pos, (model_type, model_0, model_1)) in tqdm(assigned_models, desc="Processing Blocks", unit="block"):
        i, j = block_pos
        block = target_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        ref_block_0 = reference_frames[0][i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        ref_block_1 = reference_frames[1][i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

        if modelmap[i//16, j//16] == 0:
            # Apply illuminance compensation
            ref_block_0 = illuminance_compensation(ref_block_0, block)
            ref_block_1 = illuminance_compensation(ref_block_1, block)
            compensated_block = blend_models(block, ref_block_0, ref_block_1, model_0, model_1)
        else:
            ref_block_0 = interpolation(reference_frames[0], block, i, j)
            ref_block_1 = interpolation(reference_frames[1], block, i, j)
            compensated_block = ref_block_0//2 + ref_block_1//2
        # Apply the selected model to the block
        # Update the compensated frame

        compensated_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = compensated_block

    # POST-PROCESSING below
    # Apply Overlapped Block Motion Compensation (OBMC)
    # compensated_frame = apply_obmc(compensated_frame, block_size=BLOCK_SIZE)
    # # Apply de-blocking filter to the entire compensated frame
    # compensated_frame = adaptive_deblocking_filter(compensated_frame)

    return compensated_frame


