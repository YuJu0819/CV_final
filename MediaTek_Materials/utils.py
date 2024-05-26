import numpy as np
import cv2
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# Function to derive affine motion models using AKAZE
def derive_affine_motion_model(ref_block, tgt_block):
    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors with AKAZE in both blocks
    kp1, des1 = akaze.detectAndCompute(ref_block, None)
    kp2, des2 = akaze.detectAndCompute(tgt_block, None)

    if des1 is None or des2 is None:
        print("No descriptors found")
        return np.eye(2, 3, dtype=np.float32)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Number of matches (affine): {len(matches)}")
    
    if len(matches) >= 3:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute the affine transformation using RANSAC
        affine_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
    else:
        # If there are not enough matches, use an identity matrix
        affine_matrix = np.eye(2, 3, dtype=np.float32)

    return affine_matrix

# Function to apply affine transformation
def apply_affine_transform(block, affine_matrix, interpolation=cv2.INTER_LINEAR):
    h, w = block.shape
    transformed_block = cv2.warpAffine(block, affine_matrix, (w, h), flags=interpolation)
    return transformed_block

# Function to derive perspective motion models using AKAZE
def derive_perspective_motion_model(ref_block, tgt_block):
    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors with AKAZE in both blocks
    kp1, des1 = akaze.detectAndCompute(ref_block, None)
    kp2, des2 = akaze.detectAndCompute(tgt_block, None)

    if des1 is None or des2 is None:
        print("No descriptors found")
        return np.eye(3, 3, dtype=np.float32)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Number of matches (perspective): {len(matches)}")
    
    if len(matches) >= 4:  # Minimum number of points for perspective transform is 4
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute the perspective transformation using RANSAC
        perspective_matrix, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    else:
        # If there are not enough matches, use an identity matrix
        perspective_matrix = np.eye(3, 3, dtype=np.float32)

    return perspective_matrix

# Function to apply perspective transformation
def apply_perspective_transform(block, perspective_matrix, interpolation=cv2.INTER_LINEAR):
    h, w = block.shape
    transformed_block = cv2.warpPerspective(block, perspective_matrix, (w, h), flags=interpolation)
    return transformed_block

# Function to cluster motion vectors
def cluster_motion_vectors(motion_vectors, num_clusters=12):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(motion_vectors)
    return kmeans.cluster_centers_

# Function to apply global motion compensation with boundary handling
def apply_gmc(target_frame, reference_frames, target_index):
    print(target_frame.shape)
    h, w = target_frame.shape
    compensated_frame = np.zeros_like(target_frame)

    blocks = divide_into_blocks(target_frame)
    selection_map, std_blocks = generate_selection_map(target_frame)
    selected_blocks = select_luma_blocks(blocks, target_frame, target_index, selection_map, std_blocks)

    # List to store motion models
    motion_models = []

    # Derive motion models for selected blocks
    for (i, j, block) in selected_blocks:
        ref_block_0 = reference_frames[0][i:min(i+block.shape[0], h), j:min(j+block.shape[1], w)]
        ref_block_1 = reference_frames[1][i:min(i+block.shape[0], h), j:min(j+block.shape[1], w)]
        if ref_block_0.size == 0 or ref_block_1.size == 0:
            print(f"Empty reference block at ({i}, {j})")
            continue

        affine_matrix_0 = derive_affine_motion_model(ref_block_0, block)
        perspective_matrix_0 = derive_perspective_motion_model(ref_block_0, block)
        affine_matrix_1 = derive_affine_motion_model(ref_block_1, block)
        perspective_matrix_1 = derive_perspective_motion_model(ref_block_1, block)
        
        blending_weight = random.uniform(0, 1)  # Random blending weight for demonstration
        motion_models.append((affine_matrix_0, perspective_matrix_0, affine_matrix_1, perspective_matrix_1, blending_weight))

    # Cluster motion models to derive at most 12 motion models
    # Flatten motion models for clustering
    flattened_motion_models = [np.hstack((model[0].flatten(), model[1].flatten(), model[2].flatten(), model[3].flatten(), model[4])) for model in motion_models if model[0] is not None and model[1] is not None and model[2] is not None and model[3] is not None]
    # Cluster into 12 motion models
    motion_models_clusters = cluster_motion_vectors(flattened_motion_models)

    # Apply motion models to compensate the frame
    for (i, j, block) in selected_blocks:
        ref_block_0 = reference_frames[0][i:min(i+block.shape[0], h), j:min(j+block.shape[1], w)]
        ref_block_1 = reference_frames[1][i:min(i+block.shape[0], h), j:min(j+block.shape[1], w)]
        if ref_block_0.size == 0 or ref_block_1.size == 0:
            print(f"Empty reference block at ({i}, {j})")
            continue

        affine_matrix_0 = derive_affine_motion_model(ref_block_0, block)
        perspective_matrix_0 = derive_perspective_motion_model(ref_block_0, block)
        affine_matrix_1 = derive_affine_motion_model(ref_block_1, block)
        perspective_matrix_1 = derive_perspective_motion_model(ref_block_1, block)

        # Find the closest motion model cluster
        current_model = np.hstack((affine_matrix_0.flatten(), perspective_matrix_0.flatten(), affine_matrix_1.flatten(), perspective_matrix_1.flatten(), 0.5))  # Use 0.5 as a placeholder blending weight
        distances = np.linalg.norm(motion_models_clusters - current_model, axis=1)
        closest_model_index = np.argmin(distances)
        closest_motion_model = motion_models_clusters[closest_model_index]

        # Extract the closest model's components
        closest_affine_matrix_0 = closest_motion_model[:6].reshape(2, 3)
        closest_perspective_matrix_0 = closest_motion_model[6:15].reshape(3, 3)
        closest_affine_matrix_1 = closest_motion_model[15:21].reshape(2, 3)
        closest_perspective_matrix_1 = closest_motion_model[21:30].reshape(3, 3)
        blending_weight = closest_motion_model[30]

        # Apply the closest motion model
        compensated_block_affine_0 = apply_affine_transform(block, closest_affine_matrix_0)
        compensated_block_perspective_0 = apply_perspective_transform(block, closest_perspective_matrix_0)
        compensated_block_affine_1 = apply_affine_transform(block, closest_affine_matrix_1)
        compensated_block_perspective_1 = apply_perspective_transform(block, closest_perspective_matrix_1)

        # Blend the compensated blocks
        blended_block_0 = (1 - blending_weight) * compensated_block_affine_0 + blending_weight * compensated_block_perspective_0
        blended_block_1 = (1 - blending_weight) * compensated_block_affine_1 + blending_weight * compensated_block_perspective_1
        final_blended_block = (blended_block_0 + blended_block_1) / 2

        # Debugging: Check if the block has changed
        if np.array_equal(block, final_blended_block):
            print(f"Block at ({i}, {j}) has not changed after compensation.")

        # Update the compensated frame
        compensated_frame[i:i+block.shape[0], j:j+block.shape[1]] = final_blended_block

    return compensated_frame

# # Example usage
# filepath = 'path_to_your_yuv_file'
# width, height = 1920, 1080  # Example resolution
# num_frames = 129

# frames = load_yuv_frames(filepath, width, height, num_frames)
# target_frame = frames[8]
# reference_frames = [frames[0], frames[16]]

# # Apply GMC using the selection map
# compensated_frame = apply_gmc(target_frame, reference_frames, target_index=8)
# cv2.imwrite('compensated_frame_8.png', compensated_frame)
