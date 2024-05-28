import numpy as np
import cv2
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import AffineTransform, warp
from skimage import img_as_float
from skimage.measure import ransac

BLOCK_SIZE = 16

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
def divide_into_blocks(image, block_size=BLOCK_SIZE):
    blocks = []
    h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            blocks.append((i, j, block))
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

def derive_affine_motion_model(ref_frame, tgt_frame):
    # Convert to grayscale if not already
    if len(ref_frame.shape) == 3:
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    if len(tgt_frame.shape) == 3:
        tgt_frame = cv2.cvtColor(tgt_frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance the frames to improve feature detection
    ref_frame = enhance_image(ref_frame)
    tgt_frame = enhance_image(tgt_frame)
    
    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Find keypoints and descriptors with AKAZE in both frames
    kp1, des1 = akaze.detectAndCompute(ref_frame, None)
    kp2, des2 = akaze.detectAndCompute(tgt_frame, None)

    print(f"Detected {len(kp1)} keypoints in reference frame.")
    print(f"Detected {len(kp2)} keypoints in target frame.")

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        # No descriptors found or not enough keypoints
        print("No descriptors found or not enough keypoints in one of the frames.")
        return None

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Found {len(matches)} matches.")

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
    if model_robust is None:
        # If RANSAC fails, return None
        print("RANSAC failed to estimate the affine transformation.")
        return None

    print("Affine transformation parameters:", model_robust.params)

    return model_robust

def apply_affine_transform(block, affine_transform):
    # Convert AffineTransform to a format suitable for cv2.warpAffine
    matrix = affine_transform.params[:2, :]  # Only take the first 2 rows for 2D transformation
    
    # Apply the affine transformation using OpenCV's warpAffine function
    h, w = block.shape
    transformed_block = cv2.warpAffine(block, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    print("Block min/max before transform:", block.min(), block.max())
    print("Transformed block min/max:", transformed_block.min(), transformed_block.max())

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

# Function to apply global motion compensation with boundary handling
def apply_gmc(target_frame, reference_frames, target_index):
    h, w = target_frame.shape
    compensated_frame = np.zeros_like(target_frame)

    # Derive a global motion model using the entire frame
    affine_matrix_0 = derive_affine_motion_model(reference_frames[0], target_frame)
    affine_matrix_1 = derive_affine_motion_model(reference_frames[1], target_frame)

    if affine_matrix_0 is None or affine_matrix_1 is None:
        raise ValueError("Failed to derive affine motion models for the entire frame.")

    # Process each block in the frame
    blocks = divide_into_blocks(target_frame, block_size=16)
    selection_map, std_blocks = generate_selection_map(target_frame, block_size=16)
    selected_blocks = select_luma_blocks(blocks, target_frame, target_index, selection_map, std_blocks, num_blocks=13000)

    for (i, j, block) in selected_blocks:
        ref_block_0 = reference_frames[0][i:i+block.shape[0], j:j+block.shape[1]]
        ref_block_1 = reference_frames[1][i:i+block.shape[0], j:j+block.shape[1]]

        # Check if the block is within the bounds of the reference frames
        if ref_block_0.shape != block.shape or ref_block_1.shape != block.shape:
            print(f"Skipping block at ({i}, {j}) due to size mismatch.")
            continue

        print(f"Block min/max before transform at ({i}, {j}):", block.min(), block.max())
        compensated_block_0 = apply_affine_transform(ref_block_0, affine_matrix_0)
        compensated_block_1 = apply_affine_transform(ref_block_1, affine_matrix_1)

        # Blend the compensated blocks
        final_blended_block = ((compensated_block_0.astype(np.float32) + compensated_block_1.astype(np.float32)) / 2).astype(np.uint8)

        print(f"Blended block at ({i}, {j}) min/max:", final_blended_block.min(), final_blended_block.max())

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
