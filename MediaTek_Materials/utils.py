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
    if model_robust is None:
        # If RANSAC fails, return None
        print("RANSAC failed to estimate the affine transformation.")
        return None

    # print("Affine transformation parameters:", model_robust.params)

    return model_robust

def apply_affine_transform(block, affine_transform):
    # Convert AffineTransform to a format suitable for cv2.warpAffine
    matrix = affine_transform.params[:2, :]  # Only take the first 2 rows for 2D transformation
    
    # Apply the affine transformation using OpenCV's warpAffine function
    h, w = block.shape
    transformed_block = cv2.warpAffine(block, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # print("Block min/max before transform:", block.min(), block.max())
    # print("Transformed block min/max:", transformed_block.min(), transformed_block.max())

    return transformed_block

def derive_perspective_motion_model(ref_frame, tgt_frame):
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

    if perspective_matrix is None:
        # If RANSAC fails, return None
        print("RANSAC failed to estimate the perspective transformation.")
        return None

    # print("Perspective transformation matrix:\n", perspective_matrix)

    return perspective_matrix

def apply_perspective_transform(block, perspective_matrix):
    # Apply the perspective transformation using OpenCV's warpPerspective function
    h, w = block.shape
    transformed_block = cv2.warpPerspective(block, perspective_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # print("Block min/max before transform:", block.min(), block.max())
    # print("Transformed block min/max:", transformed_block.min(), transformed_block.max())

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

# Post-processing: De-blocking filter
def deblocking_filter(image):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
    return cv2.filter2D(image, -1, kernel)

# Post-processing: Illuminance compensation
def illuminance_compensation(ref_block, tgt_block):
    ref_mean = np.mean(ref_block)
    tgt_mean = np.mean(tgt_block)
    compensation_factor = tgt_mean / ref_mean if ref_mean != 0 else 1
    return ref_block * compensation_factor

# Function to load the pre-trained U-Net model
def load_unet_model():
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
    model.eval()
    return model

# Improved segmentation function using U-Net
def segment_foreground_background(image, model):
    # Transform the image to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Convert grayscale to 3-channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        mask = model(image_tensor)[0][0]  # Get the predicted mask
    
    mask = (mask > 0.5).numpy().astype('uint8') * 255  # Convert to binary mask
    
    return cv2.resize(mask, (image.shape[1], image.shape[0]))

# Function to apply global motion compensation with boundary handling and segmentation
def apply_gmc_with_segmentation(target_frame, reference_frames, target_index, model):
    h, w = target_frame.shape
    compensated_frame = np.zeros_like(target_frame)

    # Derive global motion models using the entire frame
    affine_matrix_0 = derive_affine_motion_model(reference_frames[0], target_frame)
    perspective_matrix_0 = derive_perspective_motion_model(reference_frames[0], target_frame)
    affine_matrix_1 = derive_affine_motion_model(reference_frames[1], target_frame)
    perspective_matrix_1 = derive_perspective_motion_model(reference_frames[1], target_frame)

    if affine_matrix_0 is None or perspective_matrix_0 is None or affine_matrix_1 is None or perspective_matrix_1 is None:
        raise ValueError("Failed to derive motion models for the entire frame.")

    # Segment the target frame into foreground and background
    segmentation_mask = segment_foreground_background(target_frame, model)

    # Save the segmentation mask for debugging/visualization
    # segmentation_mask_filename = f'./processed_output/segmentation_masks/segmentation_mask_{target_index}.png'
    # cv2.imwrite(segmentation_mask_filename, segmentation_mask)

    # Process each block in the frame
    blocks = divide_into_blocks(target_frame, block_size=16)
    selection_map, std_blocks = generate_selection_map(target_frame, block_size=16)
    selected_blocks = blocks

    for (i, j, block) in tqdm(selected_blocks, desc="Processing Blocks", unit="block"):
        ref_block_0 = reference_frames[0][i:i+block.shape[0], j:j+block.shape[1]]
        ref_block_1 = reference_frames[1][i:i+block.shape[0], j:j+block.shape[1]]

        # Check if the block is within the bounds of the reference frames
        if ref_block_0.shape != block.shape or ref_block_1.shape != block.shape:
            print(f"Skipping block at ({i}, {j}) due to size mismatch.")
            continue

        # Determine if the block is foreground or background
        if np.mean(segmentation_mask[i:i+block.shape[0], j:j+block.shape[1]]) > 128:  # Foreground
            # Apply the interpolation filter (post-processing)
            ref_block_0 = apply_interpolation_filter(ref_block_0, filter_index=5)  # Example filter index
            ref_block_1 = apply_interpolation_filter(ref_block_1, filter_index=5)  # Example filter index

            # Apply illuminance compensation (post-processing)
            ref_block_0 = illuminance_compensation(ref_block_0, block)
            ref_block_1 = illuminance_compensation(ref_block_1, block)

            # Apply affine and perspective transforms
            compensated_block_affine_0 = apply_affine_transform(ref_block_0, affine_matrix_0)
            compensated_block_perspective_0 = apply_perspective_transform(ref_block_0, perspective_matrix_0)
            compensated_block_affine_1 = apply_affine_transform(ref_block_1, affine_matrix_1)
            compensated_block_perspective_1 = apply_perspective_transform(ref_block_1, perspective_matrix_1)

            # Blend the compensated blocks
            final_blended_block_0 = (compensated_block_affine_0.astype(np.float32) + compensated_block_perspective_0.astype(np.float32)) / 2
            final_blended_block_1 = (compensated_block_affine_1.astype(np.float32) + compensated_block_perspective_1.astype(np.float32)) / 2
            final_blended_block = ((final_blended_block_0 + final_blended_block_1) / 2).astype(np.uint8)
        else:  # Background
            # Use only the affine transformation for the background
            compensated_block_affine_0 = apply_affine_transform(ref_block_0, affine_matrix_0)
            compensated_block_affine_1 = apply_affine_transform(ref_block_1, affine_matrix_1)

            # Blend the compensated blocks
            final_blended_block = ((compensated_block_affine_0.astype(np.float32) + compensated_block_affine_1.astype(np.float32)) / 2).astype(np.uint8)

        # Update the compensated frame
        compensated_frame[i:i+block.shape[0], j:j+block.shape[1]] = final_blended_block

    # Apply de-blocking filter to the entire compensated frame
    compensated_frame = deblocking_filter(compensated_frame)

    return compensated_frame



