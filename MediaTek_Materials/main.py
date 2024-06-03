from utils import apply_gmc
import cv2
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as transforms
from PIL import Image

def load_unet_model():
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
    model.eval()
    return model

def load_prn_model():
    model = PixelRestorationNetwork()
    # model.load_state_dict(torch.load('path_to_pretrained_model.pth'))
    model.eval()
    return model

def initialize_background_subtractor():
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    return backSub

def main():
    # Example processing order
    processing_order = [
        # Group 1 (Pictures 0-31)
        (16, 0, 32),
        (8, 0, 16),
        (4, 0, 8),
        (2, 0, 4),
        (1, 0, 2),
        (3, 2, 4),
        (6, 4, 8),
        (5, 4, 6),
        (7, 6, 8),
        (12, 8, 16),
        (10, 8, 12),
        (9, 8, 10),
        (11, 10, 12),
        (14, 12, 16),
        (13, 12, 14),
        (15, 14, 16),
        (24, 16, 32),
        (20, 16, 24),
        (18, 16, 20),
        (17, 16, 18),
        (19, 18, 20),
        (22, 20, 24),
        (21, 20, 22),
        (23, 22, 24),
        (28, 24, 32),
        (26, 24, 28),
        (25, 24, 26),
        (27, 26, 28),
        (30, 28, 32),
        (29, 28, 30),
        (31, 30, 32),

        # Group 2 (Pictures 33-63)
        (48, 32, 64),
        (40, 32, 48),
        (36, 32, 40),
        (34, 32, 36),
        (33, 32, 34),
        (35, 34, 36),
        (38, 36, 40),
        (37, 36, 38),
        (39, 38, 40),
        (44, 40, 48),
        (42, 40, 44),
        (41, 40, 42),
        (43, 42, 44),
        (46, 44, 48),
        (45, 44, 46),
        (47, 46, 48),
        (56, 48, 64),
        (52, 48, 56),
        (50, 48, 52),
        (49, 48, 50),
        (51, 50, 52),
        (54, 52, 56),
        (53, 52, 54),
        (55, 54, 56),
        (60, 56, 64),
        (58, 56, 60),
        (57, 56, 58),
        (59, 58, 60),
        (62, 60, 64),
        (61, 60, 62),
        (63, 62, 64),

        # Group 3 (Pictures 65-95)
        (80, 64, 96),
        (72, 64, 80),
        (68, 64, 72),
        (66, 64, 68),
        (65, 64, 66),
        (67, 66, 68),
        (70, 68, 72),
        (69, 68, 70),
        (71, 70, 72),
        (76, 72, 80),
        (74, 72, 76),
        (73, 72, 74),
        (75, 74, 76),
        (78, 76, 80),
        (77, 76, 78),
        (79, 78, 80),
        (88, 80, 96),
        (84, 80, 88),
        (82, 80, 84),
        (81, 80, 82),
        (83, 82, 84),
        (86, 84, 88),
        (85, 84, 86),
        (87, 86, 88),
        (92, 88, 96),
        (90, 88, 92),
        (89, 88, 90),
        (91, 90, 92),
        (94, 92, 96),
        (93, 92, 94),
        (95, 94, 96),

        # Group 4 (Pictures 97-127)
        (112, 96, 128),
        (104, 96, 112),
        (100, 96, 104),
        (98, 96, 100),
        (97, 96, 98),
        (99, 98, 100),
        (102, 100, 104),
        (101, 100, 102),
        (103, 102, 104),
        (108, 104, 112),
        (106, 104, 108),
        (105, 104, 106),
        (107, 106, 108),
        (110, 108, 112),
        (109, 108, 110),
        (111, 110, 112),
        (120, 112, 128),
        (116, 112, 120),
        (114, 112, 116),
        (113, 112, 114),
        (115, 114, 116),
        (118, 116, 120),
        (117, 116, 118),
        (119, 118, 120),
        (124, 120, 128),
        (122, 120, 124),
        (121, 120, 122),
        (123, 122, 124),
        (126, 124, 128),
        (125, 124, 126),
        (127, 126, 128),
    ]

    # # Load YUV frames (example for 129 frames, adjust path and parameters as needed)
    # yuv_filepath = 'path_to_yuv_file.yuv'
    # frames = load_yuv_frames(yuv_filepath, 3840, 2160, 129)

    # Load YUV frames
    frames = []
    folder_path = './YUV_frames/'
    # print('Loading luma frames...')
    for i in tqdm(range(129), desc="Loading frames", unit="frames"):
        idx = str(i).zfill(3)
        img_pth = folder_path + idx + '.png'

        # print(f'{folder_path = }')
        img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
        else:
            print(f'Warning: {img_pth} could not be loaded.')

    print('-')

    # Process frames based on hierarchical-B order
    for target_index, ref_index_0, ref_index_1 in tqdm(processing_order, desc="Processing frames", unit="frames"):
        # print(target_index, ref_index_0, ref_index_1)
        target_frame = frames[target_index]
        reference_frames = [frames[ref_index_0], frames[ref_index_1]]
        compensated_frame = apply_gmc(target_frame, reference_frames, target_index)
        # compensated_frame = apply_gmc_with_segmentation(target_frame, reference_frames, target_index, model=backSub)
        # interpolated_frame = interpolate_black_regions(compensated_frame)
        # Save or further process the compensated frame
        cv2.imwrite(f'./processed_output/compensated_frame/{target_index}.png', compensated_frame)
        # cv2.imwrite(f'./processed_output/compensated_frame/{target_index}.png', interpolated_frame)

    print("Processing complete.")


if __name__ == '__main__':
    # Load the U-Net model
    # unet_model = load_unet_model()
    # backSub = initialize_background_subtractor()

    # Load the PRN model
    # prn_model = load_prn_model()

    main()