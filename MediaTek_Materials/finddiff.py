import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('./YUV_frames/005.png')
image2 = cv2.imread('./processed_output/compensated_frame/5.png')

# Check if the images are loaded properly
# if image1 is None:
#     print("Error: Could not load image1")
#     exit()
# if image2 is None:
#     print("Error: Could not load image2")
#     exit()

# Resize images to be the same size if necessary
# This step is optional and depends on the use case
# if image1.shape != image2.shape:
#     image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Subtract the images
subtracted_image = cv2.subtract(image1, image2)

# Display the result
# cv2.imshow('Subtracted Image', subtracted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite(f'./processed_output/{5}_sub.png', subtracted_image)
# print(subtracted_image)
