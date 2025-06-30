import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('../samples/10rupee/002.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess: smooth and threshold
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert if background is light and object is dark
if np.mean(thresh) > 127:
    thresh = cv2.bitwise_not(thresh)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create empty mask
mask = np.zeros_like(gray)

# Draw largest contour only
if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

# Optionally clean mask edges
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply mask to image
masked_img = cv2.bitwise_and(img, img, mask=mask)

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Object Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.title('Masked Object')
plt.axis('off')

plt.tight_layout()
plt.show()
