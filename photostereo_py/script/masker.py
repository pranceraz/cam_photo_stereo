import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load image
img_path = '../samples/10rupee/002.JPG'
image_bgr = cv2.imread(img_path)
original_h, original_w = image_bgr.shape[:2]

# Resize to fit GPU (optional, for SAM memory efficiency)
image_bgr_resized = cv2.resize(image_bgr, (1024, 1024), interpolation=cv2.INTER_AREA)
image_rgb = cv2.cvtColor(image_bgr_resized, cv2.COLOR_BGR2RGB)

# Load SAM model
checkpoint_path = "../../sam_vit_b.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
sam.eval()

# Force model to use float32
for param in sam.parameters():
    param.data = param.data.float()

# Configure automatic mask generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

# Generate mask
print("Generating mask...")
torch.cuda.empty_cache()
masks = mask_generator.generate(image_rgb)

# Pick the largest mask
masks = sorted(masks, key=lambda x: x['area'], reverse=True)
mask = masks[0]['segmentation'].astype(np.uint8) * 255

# Resize mask to original image size
mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
mask_resized = cv2.bitwise_not(mask_resized)
# Resize image back to original size
image_bgr_orig = cv2.resize(image_bgr_resized, (original_w, original_h), interpolation=cv2.INTER_AREA)

# Apply mask to original image
masked_img = cv2.bitwise_and(image_bgr_orig, image_bgr_orig, mask=mask_resized)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image_bgr_orig, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_resized, cmap='gray')
plt.title("AI-Generated Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.title("Masked Object")
plt.axis("off")

plt.tight_layout()
plt.show()
cv2.imwrite("mask.png", mask_resized)
