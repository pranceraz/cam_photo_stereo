import cv2 as cv
import numpy as np
from pathlib import Path
import logging
import affine
import homography
import os
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_16bit_image(image_path):
    """Load a 16-bit image preserving bit depth."""
    img = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    logger.info(f"Loaded image: {image_path}")
    logger.info(f"Shape: {img.shape}, dtype: {img.dtype}, Range: [{img.min()}, {img.max()}]")
    return img

def align_affine(test_path,ref_path):
    ref_path = Path(ref_path)
    test_path = Path(test_path)
    
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found at {ref_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test image not found at {test_path}")
    
    ref_img = load_16bit_image(ref_path)
    test_img = load_16bit_image(test_path)
    
    logger.info("Starting alignment...")
    # aligner = feature_extractor.ImageAligner(ref_img=ref_img, test_img=test_img)
    # aligned_img = aligner.realign()
    
    aligner = affine.ImageAligner(ref_img=ref_img, test_img=test_img)
    aligned_img = aligner.realign()
    
    logger.info("Alignment completed.")
    logger.info(f"Aligned image shape: {aligned_img.shape}, dtype: {aligned_img.dtype}, "
                f"Range: [{aligned_img.min():.3f}, {aligned_img.max():.3f}]")
    
    return aligned_img

def align_homography(test, ref_path):
    ref_path = Path(ref_path)
    
    if type(test) is str:
        test = load_16bit_image(test)
    
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found at {ref_path}")
    
    ref_img = load_16bit_image(ref_path)
    logger.info("Starting homography alignment...")
    
    aligner = homography.ImageAligner(ref_img=ref_img, test_img=test)
    aligned_img = aligner.realign() 
    logger.info("Homography alignment completed.")
    logger.info(f"Aligned image shape: {aligned_img.shape}, dtype: {aligned_img.dtype}, "
                f"Range: [{aligned_img.min():.3f}, {aligned_img.max():.3f}]")
    return aligned_img



if __name__ == "__main__":
    ref_path = "input_for_align/1rupee/1before1.png"
    test_path = "input_for_align/1rupee/1after1.png"
    
    aligned_image = align_affine(test_path,ref_path)
    aligned_final = align_homography(aligned_image,ref_path)
    os.makedirs("aligned_output", exist_ok = True)
    output_path = "aligned_output/aligned_coin2_16bit_aff+hmg.png"
    
   # Save the aligned image
    cv.imwrite("aligned_output/aligned_coin2_16bit_aff+hmg.png", aligned_final)
    logger.info(f"Saved aligned image")
    
    