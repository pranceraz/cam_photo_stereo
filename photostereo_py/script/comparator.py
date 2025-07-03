import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from misc.feature_extract import FeatureExtraction, feature_matching

import warnings
warnings.filterwarnings('ignore')

try:
    import OpenEXR
    import Imath
    EXR_AVAILABLE = True
except ImportError:
    print("Warning: OpenEXR not available. Install with: pip install OpenEXR")
    EXR_AVAILABLE = False

class NormalMapComparator:
    
    def __init__(self):
        self.ref = cv.imread('ref.png', cv2.IMREAD_GRAYSCALE)
        self.test = cv.imread('test.png', cv2.IMREAD_GRAYSCALE)
        
    def realign():
        img0 = cv.imread("lasmeninas0.jpg", cv.COLOR_GR2RGBA)
        img1 = cv.imread("lasmeninas1.jpg", cv.COLOR_BGR2RGBA)
        features0 = FeatureExtraction(img0)
        features1 = FeatureExtraction(img1)
        matches = feature_matching(features0, features1)
        # matched_image = cv.drawMatches(img0, features0.kps, \
        #     img1, features1.kps, matches, None, flags=2)

        H, _ = cv.findHomography( features0.matched_pts, \
            features1.matched_pts, cv.RANSAC, 5.0)

        h, w, c = img1.shape
        warped = cv.warpPerspective(img0, H, (w, h), \
            borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        output = np.zeros((h, w, 3), np.uint8)
        alpha = warped[:, :, 3] / 255.0
        output[:, :, 0] = (1. - alpha) * img1[:, :, 0] + alpha * warped[:, :, 0]
        output[:, :, 1] = (1. - alpha) * img1[:, :, 1] + alpha * warped[:, :, 1]
        output[:, :, 2] = (1. - alpha) * img1[:, :, 2] + alpha * warped[:, :, 2]
        
    def cross_correlate_score():
        
        pass
    
    def image_root_mse(self,normalref, normaltest ):
        from skimage.metrics import structural_similarity as ssim

        if normalref or normaltest == None:
            normalref,normaltest = self.ref,self.test
            
        score, diff_map = ssim(normalref, normaltest, full=True)

        print(f"SSIM Score: {score}")
        plt.imshow(diff_map, cmap='gray')
        plt.title('SSIM Difference Map')
        plt.colorbar()
        plt.show()

    
    from skimage.util import view_as_windows

    def sliding_ncc(ref, test, patch_size=31):
        pad = patch_size // 2
        ref_p = np.pad(ref, pad)
        test_p = np.pad(test, pad)
        
        H, W = ref.shape
        ncc_map = np.zeros((H, W))

        for y in range(H):
            for x in range(W):
                ref_patch = ref_p[y:y+patch_size, x:x+patch_size].astype(np.float32)
                test_patch = test_p[y:y+patch_size, x:x+patch_size].astype(np.float32)
                
                ref_mean = np.mean(ref_patch)
                test_mean = np.mean(test_patch)
                
                num = np.sum((ref_patch - ref_mean) * (test_patch - test_mean))
                denom = np.sqrt(np.sum((ref_patch - ref_mean) ** 2) * np.sum((test_patch - test_mean) ** 2) + 1e-6)
                
                ncc_map[y, x] = num / denom
                
        return ncc_map

        