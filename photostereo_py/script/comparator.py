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
    
    def __init__(self, ref_path:str, test_path:str):
        self.ref  = cv.imread(ref_path, cv.COLOR_BGR2RGBA) #img0
        self.test = cv.imread(test_path, cv.COLOR_BGR2RGBA) #img1
        if self.ref is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        if self.test is None:
            raise ValueError(f"Could not load test image: {test_path}")
    def realign(self):
        features0 = FeatureExtraction(self.ref)
        features1 = FeatureExtraction(self.test)
        matches = feature_matching(features0, features1)
        # matched_image = cv.drawMatches(img0, features0.kps, \
        # img1, features1.kps, matches, None, flags=2)

        H, _ = cv.findHomography( features0.matched_pts, \
            features1.matched_pts, cv.RANSAC, 5.0)

        h, w, c = self.test.shape
        warped = cv.warpPerspective(self.ref, H, (w, h), \
            borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        output = np.zeros((h, w, 3), np.uint8)
        alpha = warped[:, :, 3] / 255.0
        output[:, :, 0] = (1. - alpha) * self.test[:, :, 0] + alpha * warped[:, :, 0]
        output[:, :, 1] = (1. - alpha) * self.test[:, :, 1] + alpha * warped[:, :, 1]
        output[:, :, 2] = (1. - alpha) * self.test[:, :, 2] + alpha * warped[:, :, 2]
        #return output
                # Visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].imshow(cv.cvtColor(self.ref, cv.COLOR_BGR2RGB))
        axes[0,0].set_title('Reference')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(cv.cvtColor(self.test, cv.COLOR_BGR2RGB))
        axes[0,1].set_title('Test')
        axes[0,1].axis('off')
        
        matched_img = cv.drawMatches(self.ref, features0.kps, self.test, features1.kps, matches[:30], None, flags=2)
        axes[1,0].imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
        axes[1,0].set_title(f'Matches ({len(matches)})')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        axes[1,1].set_title('Result')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return output
        
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

    def sliding_ncc(self, patch_size=31):
        ref, test = self.ref, self.test
        self.realign()
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




