import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from misc.feature_extract import FeatureExtraction, feature_matching
import sys
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
        # Load images with all channels preserved
        self.ref = cv.imread(ref_path, cv.IMREAD_UNCHANGED)
        self.test = cv.imread(test_path, cv.IMREAD_UNCHANGED)
        self.ref_orig = self.ref.copy()
        self.test_orig = self.test.copy()
        self.ref_8bit = self.to_8bit(self.ref)
        self.test_8bit = self.to_8bit(self.test)

        if self.ref is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        if self.test is None:
            raise ValueError(f"Could not load test image: {test_path}")
        print(f"Reference image shape: {self.ref.shape}")
        print(f"Test image shape: {self.test.shape}")

    def realign(self, output_png:bool = True):
        features0 = FeatureExtraction(self.ref_orig)
        features1 = FeatureExtraction(self.test_orig)
        matches = feature_matching(features0, features1)
        # matched_image = cv.drawMatches(img0, features0.kps, \
        # img1, features1.kps, matches, None, flags=2)

        H, _ = cv.findHomography( features1.matched_pts, \
            features0.matched_pts, cv.RANSAC, 5.0)

        h, w, c = self.ref.shape
        warped = cv.warpPerspective(self.test, H, (w, h), \
            borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        output = warped
        # output = np.zeros((h, w, 3), np.uint8)
        #alpha = warped[:, :, 3] / 255.0
        # output[:, :, 0] = (1. - alpha) * self.test[:, :, 0] + alpha * warped[:, :, 0]
        # output[:, :, 1] = (1. - alpha) * self.test[:, :, 1] + alpha * warped[:, :, 1]
        # output[:, :, 2] = (1. - alpha) * self.test[:, :, 2] + alpha * warped[:, :, 2]
        #return output
                # Visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,1].imshow(cv.cvtColor(self.ref_8bit, cv.COLOR_BGR2RGB))
        axes[0,1].set_title('Reference')
        axes[0,1].axis('off')
        
        axes[0,0].imshow(cv.cvtColor(self.test_8bit, cv.COLOR_BGR2RGB))
        axes[0,0].set_title('Test')
        axes[0,0].axis('off')
        
        matched_img = cv.drawMatches(self.ref_8bit, features0.kps, self.test_8bit, features1.kps, matches[:30], None, flags=2)
        axes[1,0].imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
        axes[1,0].set_title(f'Matches ({len(matches)})')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(cv.cvtColor(self.to_8bit(output), cv.COLOR_BGR2RGB))
        axes[1,1].set_title('Result')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        print(f"[DEBUG] Output dtype: {output.dtype}, max: {output.max()}, min: {output.min()}")
        if output_png == True:
            return  cv.imwrite("aligned_test_to_ref.png", output)
        else:
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

    def sliding_ncc(self,scale = 0.5 ,patch_size=31):
        ref, test = self.ref, self.test
        #self.realign()
        ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
        test = cv.cvtColor(test, cv.COLOR_BGR2GRAY) if test.ndim == 3 else test
        ref = cv.resize(ref, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        test = cv.resize(test, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        pad = patch_size // 2
        ref_p = np.pad(ref, pad)
        test_p = np.pad(test, pad)
        
        H, W  = ref.shape
        ncc_map = np.zeros((H, W))

        for y in range(H):
            if y % 10 == 0:
                print(f"[DEBUG] Processing row {y+1}/{H}", file=sys.stderr)
            for x in range(W):
                ref_patch = ref_p[y:y+patch_size, x:x+patch_size].astype(np.float32)
                test_patch = test_p[y:y+patch_size, x:x+patch_size].astype(np.float32)
                
                ref_mean = np.mean(ref_patch)
                test_mean = np.mean(test_patch)
                
                num = np.sum((ref_patch - ref_mean) * (test_patch - test_mean))
                denom = np.sqrt(np.sum((ref_patch - ref_mean) ** 2) * np.sum((test_patch - test_mean) ** 2) + 1e-6)
                
                ncc_map[y, x] = num / denom
                
        return ncc_map

    def to_8bit(self,img16):
        return cv.normalize(img16, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def rgb_to_normal_map(img):
        """
        Converts an RGB image into a normalized normal map with vectors in [-1, 1].
        """
        normals = img.astype(np.float32) / 255.0 * 2 - 1  # scale to [-1, 1]
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        norm[norm == 0] = 1  # prevent division by zero
        return normals / norm
    ##################################################
    '''Add a way to directly enter normal matrices'''
    ##################################################
    @staticmethod
    def crop_center(img, crop_fraction):
        """
        Crops a centered square region from an image based on crop_fraction.
        crop_fraction=1/8 means the cropped region is (1/8)*width and (1/8)*height in size.
        """
        H, W = img.shape[:2]
        ch, cw = int(H * crop_fraction), int(W * crop_fraction)
        y1 = (H - ch) // 2
        x1 = (W - cw) // 2
        return img[y1:y1+ch, x1:x1+cw]
    
    def sliding_cosine_similarity(self, patch_size=31, scale=1.0, crop_frac= .8):
        """
        Computes a local cosine similarity map between reference and test normal maps.
        At each pixel, compares corresponding patches from both images using mean dot product.
        """
        # Resize and convert to normal maps
        ref_img, test_img = self.ref, self.test
        ref_img = self.crop_center(ref_img,crop_frac)
        test_img = self.crop_center(test_img,crop_frac)
        ref = cv.resize(ref_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        test = cv.resize(test_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        ref = self.rgb_to_normal_map(ref)
        test = self.rgb_to_normal_map(test)

        pad = patch_size // 2

        # Pad both images to handle borders
        ref_p = cv.copyMakeBorder(ref, pad, pad, pad, pad, cv.BORDER_REFLECT)
        test_p = cv.copyMakeBorder(test, pad, pad, pad, pad, cv.BORDER_REFLECT)

        H, W = ref.shape[:2]
        ncc_map = np.zeros((H, W), dtype=np.float32)

        for y in range(H):
            if y % 10 == 0:
                print(f"[DEBUG] Sliding similarity row {y+1}/{H}", file=sys.stderr)
            for x in range(W):
                # Extract local patches
                ref_patch = ref_p[y:y+patch_size, x:x+patch_size, :]
                test_patch = test_p[y:y+patch_size, x:x+patch_size, :]

                # Flatten and compute dot product
                dot = np.sum(ref_patch.reshape(-1, 3) * test_patch.reshape(-1, 3), axis=1)
                ncc_map[y, x] = np.mean(dot)

        return ncc_map
    
    def vector_cross_correlation(self, patch_size=31, scale=1.0, crop_frac=.8):
        """
        Computes cross-correlation of a reference patch (cropped from center) across the test image.
        At each location in the test image, computes average dot product with the reference patch.
        """
        # Resize and convert
        ref_img, test_img = self.ref, self.test
        ref = cv.resize(ref_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        test = cv.resize(test_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        ref = self.rgb_to_normal_map(ref)
        test = self.rgb_to_normal_map(test)

        # Extract reference patch from center
        ref_patch = self.crop_center(ref, crop_frac)
        ph, pw = ref_patch.shape[:2]

        H, W = test.shape[:2]
        out_h = H - ph + 1
        out_w = W - pw + 1

        corr_map = np.zeros((out_h, out_w), dtype=np.float32)

        # Slide reference patch over test image and compute mean dot product at each position
        for y in range(out_h):
            if y % 10 == 0:
                print(f"[DEBUG] Cross-correlation row {y+1}/{out_h}", file=sys.stderr)
            for x in range(out_w):
                test_patch = test[y:y+ph, x:x+pw, :]
                dot = np.sum(ref_patch.reshape(-1, 3) * test_patch.reshape(-1, 3), axis=1)
                corr_map[y, x] = np.mean(dot)

        return corr_map
    
    def visualize_similarity_map(self, sim_map, title="Similarity Map"):
        """
        Displays a similarity or correlation map using matplotlib with better contrast and color.
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(sim_map, cmap='plasma', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()