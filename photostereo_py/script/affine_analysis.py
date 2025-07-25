# import cv2 as cv
# import numpy as np
# import copy
# import logging
# import pickle
# import matplotlib.pyplot as plt
# # from image_utils import image_utils
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# orb = cv.ORB_create(
#     nfeatures=10000,
#     scaleFactor=1.2,
#     scoreType=cv.ORB_HARRIS_SCORE)

# class FeatureExtraction:
#     def __init__(self, img):
#         self.img_orig = img
#         self.img = copy.copy(img)
#         if not isinstance(img, np.ndarray):
#             raise TypeError("Input must be a numpy array.")

#         if img.ndim not in [2, 3]:
#             raise ValueError(f"Unsupported image shape: {img.shape}")

       
#         self.img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
#         logger.info(f"[FeatureExtraction] Received image with shape: {img.shape}")

#         if len(self.img_8bit.shape) == 2:
#             # Already grayscale
#             self.gray_img = copy.copy(img)
#         elif len(img.shape) == 3:
#             # Convert to grayscale
#             if self.img_8bit.shape[2] == 3:  # BGR
#                 self.gray_img = cv.cvtColor(self.img_8bit, cv.COLOR_BGR2GRAY)
#             elif self.img_8bit.shape[2] == 4:  # BGRA
#                 self.gray_img = cv.cvtColor(self.img_8bit, cv.COLOR_BGRA2GRAY)
#             else:
#                 raise ValueError(f"Unexpected number of channels: {img.shape[2]}")
#         else:
#             raise ValueError(f"Unexpected image dimensions: {img.shape}")
        
        
#         #redundant
#         if self.gray_img.dtype != np.uint8:
#              logger.info(f"Converting input image from {self.gray_img.dtype} to uint8")
#              self.gray_img = cv.normalize(self.gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

#         logger.info(f"Original image shape: {self.img.shape}")
#         logger.info(f"Grayscale image shape: {self.gray_img.shape}")
        
        
#         self.kps, self.des = orb.detectAndCompute( \
#             self.gray_img, None)
#         # self.img_kps = cv.drawKeypoints( \
#         #     self.img_8bit, self.kps, 0, \
#         #     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         self.matched_pts = []

# class FeatureMatching():
#     def __init__(self):
        
#         self.LOWES_RATIO = 0.7
#         self.MIN_MATCHES = 5
#         self.index_params = dict(
#             algorithm = 6, # FLANN_INDEX_LSH
#             table_number = 6,
#             key_size = 10,
#             multi_probe_level = 2)
#         self.search_params = dict(checks=50)
#         self.flann = cv.FlannBasedMatcher( #(Fast Library for Approximate Nearest Neighbors)
#             self.index_params,
#             self.search_params)

#     def feature_matching(self,features0: FeatureExtraction, features1:FeatureExtraction):
#         matches = [] # good matches as per Lowe's ratio test
        
#         #Checks if the first image has valid descriptors and at least 3 feature points (minimum needed for meaningful matching).
#         if(features0.des is not None and len(features0.des) > 2):
#             all_matches = self.flann.knnMatch( \
#                 features0.des, features1.des, k=2)
#             try:
#                 for m,n in all_matches:
#                     if m.distance < self.LOWES_RATIO * n.distance:
#                         matches.append(m)
#             except Exception as e:
#                 logger.error(f"Matching failed: {e}") 
                
#             if(len(matches) > self.MIN_MATCHES):    
#                 features0.matched_pts = np.float32( \
#                     [ features0.kps[m.queryIdx].pt for m in matches ] \
#                         ).reshape(-1,1,2)
                
#                 features1.matched_pts = np.float32( \
#                     [ features1.kps[m.trainIdx].pt for m in matches ] \
#                         ).reshape(-1,1,2)
#             else:
#                 logger.warning(f"Not enough matches found: {len(matches)}. not populating image features with matches Minimum required: {self.MIN_MATCHES}")   
                             
#         return matches 
    
# class ImageAligner():
#     def __init__(self, ref_img:np.array, test_img:np.array):
#         # Load images with all channels preserved
#         self.ref = ref_img
#         self.test = test_img
#         self.ref_orig = self.ref.copy()
#         self.test_orig = self.test.copy()
        
#         # self.ref_8bit = cv.normalize(self.ref, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
#         # self.test_8bit = cv.normalize(self.test, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
#         # self.ref_8bit = image_utils.to_8bit(self.ref)
#         # self.test_8bit = image_utils.to_8bit(self.test)

#         if self.ref is None:
#             logger.error('no reference image')
#         if self.test is None:
#             logger.error('no test image') 
            
#         logger.info(f"Reference image shape: {self.ref.shape}")
#         logger.info(f"Test image shape: {self.test.shape}")


    
#     def realign(self)-> np.array:
#         features0 = FeatureExtraction(self.ref_orig)
#         features1 = FeatureExtraction(self.test_orig)
#         matcher = FeatureMatching()
#         matches = matcher.feature_matching(features0, features1)
#          # Log match statistics
        
#         self.num_matches = len(matches)
#         logger.info(f"[ImageAligner] Number of keypoints in reference: {len(features0.kps)}")
#         logger.info(f"[ImageAligner] Number of keypoints in test: {len(features1.kps)}")
#         logger.info(f"[ImageAligner] Number of good matches: {self.num_matches}")
        
#         # NEW: Log match quality statistics
#         if len(matches) > 0:
#             distances = [m.distance for m in matches]
#             logger.info(f"[Match Quality] Distance stats - Mean: {np.mean(distances):.2f}, Min: {np.min(distances):.2f}, Max: {np.max(distances):.2f}, Std: {np.std(distances):.2f}")
        
#         logger.debug(f"[ImageAligner] Matches: {features0.matched_pts}, and {features1.matched_pts} and their shapes are {features0.matched_pts.shape} and {features1.matched_pts.shape}")
        
#         if self.num_matches < matcher.MIN_MATCHES:
#             logger.warning("not enough matches will not populate common points")
#             return None
         
      
        
        
#         A,_ = cv.estimateAffinePartial2D(
#         features1.matched_pts, features0.matched_pts, 
#         method=cv.RANSAC, ransacReprojThreshold=5.0)
#         h, w, c = self.ref.shape
#         if A is not None: 
#             warped = cv.warpAffine(self.test_orig, A, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        
#         output = warped

#         return output

# # ==================== ANALYSIS TOOL ====================

# class ORBAnalysisTool:
#     def __init__(self, ref_img, test_img):
#         self.ref_img = ref_img
#         self.test_img = test_img
    
#     def analyze_and_plot(self):
#         """Create simple analysis plots"""
        
#         # Run the existing alignment process to get data
#         aligner = ImageAligner(self.ref_img, self.test_img)
#         features0 = FeatureExtraction(self.ref_img)
#         features1 = FeatureExtraction(self.test_img)
#         matcher = FeatureMatching()
#         matches = matcher.feature_matching(features0, features1)
        
#         # Get pixel difference
#         ref_gray = features0.gray_img
#         test_gray = features1.gray_img
#         h, w = min(ref_gray.shape[0], test_gray.shape[0]), min(ref_gray.shape[1], test_gray.shape[1])
#         ref_resized = cv.resize(ref_gray, (w, h))
#         test_resized = cv.resize(test_gray, (w, h))
#         pixel_diff = cv.absdiff(ref_resized, test_resized)
#         mean_diff = np.mean(pixel_diff)
        
#         # Create 3 simple plots
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Plot 1: Keypoint Detection
#         ref_kps_count = len(features0.kps)
#         test_kps_count = len(features1.kps)
#         categories = ['Reference\nImage', 'Test Image\n(Rotated)']
#         keypoint_counts = [ref_kps_count, test_kps_count]
#         colors = ['green', 'orange']
        
#         bars1 = axes[0].bar(categories, keypoint_counts, color=colors, alpha=0.7)
#         axes[0].set_title('Keypoints Detected by ORB', fontsize=14, fontweight='bold')
#         axes[0].set_ylabel('Number of Keypoints')
#         axes[0].grid(True, alpha=0.3)
        
#         for bar, value in zip(bars1, keypoint_counts):
#             axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
#                         str(value), ha='center', va='bottom', fontweight='bold')
        
#         # Plot 2: Matching Performance
#         matches_count = len(matches)
#         match_data = ['Keypoints in\nReference', 'Keypoints in\nTest', 'Good Matches\nFound']
#         match_values = [ref_kps_count, test_kps_count, matches_count]
#         match_colors = ['green', 'orange', 'red' if matches_count < 10 else 'blue']
        
#         bars2 = axes[1].bar(match_data, match_values, color=match_colors, alpha=0.7)
#         axes[1].set_title('Feature Matching Results', fontsize=14, fontweight='bold')
#         axes[1].set_ylabel('Count')
#         axes[1].grid(True, alpha=0.3)
#         axes[1].axhline(y=5, color='red', linestyle='--', linewidth=2, 
#                        label='Minimum for reliable matching')
#         axes[1].legend()
        
#         for bar, value in zip(bars2, match_values):
#             axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
#                         str(value), ha='center', va='bottom', fontweight='bold')
        
#         # Plot 3: Performance Summary
#         performance_metrics = ['Pixel\nDifference', 'Match\nSuccess', 'Alignment\nSuccess']
#         pixel_score = min(100, (mean_diff / 255) * 100)
#         match_score = 100 if matches_count >= 10 else (matches_count / 10) * 100
#         alignment_score = 100 if matches_count >= 5 else 0
        
#         performance_values = [pixel_score, match_score, alignment_score]
#         perf_colors = ['red' if v > 50 else 'green' if v > 20 else 'darkred' for v in performance_values]
        
#         bars3 = axes[2].bar(performance_metrics, performance_values, color=perf_colors, alpha=0.7)
#         axes[2].set_title('Performance Impact', fontsize=14, fontweight='bold')
#         axes[2].set_ylabel('Performance Score (0-100)')
#         axes[2].set_ylim(0, 100)
#         axes[2].grid(True, alpha=0.3)
#         axes[2].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate')
#         axes[2].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Poor')
#         axes[2].legend()
        
#         for bar, value in zip(bars3, performance_values):
#             axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
#                         f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
#         plt.tight_layout()
#         plt.show()
        
#         # Print results
#         print("\n" + "="*50)
#         print("ORB ANALYSIS RESULTS")
#         print("="*50)
#         print(f"Pixel Difference (Mean): {mean_diff:.1f} / 255")
#         print(f"Reference Keypoints: {ref_kps_count}")
#         print(f"Test Image Keypoints: {test_kps_count}")
#         print(f"Good Matches Found: {matches_count}")
#         print(f"Minimum Needed: 5 matches")
#         print(f"Alignment: {'SUCCESS' if matches_count >= 5 else 'FAILED'}")
        
#         if matches_count < 5:
#             print(f"\n⚠️  WARNING: Too few matches for reliable alignment!")
#             print(f"   Reason: Large pixel differences confuse ORB keypoint detection")
#         elif matches_count < 20:
#             print(f"\n⚠️  CAUTION: Low match count may cause unstable alignment")
#         else:
#             print(f"\n✅ GOOD: Sufficient matches for reliable alignment")
        
#         return {
#             'pixel_diff_mean': mean_diff,
#             'ref_keypoints': ref_kps_count,
#             'test_keypoints': test_kps_count,
#             'matches': matches_count,
#             'success': matches_count >= 5
#         }

import cv2 as cv
import numpy as np
import copy
import logging
import pickle
import matplotlib.pyplot as plt
# from image_utils import image_utils
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
orb = cv.ORB_create(
    nfeatures=10000,
    scaleFactor=1.2,
    scoreType=cv.ORB_HARRIS_SCORE)

class FeatureExtraction:
    def __init__(self, img):
        self.img_orig = img
        self.img = copy.copy(img)
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if img.ndim not in [2, 3]:
            raise ValueError(f"Unsupported image shape: {img.shape}")

       
        self.img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        logger.info(f"[FeatureExtraction] Received image with shape: {img.shape}")

        if len(self.img_8bit.shape) == 2:
            # Already grayscale
            self.gray_img = copy.copy(img)
        elif len(img.shape) == 3:
            # Convert to grayscale
            if self.img_8bit.shape[2] == 3:  # BGR
                self.gray_img = cv.cvtColor(self.img_8bit, cv.COLOR_BGR2GRAY)
            elif self.img_8bit.shape[2] == 4:  # BGRA
                self.gray_img = cv.cvtColor(self.img_8bit, cv.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[2]}")
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")
        
        
        #redundant
        if self.gray_img.dtype != np.uint8:
             logger.info(f"Converting input image from {self.gray_img.dtype} to uint8")
             self.gray_img = cv.normalize(self.gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        logger.info(f"Original image shape: {self.img.shape}")
        logger.info(f"Grayscale image shape: {self.gray_img.shape}")
        
        
        self.kps, self.des = orb.detectAndCompute( \
            self.gray_img, None)
        # self.img_kps = cv.drawKeypoints( \
        #     self.img_8bit, self.kps, 0, \
        #     flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.matched_pts = []

class FeatureMatching():
    def __init__(self):
        
        self.LOWES_RATIO = 0.7
        self.MIN_MATCHES = 5
        self.index_params = dict(
            algorithm = 6, # FLANN_INDEX_LSH
            table_number = 6,
            key_size = 10,
            multi_probe_level = 2)
        self.search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher( #(Fast Library for Approximate Nearest Neighbors)
            self.index_params,
            self.search_params)

    def feature_matching(self,features0: FeatureExtraction, features1:FeatureExtraction):
        matches = [] # good matches as per Lowe's ratio test
        
        #Checks if the first image has valid descriptors and at least 3 feature points (minimum needed for meaningful matching).
        if(features0.des is not None and len(features0.des) > 2):
            all_matches = self.flann.knnMatch( \
                features0.des, features1.des, k=2)
            try:
                for m,n in all_matches:
                    if m.distance < self.LOWES_RATIO * n.distance:
                        matches.append(m)
            except Exception as e:
                logger.error(f"Matching failed: {e}") 
                
            if(len(matches) > self.MIN_MATCHES):    
                features0.matched_pts = np.float32( \
                    [ features0.kps[m.queryIdx].pt for m in matches ] \
                        ).reshape(-1,1,2)
                
                features1.matched_pts = np.float32( \
                    [ features1.kps[m.trainIdx].pt for m in matches ] \
                        ).reshape(-1,1,2)
            else:
                logger.warning(f"Not enough matches found: {len(matches)}. not populating image features with matches Minimum required: {self.MIN_MATCHES}")   
                             
        return matches 
    
class ImageAligner():
    def __init__(self, ref_img:np.array, test_img:np.array):
        # Load images with all channels preserved
        self.ref = ref_img
        self.test = test_img
        self.ref_orig = self.ref.copy()
        self.test_orig = self.test.copy()
        
        # self.ref_8bit = cv.normalize(self.ref, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        # self.test_8bit = cv.normalize(self.test, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        # self.ref_8bit = image_utils.to_8bit(self.ref)
        # self.test_8bit = image_utils.to_8bit(self.test)

        if self.ref is None:
            logger.error('no reference image')
        if self.test is None:
            logger.error('no test image') 
            
        logger.info(f"Reference image shape: {self.ref.shape}")
        logger.info(f"Test image shape: {self.test.shape}")


    
    def realign(self)-> np.array:
        features0 = FeatureExtraction(self.ref_orig)
        features1 = FeatureExtraction(self.test_orig)
        matcher = FeatureMatching()
        matches = matcher.feature_matching(features0, features1)
         # Log match statistics
        
        self.num_matches = len(matches)
        logger.info(f"[ImageAligner] Number of keypoints in reference: {len(features0.kps)}")
        logger.info(f"[ImageAligner] Number of keypoints in test: {len(features1.kps)}")
        logger.info(f"[ImageAligner] Number of good matches: {self.num_matches}")
        
        # NEW: Log match quality statistics
        if len(matches) > 0:
            distances = [m.distance for m in matches]
            logger.info(f"[Match Quality] Distance stats - Mean: {np.mean(distances):.2f}, Min: {np.min(distances):.2f}, Max: {np.max(distances):.2f}, Std: {np.std(distances):.2f}")
        
        logger.debug(f"[ImageAligner] Matches: {features0.matched_pts}, and {features1.matched_pts} and their shapes are {features0.matched_pts.shape} and {features1.matched_pts.shape}")
        
        if self.num_matches < matcher.MIN_MATCHES:
            logger.warning("not enough matches will not populate common points")
            return None
         
      
        
        
        A,_ = cv.estimateAffinePartial2D(
        features1.matched_pts, features0.matched_pts, 
        method=cv.RANSAC, ransacReprojThreshold=5.0)
        h, w, c = self.ref.shape
        if A is not None: 
            warped = cv.warpAffine(self.test_orig, A, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        
        output = warped

        return output

# ==================== ANALYSIS TOOL ====================

class ORBAnalysisTool:
    def __init__(self, ref_img, test_img):
        self.ref_img = ref_img
        self.test_img = test_img
    
    def analyze_and_plot(self, save_dir="analysis_plots"):
        """Create simple analysis plots and save separately"""
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Run the existing alignment process to get data
        aligner = ImageAligner(self.ref_img, self.test_img)
        features0 = FeatureExtraction(self.ref_img)
        features1 = FeatureExtraction(self.test_img)
        matcher = FeatureMatching()
        matches = matcher.feature_matching(features0, features1)
        
        # Get pixel difference
        ref_gray = features0.gray_img
        test_gray = features1.gray_img
        h, w = min(ref_gray.shape[0], test_gray.shape[0]), min(ref_gray.shape[1], test_gray.shape[1])
        ref_resized = cv.resize(ref_gray, (w, h))
        test_resized = cv.resize(test_gray, (w, h))
        pixel_diff = cv.absdiff(ref_resized, test_resized)
        mean_diff = np.mean(pixel_diff)
        
        ref_kps_count = len(features0.kps)
        test_kps_count = len(features1.kps)
        matches_count = len(matches)
        
        # Plot 1: Keypoint Detection
        plt.figure(figsize=(8, 6))
        categories = ['Reference\nImage', 'Test Image\n(Rotated)']
        keypoint_counts = [ref_kps_count, test_kps_count]
        colors = ['green', 'orange']
        
        bars1 = plt.bar(categories, keypoint_counts, color=colors, alpha=0.7, width=0.1)
        plt.title('Keypoints Detected by ORB', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Keypoints', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, keypoint_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    str(value), ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/1_keypoint_detection.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Matching Performance
        plt.figure(figsize=(10, 6))
        match_data = ['Keypoints in\nReference', 'Keypoints in\nTest', 'Good Matches\nFound']
        match_values = [ref_kps_count, test_kps_count, matches_count]
        match_colors = ['green', 'orange', 'red' if matches_count < 10 else 'blue']
        
        bars2 = plt.bar(match_data, match_values, color=match_colors, alpha=0.7, width=0.1)
        plt.title('Feature Matching Results', fontsize=16, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=5, color='red', linestyle='--', linewidth=2, 
                   label='Minimum for reliable matching')
        plt.legend(fontsize=12)
        
        for bar, value in zip(bars2, match_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(value), ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/2_matching_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Performance Summary
        plt.figure(figsize=(10, 6))
        performance_metrics = ['Pixel\nDifference', 'Match\nSuccess', 'Alignment\nSuccess']
        pixel_score = min(100, (mean_diff / 255) * 100)
        match_score = 100 if matches_count >= 10 else (matches_count / 10) * 100
        alignment_score = 100 if matches_count >= 5 else 0
        
        performance_values = [pixel_score, match_score, alignment_score]
        perf_colors = ['red' if v > 50 else 'green' if v > 20 else 'darkred' for v in performance_values]
        
        bars3 = plt.bar(performance_metrics, performance_values, color=perf_colors, alpha=0.7)
        plt.title('Performance Impact', fontsize=16, fontweight='bold')
        plt.ylabel('Performance Score (0-100)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, axis='y', color='gray', linewidth=1, linestyle='--')
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Poor')
        plt.legend(fontsize=12)
        
        for bar, value in zip(bars3, performance_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/3_performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to '{save_dir}/' directory")
        
        # Print detailed results explaining good matches
        print("\n" + "="*60)
        print("ORB ANALYSIS RESULTS")
        print("="*60)
        print(f"Pixel Difference (Mean): {mean_diff:.1f} / 255")
        print(f"Reference Keypoints: {ref_kps_count}")
        print(f"Test Image Keypoints: {test_kps_count}")
        print(f"Good Matches Found: {matches_count}")
        print(f"Minimum Needed: 5 matches")
        print(f"Alignment: {'SUCCESS' if matches_count >= 5 else 'FAILED'}")
        
        print("\n" + "="*60)
        print("HOW GOOD MATCHES ARE DETERMINED:")
        print("="*60)
        print("1. ORB detects keypoints (corners/edges) in both images")
        print("2. For each keypoint, ORB creates a 256-bit descriptor")
        print("3. FLANN matcher finds 2 closest matches for each descriptor")
        print("4. Lowe's Ratio Test: distance(best) < 0.7 × distance(second_best)")
        print("   - This eliminates ambiguous matches")
        print("   - Only distinctive matches survive")
        print("5. RANSAC removes outliers during affine estimation")
        print("6. Final 'good matches' = survived all these filters")
        
        print(f"\nMATCH QUALITY ANALYSIS:")
        if len(matches) > 0:
            distances = [m.distance for m in matches]
            print(f"  Match distances - Mean: {np.mean(distances):.2f}")
            print(f"  Lower distance = better match quality")
            print(f"  Typical good range: 20-60")
            if np.mean(distances) < 30:
                print("  ✅ Excellent match quality")
            elif np.mean(distances) < 50:
                print("  ✅ Good match quality")
            else:
                print("  ⚠️  Moderate match quality")
        else:
            print("  ❌ No matches found - descriptors too different")
        
        if matches_count < 5:
            print(f"\n⚠️  WARNING: Too few matches for reliable alignment!")
            print(f"   Root Cause: Large pixel differences from rotation")
            print(f"   Effect: ORB descriptors become too different to match")
            print(f"   Solution: Use rotation-robust features (SIFT) or preprocessing")
        elif matches_count < 20:
            print(f"\n⚠️  CAUTION: Low match count may cause unstable alignment")
            print(f"   Recommendation: Verify alignment quality manually")
        else:
            print(f"\n✅ GOOD: Sufficient matches for reliable alignment")
        
        return {
            'pixel_diff_mean': mean_diff,
            'ref_keypoints': ref_kps_count,
            'test_keypoints': test_kps_count,
            'matches': matches_count,
            'success': matches_count >= 5
        }

# Usage example:
"""
# Load your coin images
ref_coin = cv.imread('straight_coin.jpg')
test_coin = cv.imread('rotated_coin.jpg')

# Run analysis
analyzer = ORBAnalysisTool(ref_coin, test_coin)
results = analyzer.analyze_and_plot()

# Also run normal alignment if needed
aligner = ImageAligner(ref_coin, test_coin)
aligned = aligner.realign()
"""

# Usage example:
# Load your coin images
ref_coin = cv.imread(r'D:\Chandana\Photometric_Stereo\cam_photo_stereo\photostereo_py\script\input\24_7_25\10rupee_before_1\10rupee_16bit_before1.png',cv.IMREAD_UNCHANGED)
test_coin = cv.imread(r'D:\Chandana\Photometric_Stereo\cam_photo_stereo\photostereo_py\script\input\24_7_25\10rupee_after_1\normal_map_16bit_10rupee_after1 .png',cv.IMREAD_UNCHANGED)

#rotated
# ref_coin = cv.imread(r'D:\Chandana\Photometric_Stereo\cam_photo_stereo\photostereo_py\script\input\24_7_25\10rupee_before_2\normal_map_16bit (5).png',cv.IMREAD_UNCHANGED)
# test_coin = cv.imread(r'D:\Chandana\Photometric_Stereo\cam_photo_stereo\photostereo_py\script\input\24_7_25\10rupee_after_2\normal_map_16bit (5).png',cv.IMREAD_UNCHANGED)

# Run analysis
analyzer = ORBAnalysisTool(ref_coin, test_coin)
results = analyzer.analyze_and_plot()

# Also run normal alignment if needed
aligner = ImageAligner(ref_coin, test_coin)
aligned = aligner.realign()
