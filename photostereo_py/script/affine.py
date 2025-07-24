import cv2 as cv
import numpy as np
import copy
import logging
import pickle
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

       # if img.dtype != np.uint8:
        self.img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
       # else:
      #      img_8bit = img.copy()
                # Convert to grayscale properly
        logger.info(f"[FeatureExtraction] Received image with shape: {img.shape}")

        if len(self.img_8bit.shape) == 2:
            # Already grayscale
            self.gray_img = copy.copy(img)
        elif len(img.shape) == 3:
            # Convert to grayscale
            if self.img_8bit.shape[2] == 3:  # BGR
                self.gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif self.img_8bit.shape[2] == 4:  # BGRA
                self.gray_img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[2]}")
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")
        
        #redu
        #redundant
        if self.gray_img.dtype != np.uint8:
             logger.info(f"Converting input image from {self.gray_img.dtype} to uint8")
             self.gray_img = cv.normalize(self.gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        logger.info(f"Original image shape: {self.img.shape}")
        logger.info(f"Grayscale image shape: {self.gray_img.shape}")
        
        
        self.kps, self.des = orb.detectAndCompute( \
            self.gray_img, None)
        self.img_kps = cv.drawKeypoints( \
            self.img_8bit, self.kps, 0, \
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
    '''pass any image in the form of a numpy array to reallign it'''
    def __init__(self, ref_img:np.array, test_img:np.array):
        # Load images with all channels preserved
        self.ref = ref_img
        self.test = test_img
        self.ref_orig = self.ref.copy()
        self.test_orig = self.test.copy()
        
        self.ref_8bit = cv.normalize(self.ref, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        self.test_8bit = cv.normalize(self.test, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
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
        
        # NEW: Log final output statistics
        logger.info(f"[Final Output] Shape: {output.shape}, dtype: {output.dtype}")
        logger.info(f"[Final Output] Value range: [{output.min():.3f}, {output.max():.3f}]")
        logger.info(f"[Final Output] Non-zero pixels: {np.count_nonzero(output)}/{output.size} ({np.count_nonzero(output)/output.size:.2%})")

        return output
        
