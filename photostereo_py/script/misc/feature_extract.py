import cv2 as cv
import numpy as np
import copy


orb = cv.ORB_create(
    nfeatures=10000,
    scaleFactor=1.2,
    scoreType=cv.ORB_HARRIS_SCORE)

class FeatureExtraction:
    def __init__(self, img):
        self.img_orig = img
        self.img = copy.copy(img)
       # if img.dtype != np.uint8:
        self.img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
       # else:
      #      img_8bit = img.copy()
                # Convert to grayscale properly
        print(f"[FeatureExtraction] Received image with shape: {img.shape}")

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
             print(f"[INFO] Converting 16-bit image from {self.gray_img.dtype} to uint8")
             self.gray_img = cv.normalize(self.gray_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        print(f"Original image shape: {self.img.shape}")
        print(f"Grayscale image shape: {self.gray_img.shape}")
        self.kps, self.des = orb.detectAndCompute( \
            self.gray_img, None)
        self.img_kps = cv.drawKeypoints( \
            self.img_8bit, self.kps, 0, \
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.matched_pts = []


LOWES_RATIO = 0.7
MIN_MATCHES = 50
index_params = dict(
    algorithm = 6, # FLANN_INDEX_LSH
    table_number = 6,
    key_size = 10,
    multi_probe_level = 2)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(
    index_params,
    search_params)

def feature_matching(features0, features1):
    matches = [] # good matches as per Lowe's ratio test
    if(features0.des is not None and len(features0.des) > 2):
        all_matches = flann.knnMatch( \
            features0.des, features1.des, k=2)
        try:
            for m,n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if(len(matches) > MIN_MATCHES):    
            features0.matched_pts = np.float32([ features0.kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            features1.matched_pts = np.float32( \
                [ features1.kps[m.trainIdx].pt for m in matches ] \
                ).reshape(-1,1,2)
    return matches