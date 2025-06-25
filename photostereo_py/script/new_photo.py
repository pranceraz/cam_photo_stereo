import cv2 as cv
import numpy as np
import vtk

class new_photo:
    
    def  __init__(self, image_count : int ,light_matrix : list):
        self.image_count = image_count
        self.p = []
        self.q = []
        self.albedo  = [] # rho
        self.mask = []
        self.z = []
        self.pi = np.pi
        self.light_matrix = light_matrix
        self.normals = []  #N = (SS^t)S^t I
        self.intensity = [] # I
        self.source_mat = [] # S
    def process(self,images_arr_raw : list ): #
        # apply mask
        #todo
          
        image_arr = []
        for image in images_arr_raw: #image array is a list of uint8 cause cv does that at imread
            #we normalise the values and convert them to a float
            image = image.astype(np.float32) #conversion
            image = image/255 # normalising
            image_arr.append(image)
            image = image.reshape(-1) #reshape to a flat vector
            self.intensity.append(image) # add to intensities as each is a vector of pixel intensities
            
            
        h,w = image_arr[0].shape[:2] # setting height and width of the normal space 
                                  # should be the size of the image
	    	
        self.normals = np.zeros((h,w,3), dtype= np.float32)# position of each normal in x and y of the picture and then the xyz components of the normal itself 
        self.p = np.zeros((h,w), dtype = np.float32)
        self.q = np.zeros((h,w), dtype = np.float32)
        self.source_mat = self.light_matrix
        pinv_source_mat = np.linalg.pinv(self.source_mat) # does the pseudo inverse  (SS^t)S^t 
        
        # now to get N
        scaled_normals_T= pinv_source_mat@ self.intensity # 3x12 @ 12xP => 3xP matrix
        scaled_normals = scaled_normals_T.T # transpose to make it a Px3 matrix tat we can make h,w,3 again
       # big_N.reshape((h,w,3))
       #Compute albedo (rho) = ||N|| we leave out the Pi for scaling!!!!!! not sure how/why
        
        
        
        
        
            
			
            

        
        