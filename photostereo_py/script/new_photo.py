import cv2 as cv
import numpy as np
import vtk
import OpenEXR
import Imath, json5


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

    def save_normals_to_exr(self,filename, normals):
        """
        Save a 3-channel float32 normal map to an OpenEXR file.
        normals: H x W x 3 NumPy array with float32 values.
        """
        if normals.dtype != np.float32:
            normals = normals.astype(np.float32)
        
        height, width, channels = normals.shape
        assert channels == 3, "Expected a 3-channel normal map"

        # Set up OpenEXR header
        header = OpenEXR.Header(width, height)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        header['channels'] = {
            'R': Imath.Channel(FLOAT),
            'G': Imath.Channel(FLOAT),
            'B': Imath.Channel(FLOAT)
        }

        # Split channels and convert to bytes
        R = normals[:, :, 0].tobytes()
        G = normals[:, :, 1].tobytes()
        B = normals[:, :, 2].tobytes()

        # Write to EXR
        exr_file = OpenEXR.OutputFile(filename, header)
        exr_file.writePixels({'R': R, 'G': G, 'B': B})
        exr_file.close()
        
    def process(self,images_arr_raw : list ): 
        ''' process creates the normal map for the given raw images '''
        # apply mask
        #todo
          
        image_arr = []
        for image in images_arr_raw: #image array is a list of uint8 cause cv does that at imread
            #we normalise the values and convert them to a float
            image = image.astype(np.float32) #conversion
            image = image/255 # normalising
            image_arr.append(image)
            image = image.reshape(-1) #reshape to a flat vector [1,h*w] image
            self.intensity.append(image) # add to intensities as each is a vector of pixel intensities [12 ,h*w]
            
        self.intensity = np.array(self.intensity)    
        h,w = image_arr[0].shape[:2] # setting height and width of the normal space 
                                  # should be the size of the image
	    	
        self.normals = np.zeros((h,w,3), dtype= np.float32)# position of each normal in x and y of the picture and then the xyz components of the normal itself 
        self.p = np.zeros((h,w), dtype = np.float32) #[h*w]
        self.q = np.zeros((h,w), dtype = np.float32)#[h*w]
        self.source_mat = np.array(self.light_matrix) #[12 x 3] assuming light matrix is 12 x 3
        pinv_source_mat = np.linalg.pinv(self.source_mat) # does the pseudo inverse  (SS^t)^-1 S^t => [3x12]
        
        # now to get N
        scaled_normals_T= pinv_source_mat@ self.intensity # 3x12 @ [12,h*w] => (3,h*w )matrix #flat
        scaled_normals = scaled_normals_T.T # transpose to make it a Px3 matrix that we can make h,w,3 again #flat
        scaled_normals = scaled_normals.reshape((h, w, 3)) #no longer flat
        print(scaled_normals.shape , "is the scaled normal shape")
        # Compute albedo (rho) = ||scaled_normal|| for each pixel
        # Take L2 norm along the last axis (3D vector at each pixel)
        self.albedo = np.linalg.norm(scaled_normals, axis=2)  # albedo: (h, w) - magnitude of scaled normals
        epsilon = 1e-8
        albedo_safe = np.maximum(self.albedo, epsilon)  # albedo_safe: (h, w) - albedo with minimum threshold
        
        self.normals = scaled_normals
        print(self.normals.shape , "is the normal shape")
        self.normals = self.normals / albedo_safe[:, :, np.newaxis]
        nz_safe = np.maximum(np.abs(self.normals[:, :, 2]), epsilon)  # nz_safe: (h, w) - safe z-component of normals
      
        # p = -dZ/dx = -Nx/Nz, q = -dZ/dy = -Ny/Nz apparantly a very common formula in Computer Vision
        self.p = -(self.normals[:,:,0])/(nz_safe)
        self.q = -(self.normals[:,:,1])/(nz_safe)
        
        
        # Handle points where Nz is close to zero (surface nearly perpendicular to viewing direction)
        # implement if needed
        print(f"Processing complete:")
        print(f"- Image dimensions: {h} x {w}")
        print(f"- Number of light sources: {self.source_mat.shape[0]}")  # source_mat.shape: (n_images, 3)
        print(f"- Intensity matrix shape: {self.intensity.shape}")  # intensity.shape: (n_images, h*w)
        print(f"- Scaled normals shape: {scaled_normals.shape}")  # scaled_normals.shape: (h, w, 3)
        print(f"- Albedo shape: {self.albedo.shape}")  # albedo.shape: (h, w)
        print(f"- Unit normals shape: {self.normals.shape}")  # normals.shape: (h, w, 3)
        print(f"- Gradients p,q shape: {self.p.shape}, {self.q.shape}")  # p.shape, q.shape: (h, w)
        print(f"- Albedo range: [{np.min(self.albedo):.3f}, {np.max(self.albedo):.3f}]")
        print(f"- Normal range: X[{np.min(self.normals[:,:,0]):.3f}, {np.max(self.normals[:,:,0]):.3f}]")
        print(f"- Normal range: Y[{np.min(self.normals[:,:,1]):.3f}, {np.max(self.normals[:,:,1]):.3f}]")
        print(f"- Normal range: Z[{np.min(self.normals[:,:,2]):.3f}, {np.max(self.normals[:,:,2]):.3f}]")
        
        normals_float = self.normals.astype(np.float32)
        self.save_normals_to_exr("normal_mapping.exr",normals_float)

    

