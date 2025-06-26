import cv2
import numpy as np
import OpenEXR
import Imath
import height_map
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
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
        print(np.linalg.norm(self.light_matrix, axis=1))  # should be ≈1
        self.normals = []  #N = (SS^t)S^t I
        self.intensity = [] # I
        self.source_mat = [] # S

    
    def load_image_flexible(filepath):
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.png', '.bmp', '.jpg', '.jpeg']:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # or IMREAD_COLOR if needed
            if img is None:
                raise ValueError(f"Failed to load image: {filepath}")
            return img

        elif ext == '.mat':
            mat = loadmat(filepath)
            # Try to find the image data key automatically (skip __metadata__)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if not keys:
                raise ValueError(f"No valid image key found in {filepath}")
            img = mat[keys[0]]

            if img is None or not isinstance(img, np.ndarray):
                raise ValueError(f"Invalid image array in {filepath}")

            # Convert to grayscale if image is color
            if img.ndim == 3 and img.shape[2] == 3:
                # Use standard RGB to grayscale conversion
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # shape becomes (H, W)

            return img


        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
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
        
    def process(self,images_arr_raw : list ,mask): 
        ''' process creates the normal map for the given raw images '''
        # apply mask
        #todo
        if (mask is not None):
            self.mask = mask
            for id in range(0, self.image_count):
                images_arr_raw[id] = np.multiply(images_arr_raw[id], mask/255)
                
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

    def model_out(self) -> None:
        """
        Visualize the height map derived from a normalized normal vector field.

        Parameters:
            normals (np.ndarray): A normalized HxWx3 normal map with values in [-1, 1].

        Returns:
            None
        """
        normals = self.normals
        if normals.shape[-1] != 3:
            raise ValueError("Input must have shape (H, W, 3) for normalized normals.")

        # Convert back to uint8-style [0, 255] format for internal compatibility
        normal_map = ((normals / 2.0) + 0.5) * 255
        normal_map = normal_map.astype(np.uint8)

        heights = height_map.estimate_height_map(normal_map, raw_values=True,normalized_input= False,mask=self.mask)
        #heights = heights
        # Display 2D normal map and height map
        figure, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].set_title("Normal Map")
        axes[1].set_title("Height Map")
        axes[0].imshow(normal_map)
        axes[1].imshow(heights, cmap="gray")

        # Plot 3D height map
        x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        ax_3d.plot_surface(x, y, heights, cmap="viridis", edgecolor="none")

        plt.show()

        
        
        