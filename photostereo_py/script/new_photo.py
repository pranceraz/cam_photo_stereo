import cv2
import numpy as np
import OpenEXR
import Imath
import height_map
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from numpy.fft import fft2, ifft2, fftfreq
import pickle

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

    @staticmethod
    def load_image_flexible(filepath,color:bool = False):
        '''Loads images across file formats. Set color true when using with process_color'''
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.png', '.bmp', '.jpg', '.jpeg']:
            if color:
                img =  cv2.imread(filepath, cv2.IMREAD_COLOR) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
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
    
    
    @staticmethod
    def crop_square(img):
        """
        Crops a centered square region from an image
        """
        H, W = img.shape[:2] #200,100
        y1 = H // 2
        x1 = W // 2

        square_size = min(H, W)
        ch, cw = square_size//2, square_size//2

        return img[y1-ch:y1+ch, x1-cw:x1+cw]
    
    def save_normals_to_exr(self,filename, normals, reduce:bool = False):
        """
        Save a 3-channel float32 normal map to an OpenEXR file.
        normals: H x W x 3 NumPy array with float32 values.
        """
        normals[:, :, 2] *= -1  # if needed reverses the Z axis only to turn depth into height
        if normals.dtype != np.float32 and reduce == False:
            normals = normals.astype(np.float32)
        elif normals.dtype != np.float32 and reduce == True:
            normals = normals.astype(np.float16)
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
    
    def save_normals_to_png(self,filename, normals, normalize=True):
        """
        Save a 3-channel normal map to a PNG file.
        normals: H x W x 3 NumPy array with float32 or float64 values in range [-1, 1] or unnormalized.
        normalize: If True, normalize normals from [-1, 1] to [0, 255].
        """
        # Flip Z axis if needed
        #normals = normals.copy()
        normals[:, :, 2] *= -1

        if normalize:
            # Normalize from [-1, 1] to [0, 255]
            normals = (normals + 1.0) * 0.5 * 255.0
        else:
            # Clip directly to [0, 255] range
            normals = np.clip(normals, 0, 255)

        normals = np.nan_to_num(normals)  # Replace NaNs and infs with 0
        normals = normals.astype(np.uint8)

        # Convert RGB format if needed (OpenCV expects BGR)
        normals_bgr = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)

        # Save the image
        cv2.imwrite(filename, normals_bgr)

    def save_normals_to_16bit_png(self,filename, normals, square_crop = False):
        normals = normals.copy()
        if square_crop:
            normals = self.crop_square(normals)
        #normals[:, :, 2] *= -1  # Flip Z

        # Normalize [-1, 1] to [0, 65535]
        normals = np.clip((normals + 1.0) * 0.5 * 65535.0, 0, 65535)
        normals = np.nan_to_num(normals).astype(np.uint16)

        normals_bgr = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, normals_bgr)
    
    @staticmethod
    def export_normals_pickle(self,normals, filename):#.pkl
        """Export normals using pickle (preserves exact floating point precision)"""
        with open(filename, 'wb') as f:
            pickle.dump(normals, f)
        print(f"Normals exported to {filename}")
    
    def process(self,images_arr_raw : list ,mask = None): 
        ''' process creates the normal map for the given raw images '''
        # apply mask
        #todo
        if (mask is not None):
            self.mask = mask
            for id in range(0, self.image_count):
                images_arr_raw[id] = np.multiply(images_arr_raw[id], mask.astype(np.float32))
                
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
        self.save_normals_to_16bit_png("normal_mapping.png",normals=normals_float)  
        self.export_normals_pickle(self,normals_float,'normals.pkl')
    def process_color(self,images_arr_raw : list ,mask = None): 
        ''' process creates the normal map for the given raw images '''
        c = 3 #channels of color
        if (mask is not None):
            self.mask = mask
            for id in range(0, self.image_count):
                if images_arr_raw[id].ndim == c: #aka is a color image
                    for chnl in range(c): #iterating through 3 colors           
                        images_arr_raw[id][:,:,chnl] = np.multiply(images_arr_raw[id][:,:,chnl], mask.astype(np.float32)) #masking each channel 
                else:
                    print("not a color image, defaulting to greyscale")
                    images_arr_raw[id] = np.multiply(images_arr_raw[id], mask.astype(np.float32))
                    
        image_arr = [] # create 3 matrices each should have 
        is_color = images_arr_raw[0].ndim == c
        intensity_colour_channels = [[] for _ in range(c)]# contains the intensities of every channel 3 of them 
        if is_color:
            h, w, c = images_arr_raw[0].shape # setting height and width of the normal space and the channels        
            
            for image in images_arr_raw: #image array is a list of uint8 cause cv does that at imread
                #we normalise the values and convert them to a float
                image = image.astype(np.float32) #conversion
                image = image/255 # normalising
                image_arr.append(image)
                for channel in range(c):
                    image_channel = image[:,:,channel].reshape(-1)#reshape to a flat vector [1,h*w] image for each channel
                    intensity_colour_channels[channel].append(np.array(image_channel))
                 # add to intensities as each is a vector of pixel intensities [12 ,h*w]
            intensity_colour_channels = [np.array(ch_intensities) for ch_intensities in intensity_colour_channels]
               
            avg_intensity = np.mean(np.array(intensity_colour_channels), axis=0)  # shape: [12, h*w]
            self.intensity = avg_intensity
  
            #h,w = image_arr[0].shape[:2] # setting height and width of the normal space 
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
            # Get RGB albedo per channel based on projected intensity
            self.albedo = np.zeros((h, w, c), dtype=np.float32)
            for channel in range(c):
                # Element-wise multiply unit normal with light matrix and projected intensity
                ch_intensity = intensity_colour_channels[channel]  # shape: (n_lights, h*w)
                scaled_normal_ch = (pinv_source_mat @ ch_intensity).T.reshape((h, w, 3))
                albedo_ch = np.linalg.norm(scaled_normal_ch, axis=2)
                self.albedo[:, :, channel] = albedo_ch


            epsilon = 1e-8
            albedo_safe = np.maximum(self.albedo, epsilon)  # albedo_safe: (h, w) - albedo with minimum threshold

            
            # Compute unit normals
            albedo_magnitude = np.linalg.norm(scaled_normals, axis=2)
            epsilon = 1e-8
            albedo_safe = np.maximum(albedo_magnitude, epsilon)
            
            self.normals = scaled_normals / albedo_safe[:, :, np.newaxis]
            nz_safe = np.maximum(np.abs(self.normals[:, :, 2]), epsilon)
            
            self.p = -(self.normals[:,:,0]) / nz_safe
            self.q = -(self.normals[:,:,1]) / nz_safe
            
            
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
            self.save_normals_to_16bit_png("16bitnormalmap.png",normals_float)
            self.export_normals_pickle(self,normals_float,'normals.pkl')
        else:
            self.process(images_arr_raw, mask)
       
    def frankot_chellappa(self,p, q):
        """
        Reconstruct surface from gradients p and q using Frankot-Chellappa algorithm.
        """
        h, w = p.shape
        wx = np.fft.fftfreq(w) * 2 * np.pi
        wy = np.fft.fftfreq(h) * 2 * np.pi
        wx, wy = np.meshgrid(wx, wy)

        denom = wx**2 + wy**2
        denom[0, 0] = 1  # avoid division by zero at DC component

        # Fourier transforms
        p_fft = fft2(p)
        q_fft = fft2(q)

        # Surface reconstruction in Fourier domain
        z_fft = (-1j * wx * p_fft - 1j * wy * q_fft) / denom
        z_fft[0, 0] = 0  # remove DC offset

        # Inverse FFT to get height map
        z = np.real(ifft2(z_fft))
        return z
    
    def poisson_solver(self,p: np.ndarray, q: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct depth map Z from gradient fields p and q using FFT-based Poisson solver.

        Args:
            p (np.ndarray): ∂z/∂x gradient (shape HxW)
            q (np.ndarray): ∂z/∂y gradient (shape HxW)
            mask (np.ndarray, optional): binary mask to define region of interest (shape HxW)

        Returns:
            Z (np.ndarray): reconstructed depth map (shape HxW)
        """
        h, w = p.shape

        # Compute divergence of the gradient field (∂p/∂x + ∂q/∂y)
        fx = np.zeros_like(p)
        fy = np.zeros_like(q)
        fx[:, :-1] = p[:, 1:] - p[:, :-1]
        fy[:-1, :] = q[1:, :] - q[:-1, :]
        f = fx + fy

        # Discrete frequencies
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        denom = (2 * np.cos(np.pi * x / w) - 2) + (2 * np.cos(np.pi * y / h) - 2)
        denom[0, 0] = 1  # Avoid divide by zero at (0, 0)

        # FFT-based Poisson solver
        f_hat = fft2(f)
        Z_hat = f_hat / denom
        Z_hat[0, 0] = 0  # Set DC component to 0 to fix scale ambiguity

        Z = np.real(ifft2(Z_hat))

        # Apply mask if provided
        if mask is not None:
            Z = Z * (mask > 0)

        return Z
    
    def plot_depths_3d(self,depth_map:np.ndarray,title:str):
        x, y = np.meshgrid(range(depth_map.shape[1]), range(depth_map.shape[0]))
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        ax_3d.plot_surface(x, y, depth_map, cmap="viridis", edgecolor="none")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_depths_2d(self,depth_map:np.ndarray,title:str):
        plt.figure(figsize=(6, 6))
        plt.imshow(depth_map, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_normal_map(self):
        """
        Visualize the normal map components and albedo for grayscale images.
        """
        if self.normals is None or len(self.normals) == 0:
            print("No normals computed yet. Run process() first.")
            return
        self.normals[:, :, 2] *= -1  # if needed reverses the Z axis only to turn depth into height
        if self.albedo is None:
            print("No albedo map found.")
            return

        # Convert normals from [-1, 1] to [0, 1] for visualization
        normals_vis = (self.normals + 1.0) / 2.0  # shape: (H, W, 3)

        # Create a 2x2 grid to show the normal components
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Combined normal map (RGB)
        axes[0, 0].imshow(normals_vis)
        axes[0, 0].set_title('Normal Map (RGB Encoding)')
        axes[0, 0].axis('off')

        # X component
        im1 = axes[0, 1].imshow(self.normals[:, :, 0], cmap='gray', vmin=-1, vmax=1)
        axes[0, 1].set_title('Normal X Component')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        # Y component
        im2 = axes[1, 0].imshow(self.normals[:, :, 1], cmap='gray', vmin=-1, vmax=1)
        axes[1, 0].set_title('Normal Y Component')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        # Z component
        im3 = axes[1, 1].imshow(self.normals[:, :, 2], cmap='gray', vmin=-1, vmax=1)
        axes[1, 1].set_title('Normal Z Component')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        plt.tight_layout()
        plt.show()

        # Show albedo (grayscale reflectance)
        plt.figure(figsize=(6, 6))
        plt.imshow(self.albedo, cmap='gray')
        plt.title('Albedo (Grayscale Reflectance)')
        plt.axis('off')
        plt.show()
        
                        
        # 4. Histogram of Z components of normal
        fig = plt.figure(figsize=(6, 4))
        plt.hist(self.normals[:, :, 2].flatten(), bins=100, color='skyblue', edgecolor='black')
        plt.title("Distribution of Normal Z Components")
        plt.xlabel("Z value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

    def plot_color_albedo(self, white_balance=True, gamma_correct=True):
        """Visualize color albedo map with optional white balancing and gamma correction"""
        if self.albedo is None:
            print("No albedo computed yet.")
            return

        if self.albedo.ndim == 3:  # Color albedo
            plt.figure(figsize=(15, 5))
            channels = ['Red', 'Green', 'Blue']

            for i in range(3):
                channel = self.albedo[:, :, i]
                norm_channel = np.clip(channel / (np.max(channel) + 1e-8), 0, 1)
                plt.subplot(1, 4, i+1)
                plt.imshow(norm_channel, cmap='gray')
                plt.title(f'{channels[i]} Channel Albedo')
                plt.axis('off')

            # Combine color albedo without per-pixel normalization
            albedo_display = np.copy(self.albedo)

            # Optional: white balance based on average intensity per channel
            if white_balance:
                channel_means = np.mean(albedo_display, axis=(0, 1))
                scale_factors = channel_means.mean() / (channel_means + 1e-8)
                albedo_display *= scale_factors

            # Normalize for display
            albedo_display /= (np.max(albedo_display) + 1e-8)
            albedo_display = np.clip(albedo_display, 0, 1)

            # Optional: gamma correction (for display)
            if gamma_correct:
                gamma = .3
                albedo_display = np.power(albedo_display, 1 / gamma)

            plt.subplot(1, 4, 4)
            plt.imshow(albedo_display)
            plt.title('Combined Color Albedo')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        else:
            # Grayscale albedo
            plt.figure(figsize=(6, 6))
            albedo_vis = np.clip(self.albedo / (np.max(self.albedo) + 1e-8), 0, 1)
            plt.imshow(albedo_vis, cmap='gray')
            plt.title('Grayscale Albedo')
            plt.axis('off')
            plt.show()


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
        
        #normal_map = ((normals / 2.0) + 0.5) * 255
        normalized_normals = (normals)  

        
        heights = height_map.estimate_height_map(normal_map = normalized_normals, raw_values=True,normalized_input= True ,mask=self.mask)
        heights = - heights
        # height_mapz = self.frankot_chellappa(self.p, self.q)
        # height_mapz2 = self.poisson_solver(self.p,self.q)
        
        #2D plots
        self.plot_depths_2d(heights,"basic 2D height-map")
        # self.plot_depths_2d(height_mapz,"2D height-map frankott")
        # self.plot_depths_2d(height_mapz2,"basic 2D height-map poisson")
        

        #3D plots
        self.plot_depths_3d(heights,"basic 3D height-map")
        # self.plot_depths_3d(height_mapz,"3D height-map frankott")
        # self.plot_depths_3d(height_mapz2,"basic 3D height-map poisson")
 


        

        

        
        
        