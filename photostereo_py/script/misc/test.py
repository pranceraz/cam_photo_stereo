import torch
print(torch.__version__)
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

from segment_anything import sam_model_registry
print("segment-anything imported successfully")



def process_color(self, images_arr_raw: list, mask=None):
    """Process color images to extract both normal maps and color albedo"""
    
    if mask is not None:
        self.mask = mask
        for id in range(0, self.image_count):
            if images_arr_raw[id].ndim == 3:  # Color image
                for c in range(3):  # Apply mask to each channel
                    images_arr_raw[id][:,:,c] = np.multiply(
                        images_arr_raw[id][:,:,c], mask.astype(np.float32)
                    )
            else:  # Grayscale
                images_arr_raw[id] = np.multiply(images_arr_raw[id], mask.astype(np.float32))
    
    # Check if images are color or grayscale
    is_color = images_arr_raw[0].ndim == 3
    
    if is_color:
        # Process each color channel separately
        h, w, c = images_arr_raw[0].shape
        self.albedo = np.zeros((h, w, c), dtype=np.float32)  # Color albedo
        
        # Initialize intensity matrices for each channel
        intensity_channels = [[] for _ in range(c)]
        
        image_arr = []
        for image in images_arr_raw:
            image = image.astype(np.float32) / 255.0  # Normalize
            image_arr.append(image)
            
            # Store intensities for each channel
            for ch in range(c):
                channel_flat = image[:,:,ch].reshape(-1)
                intensity_channels[ch].append(channel_flat)
        
        # Convert to numpy arrays
        intensity_channels = [np.array(ch_intensities) for ch_intensities in intensity_channels]
        
        # Process normals using one channel (typically use green or average)
        # Using green channel as it's often most representative
        self.intensity = intensity_channels[1]  # Green channel
        
        # Continue with normal processing...
        self.normals = np.zeros((h, w, 3), dtype=np.float32)
        self.p = np.zeros((h, w), dtype=np.float32)
        self.q = np.zeros((h, w), dtype=np.float32)
        self.source_mat = np.array(self.light_matrix)
        pinv_source_mat = np.linalg.pinv(self.source_mat)
        
        # Compute normals from green channel
        scaled_normals_T = pinv_source_mat @ self.intensity
        scaled_normals = scaled_normals_T.T.reshape((h, w, 3))
        
        # Compute color albedo for each channel
        for ch in range(c):
            scaled_normals_ch_T = pinv_source_mat @ intensity_channels[ch]
            scaled_normals_ch = scaled_normals_ch_T.T.reshape((h, w, 3))
            self.albedo[:,:,ch] = np.linalg.norm(scaled_normals_ch, axis=2)
        
        # Compute unit normals
        albedo_magnitude = np.linalg.norm(scaled_normals, axis=2)
        epsilon = 1e-8
        albedo_safe = np.maximum(albedo_magnitude, epsilon)
        
        self.normals = scaled_normals / albedo_safe[:, :, np.newaxis]
        nz_safe = np.maximum(np.abs(self.normals[:, :, 2]), epsilon)
        
        self.p = -(self.normals[:,:,0]) / nz_safe
        self.q = -(self.normals[:,:,1]) / nz_safe
        
    else:
        # Use original grayscale processing
        self.process(images_arr_raw, mask)

# Enhanced visualization for color albedo
def plot_color_albedo(self):
    """Visualize color albedo map"""
    if self.albedo is None:
        print("No albedo computed yet.")
        return
    
    if self.albedo.ndim == 3:  # Color albedo
        plt.figure(figsize=(15, 5))
        
        # Show each channel
        channels = ['Red', 'Green', 'Blue']
        for i in range(3):
            plt.subplot(1, 4, i+1)
            plt.imshow(self.albedo[:,:,i], cmap='gray')
            plt.title(f'{channels[i]} Channel Albedo')
            plt.axis('off')
        
        # Show combined color albedo
        plt.subplot(1, 4, 4)
        # Normalize for display
        albedo_display = np.clip(self.albedo / np.max(self.albedo), 0, 1)
        plt.imshow(albedo_display)
        plt.title('Combined Color Albedo')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Grayscale albedo
        plt.figure(figsize=(6, 6))
        plt.imshow(self.albedo, cmap='gray')
        plt.title('Grayscale Albedo')
        plt.axis('off')
        plt.show()