import cv2 as cv
import numpy as np
import vtk

class new_photo:
    
    def __init__(self, image_count: int, light_matrix: list):
        self.image_count = image_count
        self.p = []
        self.q = []
        self.albedo = []  # rho
        self.mask = []
        self.z = []
        self.pi = np.pi
        self.light_matrix = np.array(light_matrix)  # Convert to numpy array
        self.normals = []  # N = (SS^t)^-1 S^t I
        self.intensity = []  # I
        self.source_mat = []  # S
    
    def process(self, images_arr_raw: list):
        """
        Process images for photometric stereo reconstruction
        """
        # Convert images to float and normalize
        image_arr = []  # List to store processed images
        intensity_matrix = []  # List to store flattened pixel intensities
        
        for image in images_arr_raw:  # image: (h, w) or (h, w, 1)
            # Convert to float and normalize
            image = image.astype(np.float32) / 255.0  # image: (h, w) - normalized [0,1]
            image_arr.append(image)
            
            # Flatten image for matrix operations
            image_flat = image.reshape(-1)  # image_flat: (h*w,) - 1D vector of all pixels
            intensity_matrix.append(image_flat)
        
        # Stack intensities into matrix form
        self.intensity = np.array(intensity_matrix)  # intensity: (n_images, h*w) - each row is one flattened image
        
        # Get image dimensions
        h, w = image_arr[0].shape[:2]  # h: height, w: width
        n_pixels = h * w  # Total number of pixels
        n_images = len(images_arr_raw)  # Number of input images
        
        # Initialize output arrays
        self.normals = np.zeros((h, w, 3), dtype=np.float32)  # normals: (h, w, 3) - unit normal vector at each pixel
        self.p = np.zeros((h, w), dtype=np.float32)  # p: (h, w) - surface gradient in x direction
        self.q = np.zeros((h, w), dtype=np.float32)  # q: (h, w) - surface gradient in y direction  
        self.albedo = np.zeros((h, w), dtype=np.float32)  # albedo: (h, w) - surface reflectance at each pixel
        
        # Light source matrix S
        self.source_mat = np.array(self.light_matrix)  # source_mat: (n_images, 3) - light directions [Lx, Ly, Lz] for each image
        
        # CRITICAL: Proper pseudo-inverse calculation
        # For photometric stereo: g = (S^T S)^-1 S^T I = pinv(S) * I
        # where g contains the surface normals scaled by albedo
        pinv_source_mat = np.linalg.pinv(self.source_mat)  # pinv_source_mat: (3, n_images) - pseudo-inverse of light matrix
        
        # Solve for scaled normals: g = pinv(S) * I
        # Matrix multiplication: (3, n_images) @ (n_images, h*w) = (3, h*w)
        scaled_normals_flat = pinv_source_mat @ self.intensity  # scaled_normals_flat: (3, h*w) - scaled normals for all pixels
        
        # Reshape back to image dimensions
        # Transpose to get (h*w, 3) then reshape to (h, w, 3)
        scaled_normals = scaled_normals_flat.T.reshape((h, w, 3))  # scaled_normals: (h, w, 3) - scaled normal vectors
        
        # Compute albedo (rho) = ||scaled_normal|| for each pixel
        # Take L2 norm along the last axis (3D vector at each pixel)
        self.albedo = np.linalg.norm(scaled_normals, axis=2)  # albedo: (h, w) - magnitude of scaled normals
        
        # Compute unit normals by normalizing scaled normals
        # Avoid division by zero
        epsilon = 1e-8
        albedo_safe = np.maximum(self.albedo, epsilon)  # albedo_safe: (h, w) - albedo with minimum threshold
        
        # Normalize to get unit normals: N = g / ||g||
        for i in range(3):
            self.normals[:, :, i] = scaled_normals[:, :, i] / albedo_safe  # normals[:,:,i]: (h, w) - i-th component of unit normals
        
        # Compute gradient components p and q from surface normals
        # p = -dZ/dx = -Nx/Nz, q = -dZ/dy = -Ny/Nz
        nz_safe = np.maximum(np.abs(self.normals[:, :, 2]), epsilon)  # nz_safe: (h, w) - safe z-component of normals
        self.p = -self.normals[:, :, 0] / nz_safe  # p: (h, w) - surface gradient in x direction
        self.q = -self.normals[:, :, 1] / nz_safe  # q: (h, w) - surface gradient in y direction
        
        # Handle points where Nz is close to zero (surface nearly perpendicular to viewing direction)
        perpendicular_mask = np.abs(self.normals[:, :, 2]) < epsilon  # perpendicular_mask: (h, w) - boolean mask
        self.p[perpendicular_mask] = 0  # Set gradients to zero for perpendicular surfaces
        self.q[perpendicular_mask] = 0
        
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
        
    def integrate_surface(self, method='frankot_chellappa'):
        """
        Integrate surface gradients to recover depth map
        """
        if method == 'frankot_chellappa':
            return self._frankot_chellappa_integration()
        else:
            return self._simple_integration()
    
    def _frankot_chellappa_integration(self):
        """
        Frankot-Chellappa integration method using FFT
        """
        h, w = self.p.shape  # p.shape, q.shape: (h, w)
        
        # Create frequency grids
        u = np.fft.fftfreq(w).reshape(1, -1)  # u: (1, w) - horizontal frequency grid
        v = np.fft.fftfreq(h).reshape(-1, 1)  # v: (h, 1) - vertical frequency grid
        
        # Take FFT of gradients
        P_fft = np.fft.fft2(self.p)  # P_fft: (h, w) - FFT of p gradients (complex)
        Q_fft = np.fft.fft2(self.q)  # Q_fft: (h, w) - FFT of q gradients (complex)
        
        # Avoid division by zero
        denominator = u**2 + v**2  # denominator: (h, w) - frequency domain denominator
        denominator[0, 0] = 1  # Handle DC component to avoid division by zero
        
        # Integrate in frequency domain
        # Z_fft = (-1j * u * P_fft - 1j * v * Q_fft) / denominator
        Z_fft = (-1j * u * P_fft - 1j * v * Q_fft) / denominator  # Z_fft: (h, w) - integrated depth in frequency domain (complex)
        Z_fft[0, 0] = 0  # Set DC component to zero (removes arbitrary offset)
        
        # Transform back to spatial domain
        self.z = np.real(np.fft.ifft2(Z_fft))  # z: (h, w) - recovered depth map (real values)
        
        return self.z
    
    def _simple_integration(self):
        """
        Simple path integration (less robust but faster)
        """
        h, w = self.p.shape  # p.shape, q.shape: (h, w)
        self.z = np.zeros((h, w), dtype=np.float32)  # z: (h, w) - depth map initialized to zeros
        
        # Integrate along first row using p gradients
        for j in range(1, w):
            # z[0,j] = z[0,j-1] - p[0,j-1] (cumulative sum of x-gradients)
            self.z[0, j] = self.z[0, j-1] - self.p[0, j-1]  # z[0,j]: scalar - depth at position (0,j)
        
        # Integrate along columns using q gradients
        for i in range(1, h):
            for j in range(w):
                # z[i,j] = z[i-1,j] - q[i-1,j] (cumulative sum of y-gradients)
                self.z[i, j] = self.z[i-1, j] - self.q[i-1, j]  # z[i,j]: scalar - depth at position (i,j)
        
        return self.z  # z: (h, w) - final depth map

# Key Issues Fixed:
# 1. Matrix dimensions and operations corrected
# 2. Proper pseudo-inverse calculation
# 3. Albedo computation as magnitude of scaled normals
# 4. Unit normal computation by dividing by albedo
# 5. Gradient computation p = -Nx/Nz, q = -Ny/Nz
# 6. Added surface integration methods
# 7. Proper handling of edge cases (division by zero)
# 8. Added debugging output