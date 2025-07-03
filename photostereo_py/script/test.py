import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import OpenEXR
import Imath
import array
import os
from numba import jit, njit, prange
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import torch

# Numba-accelerated functions for critical computations
@njit(parallel=True)
def fast_template_matching_batch(templates, target_image, positions, template_size):
    """Fast batch template matching using Numba"""
    n_templates = len(templates)
    h, w = template_size
    target_h, target_w = target_image.shape
    
    scores = np.zeros(n_templates, dtype=np.float32)
    best_positions = np.zeros((n_templates, 2), dtype=np.int32)
    
    for i in prange(n_templates):
        template = templates[i]
        best_score = -1.0
        best_x, best_y = 0, 0
        
        # Template matching with normalized cross-correlation
        for y in range(0, target_h - h + 1, 2):  # Step by 2 for speed
            for x in range(0, target_w - w + 1, 2):
                # Extract region
                region = target_image[y:y+h, x:x+w]
                
                # Normalized cross-correlation
                template_mean = np.mean(template)
                region_mean = np.mean(region)
                
                template_std = np.std(template)
                region_std = np.std(region)
                
                if template_std > 0 and region_std > 0:
                    correlation = np.mean((template - template_mean) * (region - region_mean))
                    correlation /= (template_std * region_std)
                    
                    if correlation > best_score:
                        best_score = correlation
                        best_x, best_y = x, y
        
        scores[i] = best_score
        best_positions[i, 0] = best_x
        best_positions[i, 1] = best_y
    
    return scores, best_positions

@njit(parallel=True)
def fast_mask_validation(positions, mask, template_size, mask_threshold):
    """Fast mask validation using Numba"""
    n_positions = len(positions)
    valid_flags = np.zeros(n_positions, dtype=np.bool_)
    h, w = template_size
    
    for i in prange(n_positions):
        x, y = positions[i]
        
        # Check bounds
        if y + h > mask.shape[0] or x + w > mask.shape[1]:
            valid_flags[i] = False
            continue
        
        # Count valid pixels in mask region
        valid_count = 0
        total_count = h * w
        
        for dy in range(h):
            for dx in range(w):
                if not mask[y + dy, x + dx]:
                    valid_count += 1
        
        valid_ratio = valid_count / total_count
        valid_flags[i] = valid_ratio >= mask_threshold
    
    return valid_flags


def fast_difference_detection_torch(img1, img2, threshold=0.1):
    # img1, img2: (H, W) torch tensors on CUDA
    diff = torch.abs(img1 - img2)
    diff_map = torch.where(diff > threshold, diff, torch.zeros_like(diff))
    return diff_map


class OptimizedNormalMapDetector:
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or mp.cpu_count()
        print(f"Using {self.n_workers} worker processes")
    
    def load_exr_image(self, filename):
        """Optimized EXR loader with caching"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"EXR file not found: {filename}")
        
        file_size = os.path.getsize(filename)
        if file_size == 0:
            raise ValueError(f"EXR file is empty: {filename}")
        
        print(f"Loading EXR file: {filename} ({file_size} bytes)")
        
        start_time = time.time()
        
        exr_file = OpenEXR.InputFile(filename)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        channels = list(header['channels'].keys())
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Read channels in parallel if possible
        channel_data = {}
        for channel in ['R', 'G', 'B']:
            if channel in channels:
                channel_str = exr_file.channel(channel, FLOAT)
                channel_array = array.array('f', channel_str)
                channel_data[channel] = np.array(channel_array, dtype=np.float32).reshape(height, width)
        
        if 'R' in channel_data and 'G' in channel_data and 'B' in channel_data:
            img = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=2)
        elif 'R' in channel_data:
            img = channel_data['R']
        else:
            first_channel = channels[0]
            channel_str = exr_file.channel(first_channel, FLOAT)
            channel_array = array.array('f', channel_str)
            img = np.array(channel_array, dtype=np.float32).reshape(height, width)
        
        exr_file.close()
        
        # Fast normalization
        img = np.clip(img, 0, 1)
        
        load_time = time.time() - start_time
        print(f"Loaded in {load_time:.2f}s - Shape: {img.shape}")
        
        return img
    
    def create_fast_mask(self, image, method='auto', threshold=0.1):
        """Fast mask creation using vectorized operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        if method == 'auto':
            # Vectorized mask creation
            mask_dark = gray < (threshold * 255)
            mask_bright = gray > ((1 - threshold) * 255)
            mask = mask_dark | mask_bright
        else:
            mask = gray < (threshold * 255)
        
        return mask
    
    def extract_templates_vectorized(self, image, template_size=(64, 64), overlap=0.5):
        """Vectorized template extraction"""
        step_y = int(template_size[0] * (1 - overlap))
        step_x = int(template_size[1] * (1 - overlap))
        
        positions = []
        for y in range(0, image.shape[0] - template_size[0] + 1, step_y):
            for x in range(0, image.shape[1] - template_size[1] + 1, step_x):
                positions.append((x, y))
        
        # Pre-allocate template array
        n_templates = len(positions)
        templates = np.zeros((n_templates, template_size[0], template_size[1]), dtype=np.float32)
        
        # Extract templates in batch
        for i, (x, y) in enumerate(positions):
            templates[i] = image[y:y+template_size[0], x:x+template_size[1]]
        
        return templates, np.array(positions)
    
    def detect_differences_fast(self, original_file, modified_file, template_size=(64, 64), 
                               overlap=0.5, mask_method='auto', mask_threshold=0.1):
        """Main optimized difference detection"""
        
        print("=== OPTIMIZED NORMAL MAP DIFFERENCE DETECTION ===")
        total_start = time.time()
        
        # Load images
        print("\n1. Loading images...")
        start_time = time.time()
        original = self.load_exr_image(original_file)
        modified = self.load_exr_image(modified_file)
        print(f"Image loading took: {time.time() - start_time:.2f}s")
        
        # Create mask
        print("\n2. Creating mask...")
        start_time = time.time()
        mask = self.create_fast_mask(original, method=mask_method, threshold=mask_threshold)
        mask_time = time.time() - start_time
        print(f"Mask creation took: {mask_time:.2f}s")
        print(f"Masked pixels: {np.sum(mask)}/{mask.size} ({np.sum(mask)/mask.size*100:.1f}%)")
        
        # Convert to grayscale for processing
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            modified_gray = cv2.cvtColor((modified * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            original_gray = (original * 255).astype(np.float32)
            modified_gray = (modified * 255).astype(np.float32)
        
        # Method 1: Fast pixel-wise difference (for quick overview)
        print("\n3. Computing pixel-wise differences...")
        start_time = time.time()
        diff_map = fast_difference_detection(original_gray, modified_gray, threshold=10.0)
        pixel_diff_time = time.time() - start_time
        print(f"Pixel-wise difference took: {pixel_diff_time:.2f}s")
        
        # Method 2: Template-based detection (more accurate)
        print("\n4. Template-based detection...")
        start_time = time.time()
        
        # Extract templates
        templates, positions = self.extract_templates_vectorized(original_gray, template_size, overlap)
        print(f"Extracted {len(templates)} templates")
        
        # Validate templates against mask
        valid_flags = fast_mask_validation(positions, mask, template_size, 0.8)
        valid_templates = templates[valid_flags]
        valid_positions = positions[valid_flags]
        
        print(f"Valid templates after mask filtering: {len(valid_templates)}")
        
        # Fast template matching using OpenCV (much faster than custom implementation)
        correlation_scores = []
        best_matches = []
        
        for i, template in enumerate(valid_templates):
            # Use OpenCV's optimized template matching
            result = cv2.matchTemplate(modified_gray, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)

            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            correlation_scores.append(max_val)
            best_matches.append((max_loc, max_val))
            
            if i % 50 == 0:
                print(f"Processed {i+1}/{len(valid_templates)} templates")
        
        template_time = time.time() - start_time
        print(f"Template matching took: {template_time:.2f}s")
        
        # Calculate threshold and find differences
        scores_array = np.array(correlation_scores)
        threshold = max(np.mean(scores_array) - 2.0 * np.std(scores_array), 0.8)
        
        differences = []
        for i, (score, pos) in enumerate(zip(correlation_scores, valid_positions)):
            if score < threshold:
                differences.append({
                    'position': tuple(pos),
                    'score': score,
                    'template_index': i,
                    'size': template_size
                })
        
        total_time = time.time() - total_start
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Image loading: {time.time() - total_start - template_time - pixel_diff_time - mask_time:.2f}s")
        print(f"Mask creation: {mask_time:.2f}s")
        print(f"Pixel differences: {pixel_diff_time:.2f}s")
        print(f"Template matching: {template_time:.2f}s")
        print(f"Templates per second: {len(valid_templates)/template_time:.1f}")
        
        # Visualization
        self.visualize_fast_results(original, modified, differences, diff_map, scores_array, threshold, mask)
        
        return differences, correlation_scores, threshold, mask, diff_map
    
    def visualize_fast_results(self, original, modified, differences, diff_map, scores, threshold, mask):
        """Fast visualization with optimized plotting"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Normal Map')
        axes[0, 0].axis('off')
        
        # Modified
        axes[0, 1].imshow(modified)
        axes[0, 1].set_title('Modified Normal Map')
        axes[0, 1].axis('off')
        
        # Pixel-wise difference map
        axes[0, 2].imshow(diff_map, cmap='hot')
        axes[0, 2].set_title('Pixel-wise Differences')
        axes[0, 2].axis('off')
        
        # Template-based differences
        axes[1, 0].imshow(modified)
        for diff in differences:
            rect = Rectangle(diff['position'], diff['size'][1], diff['size'][0],
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[1, 0].add_patch(rect)
        axes[1, 0].set_title(f'Template Differences ({len(differences)})')
        axes[1, 0].axis('off')
        
        # Score distribution
        axes[1, 1].hist(scores, bins=50, alpha=0.7, color='blue')
        axes[1, 1].axvline(threshold, color='red', linestyle='--', 
                          label=f'Threshold: {threshold:.3f}')
        axes[1, 1].set_xlabel('Correlation Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined view
        axes[1, 2].imshow(modified, alpha=0.7)
        axes[1, 2].imshow(diff_map, cmap='Reds', alpha=0.5)
        if mask is not None:
            axes[1, 2].imshow(mask, cmap='Blues', alpha=0.3)
        axes[1, 2].set_title('Combined View')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Check if required libraries are available
    try:
        import numba
        print("✓ Numba available - using JIT compilation")
    except ImportError:
        print("⚠ Numba not available - install with: pip install numba")
        print("Code will still work but will be slower")
    
    # File paths
    original_file = "normal_mapping_stamp.exr"
    modified_file = "normal_mapping_stamp_particle.exr"
    
    # Parameters
    template_size = (32, 32)  # Smaller templates for speed
    overlap = 0.3  # Less overlap for speed
    
    print("=== OPTIMIZED NORMAL MAP DIFFERENCE DETECTION ===")
    print(f"Template size: {template_size}")
    print(f"Overlap: {overlap}")
    print(f"CPU cores: {mp.cpu_count()}")
    
    # Check files exist
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    exr_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.exr')]
    print(f"Found EXR files: {exr_files}")
    
    if not os.path.exists(original_file):
        print(f"ERROR: Original file not found: {original_file}")
        return
    
    if not os.path.exists(modified_file):
        print(f"ERROR: Modified file not found: {modified_file}")
        return
    
    try:
        detector = OptimizedNormalMapDetector()
        differences, scores, threshold, mask, diff_map = detector.detect_differences_fast(
            original_file, modified_file, template_size=template_size, overlap=overlap
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Total differences found: {len(differences)}")
        print(f"Threshold used: {threshold:.3f}")
        print(f"Mean correlation score: {np.mean(scores):.3f}")
        print(f"Std correlation score: {np.std(scores):.3f}")
        print(f"Pixel differences detected: {np.sum(diff_map > 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()