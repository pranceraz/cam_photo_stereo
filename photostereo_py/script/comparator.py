import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import registration, morphology, measure, filters
from scipy import ndimage
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import OpenEXR
    import Imath
    EXR_AVAILABLE = True
except ImportError:
    print("Warning: OpenEXR not available. Install with: pip install OpenEXR")
    EXR_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    print("Warning: imageio not available. Install with: pip install imageio")
    IMAGEIO_AVAILABLE = False

class NormalMapComparator:
    def __init__(self, angular_threshold=3.0, min_defect_size=10, registration_enabled=True):
        """
        Advanced Normal Map Comparison Tool
        
        Parameters:
        -----------
        angular_threshold : float
            Threshold in degrees for defect detection
        min_defect_size : int
            Minimum size of defects to consider (in pixels)
        registration_enabled : bool
            Whether to perform image registration for alignment
        """
        self.angular_threshold = angular_threshold
        self.min_defect_size = min_defect_size
        self.registration_enabled = registration_enabled
        self.results = {}
    
    def load_exr_image(self, exr_path):
        """
        Load EXR file and extract RGB channels
        Supports both OpenEXR and imageio backends
        """
        exr_path = str(exr_path)
        
        if EXR_AVAILABLE:
            try:
                # Try OpenEXR first (more robust for complex EXR files)
                exr_file = OpenEXR.InputFile(exr_path)
                header = exr_file.header()
                
                # Get image dimensions
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Get channel names
                channels = header['channels'].keys()
                print(f"EXR channels available: {list(channels)}")
                
                # Try to find RGB channels (various naming conventions)
                rgb_channels = self._find_rgb_channels(channels)
                
                if not rgb_channels:
                    raise ValueError("Could not find RGB channels in EXR file")
                
                # Read channels
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                
                r_str = exr_file.channel(rgb_channels['R'], pixel_type)
                g_str = exr_file.channel(rgb_channels['G'], pixel_type)  
                b_str = exr_file.channel(rgb_channels['B'], pixel_type)
                
                # Convert to numpy arrays
                r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
                
                # Stack channels
                image = np.stack([r, g, b], axis=2)
                exr_file.close()
                
                print(f"Loaded EXR with OpenEXR: {width}x{height}, range: [{np.min(image):.3f}, {np.max(image):.3f}]")
                return image
                
            except Exception as e:
                print(f"OpenEXR loading failed: {e}, trying imageio...")
        
        if IMAGEIO_AVAILABLE:
            try:
                # Try imageio as fallback
                image = imageio.imread(exr_path, format='EXR-FI')
                if len(image.shape) == 2:
                    # Grayscale, convert to RGB
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] > 3:
                    # Take first 3 channels
                    image = image[:, :, :3]
                
                print(f"Loaded EXR with imageio: {image.shape}, range: [{np.min(image):.3f}, {np.max(image):.3f}]")
                return image.astype(np.float32)
                
            except Exception as e:
                print(f"imageio EXR loading failed: {e}")
        
        raise ValueError(f"Could not load EXR file: {exr_path}. Install OpenEXR or imageio with EXR support.")
    
    def _find_rgb_channels(self, channels):
        """Find RGB channel names in EXR file (handles various naming conventions)"""
        channels = list(channels)
        rgb_mapping = {}
        
        # Common naming patterns for RGB channels
        rgb_patterns = {
            'R': ['R', 'r', 'Red', 'red', 'X', 'x'],
            'G': ['G', 'g', 'Green', 'green', 'Y', 'y'], 
            'B': ['B', 'b', 'Blue', 'blue', 'Z', 'z']
        }
        
        for color, patterns in rgb_patterns.items():
            for pattern in patterns:
                if pattern in channels:
                    rgb_mapping[color] = pattern
                    break
            
            # Also check for layered channels like "diffuse.R", "beauty.R", etc.
            if color not in rgb_mapping:
                for channel in channels:
                    if any(channel.endswith('.' + p) or channel.endswith('_' + p) for p in patterns):
                        rgb_mapping[color] = channel
                        break
        
        # Return mapping if we found all three channels
        if len(rgb_mapping) == 3:
            return rgb_mapping
        return None
    
    def normalize_exr_data(self, image):
        """
        Normalize EXR data to appropriate range for normal map processing
        EXR normals might be in [-1,1] range already or need conversion
        """
        # Check if data is already in normal map range [-1, 1]
        if np.min(image) >= -1.1 and np.max(image) <= 1.1:
            print("EXR appears to contain pre-normalized normal data")
            # Clamp to exact range and convert to [0,1] for processing
            image_clamped = np.clip(image, -1.0, 1.0)
            image_normalized = (image_clamped + 1.0) / 2.0
        else:
            print(f"EXR data range: [{np.min(image):.3f}, {np.max(image):.3f}]")
            # Normalize to [0,1] range
            image_min = np.min(image)
            image_max = np.max(image)
            if image_max > image_min:
                image_normalized = (image - image_min) / (image_max - image_min)
            else:
                image_normalized = np.zeros_like(image)
        
        return image_normalized
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess normal map image (supports PNG, JPG, TIF, EXR)"""
        image_path = Path(image_path)
        
        if image_path.suffix.lower() in ['.exr']:
            # Load EXR file
            image = self.load_exr_image(image_path)
            
            # Normalize EXR data
            image = self.normalize_exr_data(image)
            
            # Convert to 8-bit for consistency with rest of pipeline
            # (Note: This loses precision but maintains compatibility)
            image = (image * 255.0).astype(np.uint8)
            
        else:
            # Load standard image formats
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def register_images(self, reference, target):
        """
        Register target image to reference using phase cross-correlation
        Returns registered image, shift vector, and registration error
        """
        try:
            # Convert to grayscale for registration
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian filter to reduce noise
            ref_gray = filters.gaussian(ref_gray, sigma=1.0)
            target_gray = filters.gaussian(target_gray, sigma=1.0)
            
            # Phase cross correlation with sub-pixel accuracy
            shift, error, diffphase = registration.phase_cross_correlation(
                ref_gray, target_gray, upsample_factor=100
            )
            
            # Apply translation to all channels
            registered = np.zeros_like(target)
            for i in range(3):
                registered[:,:,i] = ndimage.shift(target[:,:,i], shift, order=1, cval=0)
            
            return registered.astype(np.uint8), shift, error
            
        except Exception as e:
            print(f"Registration failed: {e}. Using original image.")
            return target, (0, 0), float('inf')
    
    def rgb_to_normal_vectors(self, rgb_image):
        """
        Convert RGB normal map to unit normal vectors
        Standard encoding: RGB [0,255] -> XYZ [-1,1]
        """
        # Normalize to [0,1]
        rgb_float = rgb_image.astype(np.float32) / 255.0
        
        # Convert to normal space [-1,1]
        normals = rgb_float * 2.0 - 1.0
        
        # Ensure unit length (handle potential encoding errors)
        magnitude = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        magnitude = np.maximum(magnitude, 1e-8)  # Avoid division by zero
        normals = normals / magnitude
        
        return normals
    
    def calculate_angular_difference(self, normals1, normals2):
        """Calculate angular difference between normal vectors in degrees"""
        # Dot product for each pixel
        dot_product = np.sum(normals1 * normals2, axis=2)
        
        # Clamp to valid range to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def process_defects(self, angular_diff):
        """
        Process angular differences to identify and characterize defects
        """
        # Create initial defect mask
        defect_mask = angular_diff > self.angular_threshold
        
        # Morphological operations to clean up the mask
        # Remove small noise objects
        defect_mask_clean = morphology.remove_small_objects(
            defect_mask, min_size=self.min_defect_size
        )
        
        # Fill small holes within defects
        defect_mask_clean = morphology.remove_small_holes(
            defect_mask_clean, area_threshold=self.min_defect_size//2
        )
        
        # Apply closing operation to connect nearby defects
        selem = morphology.disk(2)
        defect_mask_clean = morphology.closing(defect_mask_clean, selem)
        
        # Label connected components
        labeled_defects = measure.label(defect_mask_clean)
        defect_properties = measure.regionprops(labeled_defects)
        
        # Analyze each defect
        defect_info = []
        for prop in defect_properties:
            # Get defect region
            defect_region = labeled_defects == prop.label
            defect_errors = angular_diff[defect_region]
            
            # Calculate defect statistics
            defect_data = {
                'label': prop.label,
                'area': prop.area,
                'centroid': prop.centroid,
                'bbox': prop.bbox,  # (min_row, min_col, max_row, max_col)
                'max_error': np.max(defect_errors),
                'avg_error': np.mean(defect_errors),
                'std_error': np.std(defect_errors),
                'severity': self.calculate_defect_severity(prop.area, np.max(defect_errors))
            }
            defect_info.append(defect_data)
        
        # Sort by severity (combination of size and error magnitude)
        defect_info.sort(key=lambda x: x['severity'], reverse=True)
        
        return defect_mask_clean, labeled_defects, defect_info
    
    def calculate_defect_severity(self, area, max_error):
        """Calculate defect severity score based on area and maximum error"""
        # Normalize area (assuming typical defects are 10-1000 pixels)
        area_score = min(area / 100.0, 10.0)
        
        # Normalize error (assuming typical max errors are 1-45 degrees)
        error_score = max_error / 45.0
        
        # Combined severity score
        severity = area_score * error_score * 100
        return severity
    
    def generate_statistics(self, angular_diff, defect_info, registration_shift, registration_error):
        """Generate comprehensive statistics"""
        total_pixels = angular_diff.size
        defect_pixels = sum([d['area'] for d in defect_info])
        
        stats = {
            'registration': {
                'shift_x': float(registration_shift[1]),
                'shift_y': float(registration_shift[0]),
                'registration_error': float(registration_error)
            },
            'angular_analysis': {
                'mean_error_deg': float(np.mean(angular_diff)),
                'median_error_deg': float(np.median(angular_diff)),
                'max_error_deg': float(np.max(angular_diff)),
                'std_error_deg': float(np.std(angular_diff)),
                'percentile_95_deg': float(np.percentile(angular_diff, 95)),
                'percentile_99_deg': float(np.percentile(angular_diff, 99))
            },
            'defect_analysis': {
                'total_defects': len(defect_info),
                'defect_coverage_percent': float((defect_pixels / total_pixels) * 100),
                'largest_defect_area': int(max([d['area'] for d in defect_info])) if defect_info else 0,
                'average_defect_area': float(np.mean([d['area'] for d in defect_info])) if defect_info else 0,
                'most_severe_error_deg': float(max([d['max_error'] for d in defect_info])) if defect_info else 0
            },
            'quality_metrics': {
                'threshold_used_deg': self.angular_threshold,
                'pixels_above_threshold': int(np.sum(angular_diff > self.angular_threshold)),
                'quality_score': self.calculate_quality_score(angular_diff, defect_info)
            }
        }
        
        return stats
    
    def calculate_quality_score(self, angular_diff, defect_info):
        """Calculate overall quality score (0-100, higher is better)"""
        # Base score starts at 100
        quality_score = 100.0
        
        # Penalize based on mean error
        mean_error = np.mean(angular_diff)
        quality_score -= min(mean_error * 5, 50)  # Max 50 point penalty
        
        # Penalize based on defect coverage
        if defect_info:
            defect_coverage = sum([d['area'] for d in defect_info]) / angular_diff.size
            quality_score -= min(defect_coverage * 100 * 10, 40)  # Max 40 point penalty
        
        # Penalize based on maximum error
        max_error = np.max(angular_diff)
        quality_score -= min((max_error - self.angular_threshold) * 2, 30)  # Max 30 point penalty
        
        return max(quality_score, 0.0)
    
    def create_visualization(self, normal1, normal2, angular_diff, defect_mask, 
                           labeled_defects, defect_info, stats):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Advanced Normal Map Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Row 1: Original images and angular difference
        axes[0,0].imshow(normal1)
        axes[0,0].set_title('Reference Normal Map', fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(normal2)
        shift = stats['registration']['shift_x'], stats['registration']['shift_y']
        axes[0,1].set_title(f'Target Normal Map\n(Shift: {shift[0]:.1f}, {shift[1]:.1f})', fontweight='bold')
        axes[0,1].axis('off')
        
        im1 = axes[0,2].imshow(angular_diff, cmap='hot', vmin=0, vmax=20)
        axes[0,2].set_title(f'Angular Difference\n(Max: {stats["angular_analysis"]["max_error_deg"]:.1f}°)', fontweight='bold')
        axes[0,2].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0,2], fraction=0.046, pad=0.04)
        cbar1.set_label('Degrees', rotation=270, labelpad=15)
        
        # Row 2: Defect analysis
        axes[1,0].imshow(defect_mask, cmap='Reds', alpha=0.8)
        axes[1,0].set_title(f'Defect Mask\n(>{self.angular_threshold}° threshold)', fontweight='bold')
        axes[1,0].axis('off')
        
        # Labeled defects with annotations
        im2 = axes[1,1].imshow(labeled_defects, cmap='nipy_spectral', alpha=0.8)
        axes[1,1].imshow(normal1, alpha=0.3)  # Overlay reference for context
        
        # Annotate top 5 defects
        for i, defect in enumerate(defect_info[:5]):
            y, x = defect['centroid']
            axes[1,1].plot(x, y, 'r*', markersize=12, markeredgecolor='white', markeredgewidth=1)
            axes[1,1].text(x+5, y-5, f'{i+1}', color='white', fontweight='bold', 
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        axes[1,1].set_title(f'Labeled Defects\n({len(defect_info)} regions found)', fontweight='bold')
        axes[1,1].axis('off')
        
        # Defect severity visualization
        if defect_info:
            severities = [d['severity'] for d in defect_info[:10]]  # Top 10
            defect_labels = [f"D{i+1}" for i in range(len(severities))]
            bars = axes[1,2].bar(defect_labels, severities, color='red', alpha=0.7)
            axes[1,2].set_title('Top Defects by Severity', fontweight='bold')
            axes[1,2].set_ylabel('Severity Score')
            axes[1,2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, sev in zip(bars, severities):
                height = bar.get_height()
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{sev:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1,2].text(0.5, 0.5, 'No defects found', ha='center', va='center',
                          transform=axes[1,2].transAxes, fontsize=14, fontweight='bold')
            axes[1,2].set_title('Defect Severity Analysis', fontweight='bold')
        
        # Row 3: Statistical analysis
        # Error distribution histogram
        axes[2,0].hist(angular_diff.flatten(), bins=50, alpha=0.7, density=True, color='blue')
        axes[2,0].axvline(self.angular_threshold, color='red', linestyle='--', linewidth=2,
                         label=f'Threshold: {self.angular_threshold}°')
        axes[2,0].axvline(stats['angular_analysis']['mean_error_deg'], color='green', 
                         linestyle='-', linewidth=2, label=f'Mean: {stats["angular_analysis"]["mean_error_deg"]:.2f}°')
        axes[2,0].set_xlabel('Angular Error (degrees)')
        axes[2,0].set_ylabel('Density')
        axes[2,0].set_title('Error Distribution', fontweight='bold')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # Defect size distribution
        if defect_info:
            areas = [d['area'] for d in defect_info]
            axes[2,1].hist(areas, bins=min(20, len(areas)), alpha=0.7, color='orange')
            axes[2,1].set_xlabel('Defect Area (pixels)')
            axes[2,1].set_ylabel('Frequency')
            axes[2,1].set_title('Defect Size Distribution', fontweight='bold')
            axes[2,1].grid(True, alpha=0.3)
        else:
            axes[2,1].text(0.5, 0.5, 'No defects to analyze', ha='center', va='center',
                          transform=axes[2,1].transAxes, fontsize=12)
            axes[2,1].set_title('Defect Size Distribution', fontweight='bold')
        
        # Summary statistics
        summary_text = f"""
REGISTRATION RESULTS:
  Shift (X,Y): ({stats['registration']['shift_x']:.2f}, {stats['registration']['shift_y']:.2f})
  Registration Error: {stats['registration']['registration_error']:.6f}

ANGULAR ANALYSIS:
  Mean Error: {stats['angular_analysis']['mean_error_deg']:.2f}°
  Median Error: {stats['angular_analysis']['median_error_deg']:.2f}°
  Max Error: {stats['angular_analysis']['max_error_deg']:.2f}°
  95th Percentile: {stats['angular_analysis']['percentile_95_deg']:.2f}°

DEFECT ANALYSIS:
  Total Defects: {stats['defect_analysis']['total_defects']}
  Coverage: {stats['defect_analysis']['defect_coverage_percent']:.2f}%
  Largest Defect: {stats['defect_analysis']['largest_defect_area']} pixels
  Most Severe Error: {stats['defect_analysis']['most_severe_error_deg']:.2f}°

QUALITY ASSESSMENT:
  Overall Quality Score: {stats['quality_metrics']['quality_score']:.1f}/100
  Pixels Above Threshold: {stats['quality_metrics']['pixels_above_threshold']:,}
        """
        
        axes[2,2].text(0.05, 0.95, summary_text, transform=axes[2,2].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        axes[2,2].set_xlim(0, 1)
        axes[2,2].set_ylim(0, 1)
        axes[2,2].axis('off')
        axes[2,2].set_title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_dir, filename_prefix="normal_comparison", save_exr=True):
        """Save analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save statistics as JSON
        stats_file = output_path / f"{filename_prefix}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.results['statistics'], f, indent=2)
        
        # Save defect information
        defects_file = output_path / f"{filename_prefix}_defects.json"
        defect_data = {
            'defect_count': len(self.results['defect_info']),
            'defects': []
        }
        
        for i, defect in enumerate(self.results['defect_info']):
            defect_copy = defect.copy()
            defect_copy['centroid'] = [float(defect_copy['centroid'][0]), float(defect_copy['centroid'][1])]
            defect_copy['bbox'] = [int(x) for x in defect_copy['bbox']]
            defect_data['defects'].append(defect_copy)
        
        with open(defects_file, 'w') as f:
            json.dump(defect_data, f, indent=2)
        
        # Save difference map as image (PNG for compatibility)
        diff_image = (self.results['angular_difference'] * 255 / 20).astype(np.uint8)
        diff_file = output_path / f"{filename_prefix}_difference_map.png"
        cv2.imwrite(str(diff_file), diff_image)
        
        # Save defect mask
        defect_image = (self.results['defect_mask'] * 255).astype(np.uint8)
        defect_file = output_path / f"{filename_prefix}_defect_mask.png"
        cv2.imwrite(str(defect_file), defect_image)
        
        # Save high-precision difference map as EXR if requested and available
        if save_exr and (EXR_AVAILABLE or IMAGEIO_AVAILABLE):
            try:
                diff_exr_file = output_path / f"{filename_prefix}_difference_map.exr"
                
                if IMAGEIO_AVAILABLE:
                    # Use imageio to save EXR
                    # Normalize angular difference to [0,1] range for EXR storage
                    diff_normalized = self.results['angular_difference'] / 180.0  # Max 180 degrees
                    imageio.imwrite(str(diff_exr_file), diff_normalized.astype(np.float32), format='EXR-FI')
                    print(f"High-precision difference map saved as EXR: {diff_exr_file}")
                
            except Exception as e:
                print(f"Could not save EXR difference map: {e}")
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def compare(self, reference_path, target_path, show_visualization=True, save_results=False, output_dir="./results"):
        """
        Main comparison function
        
        Parameters:
        -----------
        reference_path : str or Path
            Path to reference normal map
        target_path : str or Path  
            Path to target normal map to compare
        show_visualization : bool
            Whether to display the visualization
        save_results : bool
            Whether to save results to files
        output_dir : str
            Directory to save results
        
        Returns:
        --------
        dict : Complete analysis results
        """
        print("Loading normal maps...")
        normal1 = self.load_and_preprocess(reference_path)
        normal2 = self.load_and_preprocess(target_path)
        
        # Ensure same dimensions
        if normal1.shape != normal2.shape:
            print(f"Resizing target image from {normal2.shape} to {normal1.shape}")
            normal2 = cv2.resize(normal2, (normal1.shape[1], normal1.shape[0]))
        
        # Image registration
        registration_shift = (0, 0)
        registration_error = 0
        if self.registration_enabled:
            print("Performing image registration...")
            normal2, registration_shift, registration_error = self.register_images(normal1, normal2)
            print(f"Registration shift: {registration_shift}, error: {registration_error:.6f}")
        
        # Convert to normal vectors
        print("Converting RGB to normal vectors...")
        normals1 = self.rgb_to_normal_vectors(normal1)
        normals2 = self.rgb_to_normal_vectors(normal2)
        
        # Calculate angular differences
        print("Calculating angular differences...")
        angular_diff = self.calculate_angular_difference(normals1, normals2)
        
        # Process defects
        print("Processing defects...")
        defect_mask, labeled_defects, defect_info = self.process_defects(angular_diff)
        
        # Generate statistics
        print("Generating statistics...")
        stats = self.generate_statistics(angular_diff, defect_info, registration_shift, registration_error)
        
        # Store results
        self.results = {
            'angular_difference': angular_diff,
            'defect_mask': defect_mask,
            'labeled_defects': labeled_defects,
            'defect_info': defect_info,
            'statistics': stats,
            'normal_maps': {
                'reference': normal1,
                'target': normal2
            }
        }
        
        # Create visualization
        if show_visualization:
            print("Creating visualization...")
            fig = self.create_visualization(normal1, normal2, angular_diff, defect_mask, 
                                          labeled_defects, defect_info, stats)
            plt.show()
        
        # Save results if requested
        if save_results:
            self.save_results(output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Quality Score: {stats['quality_metrics']['quality_score']:.1f}/100")
        print(f"Total Defects Found: {stats['defect_analysis']['total_defects']}")
        print(f"Defect Coverage: {stats['defect_analysis']['defect_coverage_percent']:.2f}%")
        print(f"Mean Angular Error: {stats['angular_analysis']['mean_error_deg']:.2f}°")
        print(f"Max Angular Error: {stats['angular_analysis']['max_error_deg']:.2f}°")
        
        if defect_info:
            print(f"\nTop 3 Most Severe Defects:")
            for i, defect in enumerate(defect_info[:3]):
                print(f"  {i+1}. Area: {defect['area']} pixels, Max Error: {defect['max_error']:.2f}°, "
                      f"Severity: {defect['severity']:.1f}")
        
        return self.results

# Usage Example:
def run_comparison_example():
    """Example usage of the NormalMapComparator with EXR support"""
    
    # Check dependencies
    print("Checking EXR support...")
    if EXR_AVAILABLE:
        print("✓ OpenEXR available - full EXR support enabled")
    elif IMAGEIO_AVAILABLE:
        print("✓ imageio available - basic EXR support enabled")  
    else:
        print("⚠ No EXR support - install OpenEXR or imageio for EXR files")
    
    # Initialize comparator with custom parameters
    comparator = NormalMapComparator(
        angular_threshold=2.0,      # 2 degree threshold (tighter for high-precision EXR)
        min_defect_size=10,         # Minimum 20 pixels for a defect  
        registration_enabled=True   # Enable image registration
    )
    
    # Run comparison (supports .exr, .png, .jpg, .tif files)
    results = comparator.compare(
        reference_path="exr_comps/normal_10rupee.exr",
        target_path="exr_comps/normal_new10.exr", 
        show_visualization=True,
        save_results=True,
        output_dir="./exr_analysis_results"
    )
    
    # Access specific results
    # print(f"Found {len(results['defect_info'])} defects")
    # print(f"Quality score: {results['statistics']['quality_metrics']['quality_score']:.1f}")
    
    return comparator

def install_exr_dependencies():
    """Helper function to show how to install EXR dependencies"""
    print("""
To enable full EXR support, install one of these packages:

Option 1 - OpenEXR (recommended for professional workflows):
    pip install OpenEXR

Option 2 - imageio with EXR plugin (easier installation):
    pip install imageio
    pip install imageio[pyexr]

Option 3 - Alternative EXR support:
    pip install imageio-flif  # Alternative EXR backend

For conda users:
    conda install -c conda-forge openexr-python
    conda install -c conda-forge imageio

Note: OpenEXR may require additional system libraries on some platforms.
""")

# Uncomment to run example:
comparator = run_comparison_example()
# install_exr_dependencies()  # Show installation instructions