import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os
import torch
import torch.nn.functional as F

class VectorBasedNormalMapProcessor:
    def __init__(self, template_size: Tuple[int, int] = (32, 32), 
                 downscale_factor: float = 0.2, overlap_ratio: float = 0.5):
        """
        Vector-based normal map comparison processor
        
        Args:
            template_size: Size of templates to extract (increased from 16x16)
            downscale_factor: Factor to downscale images
            overlap_ratio: Overlap between templates (reduced for better coverage)
        """
        self.template_size = template_size
        self.downscale_factor = downscale_factor
        self.overlap_ratio = overlap_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rgb_to_normal_vector(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB normal map to normal vectors
        
        Args:
            rgb_image: Input RGB image (0-255 range)
            
        Returns:
            Normal vectors in range [-1, 1] with shape (H, W, 3)
        """
        # Convert to float and normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to normal vector range [-1, 1]
        normal_vectors = normalized * 2.0 - 1.0
        
        # Ensure Z component is positive (normal maps typically have positive Z)
        normal_vectors[:, :, 2] = np.abs(normal_vectors[:, :, 2])
        
        # Normalize vectors to unit length
        norms = np.linalg.norm(normal_vectors, axis=2, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normal_vectors = normal_vectors / norms
        
        return normal_vectors

    def compute_angular_difference(self, normal1: np.ndarray, normal2: np.ndarray) -> np.ndarray:
        """
        Compute angular difference between normal vectors
        
        Args:
            normal1: First normal vector array (H, W, 3)
            normal2: Second normal vector array (H, W, 3)
            
        Returns:
            Angular differences in radians (H, W)
        """
        # Compute dot product
        dot_product = np.sum(normal1 * normal2, axis=2)
        
        # Clamp to [-1, 1] to avoid numerical errors in arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angular difference
        angular_diff = np.arccos(np.abs(dot_product))
        
        return angular_diff

    def compute_similarity_score(self, template_vectors: np.ndarray, 
                               region_vectors: np.ndarray) -> float:
        """
        Compute similarity score between template and region normal vectors
        
        Args:
            template_vectors: Template normal vectors (H, W, 3)
            region_vectors: Region normal vectors (H, W, 3)
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        angular_diff = self.compute_angular_difference(template_vectors, region_vectors)
        
        # Convert angular difference to similarity score
        # Angular difference of 0 = similarity of 1
        # Angular difference of π/2 = similarity of 0
        similarity = 1.0 - (angular_diff / (np.pi / 2))
        similarity = np.clip(similarity, 0.0, 1.0)
        
        # Return mean similarity across the template
        return np.mean(similarity)

    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load and process mask file"""
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask from: {mask_path}")

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        print(f"Loaded mask: {mask.shape[1]}x{mask.shape[0]}")
        return mask

    def find_crop_boundaries(self, mask: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
        """Find boundaries for cropping based on mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No white regions found in mask")
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(mask.shape[1] - x, w + 2 * padding)
        h = min(mask.shape[0] - y, h + 2 * padding)
        
        print(f"Crop boundaries: x={x}, y={y}, width={w}, height={h}")
        return x, y, w, h

    def crop_image_with_mask(self, image: np.ndarray, mask: np.ndarray, 
                           padding: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop image based on mask boundaries"""
        if image.shape[:2] != mask.shape[:2]:
            print(f"Resizing mask from {mask.shape} to match image {image.shape[:2]}")
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        x, y, w, h = self.find_crop_boundaries(mask, padding)
        cropped_image = image[y:y+h, x:x+w]
        
        print(f"Original image size: {image.shape[1]}x{image.shape[0]}")
        print(f"Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        
        return cropped_image, (x, y, w, h)

    def downscale_image(self, image: np.ndarray) -> np.ndarray:
        """Downscale image while preserving normal map properties"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * self.downscale_factor), int(w * self.downscale_factor)
        
        if new_h < self.template_size[1] or new_w < self.template_size[0]:
            print("Downscaled image too small for template extraction. Skipping downscale.")
            return image
        
        print(f"Downscaling from {w}x{h} to {new_w}x{new_h}")
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return downscaled

    def extract_templates_with_vectors(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        """
        Extract templates and convert to normal vectors
        
        Returns:
            List of (template_rgb, template_vectors, (x, y)) tuples
        """
        templates = []
        h, w = image.shape[:2]
        template_w, template_h = self.template_size
        
        if h < template_h or w < template_w:
            print("Image too small for template extraction. Skipping.")
            return []
        
        step_x = int(template_w * (1 - self.overlap_ratio))
        step_y = int(template_h * (1 - self.overlap_ratio))
        
        print(f"Template extraction - Size: {template_w}x{template_h}, Step: {step_x}x{step_y}")
        
        for y in range(0, h - template_h + 1, step_y):
            for x in range(0, w - template_w + 1, step_x):
                template_rgb = image[y:y+template_h, x:x+template_w]
                template_vectors = self.rgb_to_normal_vector(template_rgb)
                templates.append((template_rgb, template_vectors, (x, y)))
        
        print(f"Extracted {len(templates)} templates with normal vectors")
        return templates

    def compare_with_fake_vectors(self, templates: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]], 
                                 fake_image: np.ndarray, 
                                 angular_threshold: float = 0.25) -> List[Tuple[int, int, float]]:
        """
        Compare templates with fake image using vector-based approach
        
        Args:
            templates: List of (template_rgb, template_vectors, (x, y))
            fake_image: Target image to compare against
            angular_threshold: Threshold for angular difference (in radians)
            
        Returns:
            List of (x, y, similarity_score) for suspect regions
        """
        fake_vectors = self.rgb_to_normal_vector(fake_image)
        suspect_regions = []
        
        print(f"Comparing {len(templates)} templates using vector-based approach...")
        
        for i, (template_rgb, template_vectors, (x, y)) in enumerate(templates):
            # Extract corresponding region from fake image
            h, w = template_vectors.shape[:2]
            
            # Check bounds
            if y + h > fake_vectors.shape[0] or x + w > fake_vectors.shape[1]:
                continue
            
            fake_region_vectors = fake_vectors[y:y+h, x:x+w]
            
            # Compute similarity score
            similarity = self.compute_similarity_score(template_vectors, fake_region_vectors)
            
            # Convert similarity to angular difference for thresholding
            # Low similarity indicates high angular difference
            if similarity < (1.0 - angular_threshold / (np.pi / 2)):
                suspect_regions.append((x, y, similarity))
        
        print(f"Found {len(suspect_regions)} suspect regions with vector-based comparison")
        return suspect_regions

    def visualize_angular_differences(self, image1: np.ndarray, image2: np.ndarray, 
                                    output_path: str = "angular_differences.jpg"):
        """
        Create a heatmap showing angular differences between two normal maps
        """
        vectors1 = self.rgb_to_normal_vector(image1)
        vectors2 = self.rgb_to_normal_vector(image2)
        
        angular_diff = self.compute_angular_difference(vectors1, vectors2)
        
        # Convert to degrees for better interpretation
        angular_diff_degrees = np.degrees(angular_diff)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(angular_diff_degrees, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Angular Difference (degrees)')
        plt.title('Angular Differences Between Normal Maps')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Angular difference heatmap saved as: {output_path}")
        return angular_diff_degrees

    def mark_suspect_areas_vector_based(self, image: np.ndarray, 
                                      suspect_regions: List[Tuple[int, int, float]]) -> np.ndarray:
        """Mark suspect areas with vector-based scores"""
        marked = image.copy()
        overlay = marked.copy()
        
        for (x, y, similarity) in suspect_regions:
            x0, y0 = x, y
            w, h = self.template_size
            
            # Color based on similarity score (red for low similarity)
            color_intensity = int(255 * (1 - similarity))
            cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, color_intensity), -1)
            
            # Add text with similarity score
            cv2.putText(overlay, f"S:{similarity:.2f}", (x0, y0 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.4, marked, 0.6, 0, marked)
        return marked

    def process_normal_maps(self, reference_path: str, test_path: str, mask_path: str,
                          padding: int = 10, angular_threshold: float = 0.25,
                          save_intermediate: bool = True) -> Tuple[np.ndarray, List]:
        """
        Process normal maps using vector-based comparison
        
        Args:
            reference_path: Path to reference normal map
            test_path: Path to test normal map
            mask_path: Path to mask file
            padding: Padding for cropping
            angular_threshold: Angular threshold in radians (0.25 ≈ 14 degrees)
            save_intermediate: Whether to save intermediate results
        """
        print("="*60)
        print("VECTOR-BASED NORMAL MAP COMPARISON PIPELINE")
        print("="*60)
        
        # Load images
        print("1. Loading reference and test normal maps...")
        reference_image = cv2.imread(reference_path)
        test_image = cv2.imread(test_path)
        
        if reference_image is None or test_image is None:
            raise FileNotFoundError("Could not load one or both images")
        
        mask = self.load_mask(mask_path)
        
        # Process reference image
        print("\n2. Processing reference image...")
        ref_cropped, crop_bounds = self.crop_image_with_mask(reference_image, mask, padding)
        ref_downscaled = self.downscale_image(ref_cropped)
        
        # Process test image
        print("\n3. Processing test image...")
        test_cropped, _ = self.crop_image_with_mask(test_image, mask, padding)
        test_downscaled = self.downscale_image(test_cropped)
        
        # Extract templates with vectors
        print("\n4. Extracting templates and converting to normal vectors...")
        templates = self.extract_templates_with_vectors(ref_downscaled)
        
        # Compare using vector-based approach
        print("\n5. Performing vector-based comparison...")
        suspect_regions = self.compare_with_fake_vectors(templates, test_downscaled, angular_threshold)
        
        # Create visualizations
        if save_intermediate:
            print("\n6. Saving results...")
            
            # Save processed images
            cv2.imwrite("ref_processed.jpg", ref_downscaled)
            cv2.imwrite("test_processed.jpg", test_downscaled)
            
            # Create angular difference heatmap
            angular_diff = self.visualize_angular_differences(ref_downscaled, test_downscaled)
            
            # Mark suspect areas
            marked = self.mark_suspect_areas_vector_based(test_downscaled, suspect_regions)
            cv2.imwrite("marked_suspect_vector_based.jpg", marked)
            
            print("Saved: ref_processed.jpg, test_processed.jpg, marked_suspect_vector_based.jpg")
            print("Saved: angular_differences.jpg (heatmap)")
        
        print(f"\nComparison complete. Found {len(suspect_regions)} suspect regions.")
        print(f"Angular threshold used: {angular_threshold:.3f} radians ({np.degrees(angular_threshold):.1f} degrees)")
        
        return test_downscaled, suspect_regions

def main():
    """Main function to run vector-based normal map comparison"""
    processor = VectorBasedNormalMapProcessor(
        template_size=(32, 32),  # Larger templates for better context
        downscale_factor=0.2,
        overlap_ratio=0.5        # Less overlap for better coverage
    )
    
    # File paths
    reference_path = "5cent_before.png"
    test_path = "aligned_test_to_ref.png"
    mask_path = "mask.png"
    
    # Check if files exist
    if not all(os.path.exists(path) for path in [reference_path, test_path, mask_path]):
        print("Error: One or more required files not found.")
        print(f"Required files: {reference_path}, {test_path}, {mask_path}")
        return
    
    try:
        # Process normal maps
        result_image, suspect_regions = processor.process_normal_maps(
            reference_path=reference_path,
            test_path=test_path,
            mask_path=mask_path,
            padding=20,
            angular_threshold=0.25,  # About 14 degrees
            save_intermediate=True
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total suspect regions found: {len(suspect_regions)}")
        
        if suspect_regions:
            similarities = [score for _, _, score in suspect_regions]
            print(f"Similarity scores range: {min(similarities):.3f} - {max(similarities):.3f}")
            print(f"Mean similarity: {np.mean(similarities):.3f}")
        
        print("\nVector-based comparison completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()