import cv2
import numpy as np
import open3d as o3d
import time

from masker import generate_mask_generic

# --- Load normal map and generate mask ---
image_path = "stamp_particle.png"
print("Loading normal map...")
normal_map = cv2.imread(image_path).astype(np.float32) / 255.0  #Converts the RGB values from integers (0–255) to floating point values between 0.0–1.0.

print("Generating mask for normal map using SAM...")
mask = generate_mask_generic(image_path,make_reversed=False)
mask = mask.astype(bool)

# --- Process normals ---
print("Converting RGB to XYZ normals...")
normals = (normal_map - 0.5) * 2
h, w, _ = normals.shape
print(f"Image dimensions: {h}x{w}")

# --- Create 2D grid and apply mask ---
x = np.arange(w)
y = np.arange(h)
xx, yy = np.meshgrid(x, y)

xx_flat = xx.flatten()
yy_flat = yy.flatten()
normals_flat = normals.reshape(-1, 3)
mask_flat = mask.flatten()

# --- Apply mask to create 3D points ---
print("Creating 3D points from unmasked regions...")
points = np.zeros((np.count_nonzero(mask_flat), 3), dtype=np.float32)
points[:, 0] = xx_flat[mask_flat] + normals_flat[mask_flat, 0] * 10#pandas like filtering
points[:, 1] = yy_flat[mask_flat] + normals_flat[mask_flat, 1] * 10 #like normals_flat[mask_flat][:, 0]	
points[:, 2] = normals_flat[mask_flat, 2] * 10

print(f"Points created: {len(points)}")

# --- Create Open3D point cloud ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.normals = o3d.utility.Vector3dVector(normals_flat[mask_flat])

# --- Optional: Clean point cloud ---
print("Removing statistical outliers...")
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# --- Create mesh using Ball Pivoting (safer than Poisson for sparse/unstructured data) ---
print("Estimating mesh using Ball Pivoting...")
radii = [1.0, 2.0, 4.0]
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)

# --- Clean mesh ---
print("Cleaning mesh...")
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_non_manifold_edges()

# --- Save to files ---
print("Saving to STL and PLY...")
o3d.io.write_triangle_mesh("output_masked.stl", mesh)
o3d.io.write_triangle_mesh("output_masked.ply", mesh)

print("Done! Output files:")
print(" - output_masked.stl")
print(" - output_masked.ply")

# --- Optional: Visualize ---
# o3d.visualization.draw_geometries([mesh])
