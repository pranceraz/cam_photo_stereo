import torch
import pickle  
import numpy as np
print(torch.__version__)
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

from segment_anything import sam_model_registry
print("segment-anything imported successfully")

def load_normals_pickle(filename):
        """Load normals from pickle file"""
        with open(filename, 'rb') as f:
            normals = pickle.load(f)
        normals_matrix = np.array(normals)
    
        return normals_matrix

from comparator import NormalMapComparator

# compare = NormalMapComparator("stamp_pink.png","stamp_particle.png")

# compare.realign()

new_comp = NormalMapComparator("stamp_pink.png","aligned_particle.png")
# ncc_map = new_comp.sliding_cosine_similarity(4,.3,.8)
#new_comp.visualize_similarity_map(ncc_map)
corr_map = new_comp.vector_cross_correlation(16,.2,.8)

# Normalize for better visualization (optional)
vis_map = (corr_map - corr_map.min()) / (corr_map.max() - corr_map.min())

# Plot it
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(vis_map, cmap='inferno')  # or 'viridis', 'plasma', 'gray'
plt.title("Vector Cross-Correlation Map")
plt.colorbar(label='Mean Dot Product')
plt.axis('off')
plt.show()
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 6))

# # Emphasize poor matches (low NCC values). Use 'plasma' or 'magma' for better low-end contrast
# plt.imshow(ncc_map, cmap='magma', vmin=-1.0, vmax=0.5)

# # Optional: mask high matches for stronger effect
# highlight_map = np.copy(ncc_map)
# highlight_map[ncc_map > 0.5] = np.nan  # Hide good matches

# plt.imshow(highlight_map, cmap='magma')
# plt.colorbar(label='NCC Score (Low = Bad Match)')
# plt.title('Highlighted Low-Match Regions (NCC)')
# plt.axis('off')
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assume ncc_map is your sliding NCC result (values in [-1, 1])

# # Invert and clip the map to only show poor matches
# bad_match_map = 1 - np.clip(ncc_map, 0, 1)

# # Optional: emphasize only values below a certain threshold (e.g., 0.6)
# threshold = 0.6
# emphasized = (ncc_map < threshold).astype(float) * (1 - ncc_map)

# plt.figure(figsize=(10, 6))
# plt.imshow(ncc_map, cmap='magma', vmin=0, vmax=1)  # 'magma' is good for visualizing low intensities
# plt.colorbar(label='Bad Match Emphasis (1 - NCC)')
# plt.title('Low-Correlation (Bad Match) Map')
# plt.axis('off')
# plt.tight_layout()
# plt.show()

# import matplotlib.colors as mcolors

# plt.figure(figsize=(10, 6))
# div_norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
# plt.imshow(ncc_map, cmap='coolwarm', norm=div_norm)
# plt.colorbar(label='NCC Score')
# plt.title('NCC Map with Diverging Colors')
# plt.axis('off')
# plt.tight_layout()
# plt.show()
