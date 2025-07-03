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

compare = NormalMapComparator("stamp_pink.png","stamp_particle.png")

compare.realign()