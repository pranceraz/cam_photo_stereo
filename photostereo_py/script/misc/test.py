import torch
print(torch.__version__)
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

from segment_anything import sam_model_registry
print("segment-anything imported successfully")

from comparator import NormalMapComparator

compare = NormalMapComparator()

compare.image_root_mse()