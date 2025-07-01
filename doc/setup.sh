#!/bin/bash

pip install uv
uv pip install -r requirements.txt
#uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#to run use uv run example.py
# uv pip install git+https://github.com/facebookresearch/segment-anything.git
## Using wget
#wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth