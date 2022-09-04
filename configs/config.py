import torch
from typing import Dict, Any

# Get device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Get ROILANet trained weight
# ROI_WEIGHT_PATH = "./models/ROILAnet/weights_path/ROI_extractor_augmented_TJ-NTU.pt"

# # Get mobilenet pretrained weights on ImageNet dataset
# MOBILENET_PRETRAINED_WEIGHT = "./models/backbones/weights_path/mobilenet_v2.pth"

# # Get deep metric learning weight
# CHECKPOINT_PATH: str = "./checkpoints/mobilenetv2/30_07/epoch115.pth"

# CHECKPOINT: Dict[str, Any] = torch.load(CHECKPOINT_PATH, map_location="cpu")

# CONFIG: Dict[str, Any] = CHECKPOINT["config"]

