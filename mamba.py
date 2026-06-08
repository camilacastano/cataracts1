# first use of Video Mamba model with our own data

import os
import torch

# Load directly via hub configuration or git import 
# Source: https://github.com/OpenGVLab/VideoMamba
model_url = "https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_b16_in1k_res224.pth"
checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")

print("Official checkpoint loaded successfully.")

