# first use of Video Mamba model with our own data

import os
import torch
from videomamba import VideoMamba
import numpy as np
from torchvision import transforms

# Source: https://github.com/OpenGVLab/VideoMamba
model_url = "https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_b16_in1k_res224.pth"
checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")

print("Official checkpoint loaded successfully.")

"""----------------------"""

video_path="/Users/camilacastano/Desktop/cataracts1/1/Maria Edith Feria Escalera OI .mp4"
output_path="/Users/camilacastano/Desktop/cataracts1/1/processed_video.mp4"


num_frames=16
img_size=224

devide=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

