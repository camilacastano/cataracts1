"""This is the first test on segmentation based on our dataset
called "pre-dataset"

Segmentation of cataract fragments using a simple thresholding method where we will
enclose cataract fragments in bounding boxes with individual IDs during phacoemulsification

For each fragment:
* Persistent and unique ID across video frames
* Colored bounding box with ID label

Dependencies: OpenCV, NumPy, PyTorch (for potential future use of deep learning models), scipy,
filterpy, rich, tqdm



"""
import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from collections import deque
import math


