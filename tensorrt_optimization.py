import time
import torch
from model import UNET 
from utils import save_prediction_as_imgs
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torchvision
import cv2
import torch_tensorrt

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 720
BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = True

test_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 255.0,
        ),
        ToTensorV2(),
    ]
)

device = 'cuda'

# Load model
model = UNET(in_channels=3, out_channels=1)
model_path = 'model/model_2.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()

traced_model = torch.jit.trace(model, [torch.randn((1, 3, 480, 720)).to("cuda")])

inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 3, 240, 360],
        opt_shape=[1, 3, 480, 720],
        max_shape=[1, 3, 480, 720],
        dtype=torch.float32,
    )
]
enabled_precisions = {torch.float32}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    traced_model, inputs=inputs, 
    enabled_precisions=enabled_precisions
)

input_data = input_data.to(device)
result = trt_ts_module(input_data)
torch.jit.save(trt_ts_module, "trt_ts_module.ts")
