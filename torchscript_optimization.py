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

# Load model
model = UNET(in_channels=3, out_channels=1)
model_path = 'model/model_2.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# Inference function
def inference_image(image_path, model, image_transform, device='cuda'):

    print("Device: ", device)

    model = model.to(device)
    img = np.array(Image.open(image_path).convert("RGB"))
    img_normalized = image_transform(image=img)
    img_normalized = img_normalized['image'].unsqueeze(0).to(device)

    model.eval()
    
    with torch.no_grad():

        preds = torch.sigmoid(model(img_normalized))
        preds = (preds>0.5).float()

        torchvision.utils.save_image(
            preds, "pred.png"
        )
       
    img_tensor = img_normalized*255
    mask_tensor = preds*255

    # Convert Image to OpenCV
    cv_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    cv_mask = mask_tensor[0].cpu().numpy().transpose(1, 2, 0)

    # Masking
    masked_car = np.copy(cv_img)
    masked_car[(cv_mask>254).all(-1)] = [0,255,0]
    masked_car_w = cv2.addWeighted(masked_car, 0.3, cv_img, 0.7, 0, masked_car)
    cv2.imwrite('masked_car_w.jpg', masked_car_w)


# CPU Torch
image_path = 'data/test/0ee135a3cccc_04.jpg'
start_time = time.time()
inference_image(image_path, model, test_transform, device='cpu')
elapsed_time = time.time() - start_time
print("Torch Model CPU time: ", elapsed_time)

# GPU Torch
if torch.cuda.is_available():
    start_time = time.time()
    inference_image(image_path, model, test_transform, device='cuda')
    elapsed_time = time.time() - start_time
    print("Torch Model GPU time: ", elapsed_time)

# Scripting Model 
scripted_model = torch.jit.script(model)

# CPU
start_time = time.time()
inference_image(image_path, scripted_model, test_transform, device='cpu')
elapsed_time = time.time() - start_time
print("Scripted Model CPU time: ", elapsed_time)

# GPU Torch
if torch.cuda.is_available():
    start_time = time.time()
    inference_image(image_path, scripted_model, test_transform, device='cuda')
    elapsed_time = time.time() - start_time
    print("Scripted Model GPU time: ", elapsed_time)