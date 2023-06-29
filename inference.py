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
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True

TEST_IMG_DIR = 'data/test/'
TEST_MASK_DIR = 'data/test_masks/'

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

test_ds = CarvanaDataset(
    image_dir=TEST_IMG_DIR,
    mask_dir=TEST_MASK_DIR,
    transform=test_transform
)

test_loader= DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False,
)

# Load model
model = UNET(in_channels=3, out_channels=1)
model_path = 'model/model_2.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# save_prediction_as_imgs(test_loader, model, folder="saved_images/", device='cuda')

# Inference function
def inference_image(image_path, model, image_transform, device='cuda'):

    model = model.to(device)
    
    img = np.array(Image.open(image_path).convert("RGB"))
    # get normalized image
    img_normalized = image_transform(image=img)
    img_normalized = img_normalized['image'].unsqueeze(0).to(device)
    print(img_normalized.shape)

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


# image_path = 'data/test/0cdf5b5d0ce1_04.jpg'
image_path = 'data/test/0ee135a3cccc_04.jpg'
inference_image(image_path, model, test_transform)