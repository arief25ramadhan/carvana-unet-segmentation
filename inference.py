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

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = 'cuda'

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
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model_path = 'model/model.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# save_prediction_as_imgs(test_loader, model, folder="saved_images/", device='cuda')

# Inference function
def inference_image(image_path, model, image_transform):
    
    img = np.array(Image.open(image_path).convert("RGB"))
    # get normalized image
    img_normalized = image_transform(image=img)
    img_normalized = img_normalized['image'].unsqueeze(0).to(DEVICE)
    print(img_normalized.shape)

    model.eval()
    
    with torch.no_grad():

        preds = torch.sigmoid(model(img_normalized))
        preds = (preds>0.5).float()
        print(preds.shape)
        torchvision.utils.save_image(
            preds, "pred_test.png"
        )

image_path = 'data/test/0cdf5b5d0ce1_04.jpg'
inference_image(image_path, model, test_transform)