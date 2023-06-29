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

import numpy as np
import onnx
import onnxruntime
import torch.onnx

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

model = UNET(in_channels=3, out_channels=1)
model_path = 'model/model_2.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

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

# ONNX
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

device_ = torch.device('cpu')
model = model.to(device_)
img = np.array(Image.open(image_path).convert("RGB"))
img_normalized = test_transform(image=img)
img_normalized = img_normalized['image'].unsqueeze(0).to(device_)

# # Export the model
# torch.onnx.export(model,               # model being run
#                   img_normalized,                         # model input (or a tuple for multiple inputs)
#                   "model/model_2.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=15,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                  )


torch.onnx.export(
        model,  # model being run
        img_normalized,  # model input (or a tuple for multiple inputs)
        "model/model_3.onnx",  # where to save the model (can be a file or file-like   object)
        export_params=True,  # store the trained parameter weights inside the model     file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=['inputs'],  # the model's input names
        output_names=['outputs'],  # the model's output names
        dynamic_axes={
            'inputs': {
                0: 'batch_size'
            },  # variable lenght axes
            'outputs': {
                0: 'batch_size'
            }
        })

onnx_model = onnx.load("model/model_3.onnx")
onnx.checker.check_model(onnx_model)

def inference_onnx(model, input, device):
    print("Device Onnx: ", device)
    providers=['CPUExecutionProvider']
    if 'cuda':
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model, providers=providers)
    ort_inputs = {ort_session.get_inputs()[0].name: input}
    _ = ort_session.run(None, ort_inputs)
    

start_time = time.time()
inference_onnx(onnx_model, to_numpy(img_normalized), device_)
elapsed_time = time.time() - start_time
print("Onnx Model CPU time: ", elapsed_time)