import cv2
import numpy as np
import torch
import os
from unet3d import UNet3D
from skimage import io
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.optim import Adam
from unet3d import UNet3D
import os
from datasets_01 import UNetDataset
from torch.utils.data import DataLoader
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load net
print('load net')
model = UNet3D(in_channels=1, num_classes=1).to(device)
path_checkpoint = "./models/checkpoint/net.pth"  # 断点路径
checkpoint = torch.load(path_checkpoint)
model.load_state_dict(checkpoint['net'])


# load img
print('load img')
test_dataset = UNetDataset('./data/test/raw/', './data/test/gt/')
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=0)


model.eval()

for step, (image, ground_truth) in enumerate(test_dataLoader):
    image = image.to(device)
    output = model(image)
    output = output.to(device)


dir1 = './data/test/net/'
for i in range(0,16):
    arrayImg = output.cpu().detach().numpy()  # transfer tensor to array

    arrayShow = np.squeeze(arrayImg[0], 0)

    arrayShow = arrayShow[i, ::]  # extract the image being showed
    cv2.imwrite(dir1+'%02d.tiff'%i, arrayShow)