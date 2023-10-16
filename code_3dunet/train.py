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

if not os.path.exists('./weight'):
    os.mkdir('./weight')

BATCH_SIZE = 4
TRAINING_EPOCH = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight = './weight/weight.pth'
weight_with_optimizer = './weight/weight_with_optimizer.pth'

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = UNetDataset('./data/train/', './data/train_cleaned/')
dataLoader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = UNetDataset('./data_val/train/', './data_val/train_cleaned/')
val_dataLoader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = UNet3D(in_channels=1, num_classes=1)
if torch.cuda.is_available() and True:
    model = model.cuda()
elif not torch.cuda.is_available() and True:
    print('cuda not available! Training initialized on cpu ...')

optimizer = Adam(params=model.parameters(), lr=0.001)


for epoch in range(TRAINING_EPOCH):

    train_loss = 0.0
    model.train()
    # for data in train_dataloader:
    for step, (image, ground_truth) in enumerate(dataLoader):
        optimizer.zero_grad()
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        target = model(image)
        target = target.to(device)
        loss = F.mse_loss(target, ground_truth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for step, (image, ground_truth) in enumerate(val_dataLoader):
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        target = model(image)
        target = target.to(device)
        # loss = criterion(target, ground_truth).to(device)
        loss = F.mse_loss(target, ground_truth)
        valid_loss = loss.item()

    print('epoch: %d | train_loss: %.4f' % (epoch, train_loss))
    print('epoch: %d | valid_loss: %.4f' % (epoch, valid_loss))

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    if not os.path.isdir("./models/checkpoint"):
        os.mkdir("./models/checkpoint")
    torch.save(checkpoint, './models/checkpoint/net%s.pth' % (str(epoch)))

