from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io


class UNetDataset(Dataset):
    def __init__(self, dir_train, dir_mask, transform=None):
        self.dirTrain = dir_train
        self.dirMask = dir_mask
        self.transform = transform
        self.dataTrain = [os.path.join(self.dirTrain, filename)
                          for filename in os.listdir(self.dirTrain)
                          if filename.endswith('.tiff') or filename.endswith('.png')]
        self.dataMask = [os.path.join(self.dirMask, filename)
                         for filename in os.listdir(self.dirMask)
                         if filename.endswith('.tiff') or filename.endswith('.png')]
        self.trainDataSize = len(self.dataTrain)
        self.maskDataSize = len(self.dataMask)

    # 根据索引获取data和label
    def __getitem__(self, index):
        assert self.trainDataSize == self.maskDataSize
        image = io.imread(self.dataTrain[index]).astype(np.float32)
        label = io.imread(self.dataMask[index]).astype(np.float32)

        return image[np.newaxis], label[np.newaxis]

    # 获取数据集的大小
    def __len__(self):
        assert self.trainDataSize == self.maskDataSize
        return self.trainDataSize




if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = UNetDataset('./data/train', './data/train_cleaned', transform=None)
    dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    for index, (image, gt) in enumerate(dataLoader):

        img1 = image
        img2 = gt
        print(img1.shape)




