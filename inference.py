import torch
import torch.nn as nn
from model import Unet
from utils import load_checkpoint, save_checkpoint, get_loaders
from dataset import create_io_pairs, Brain_Segmentation_Dataset
import torchvision.transforms as Transforms
import cv2

def test():
    path = "D:/U-Net/unet/data/kaggle_3m"
    model_path = "D:/U-Net/my_checkpoint.pth.tar"
    transforms = Transforms.Compose(
        [
            Transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            )
        ],
    )
    pairs = create_io_pairs(path)
    dataset = Brain_Segmentation_Dataset(pairs, transforms)
    item = dataset.__getitem__(2)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet()
    load_checkpoint(torch.load(model_path), model)
    model.to(DEVICE)
    print('Model loaded', model)

    data = item[0].unsqueeze(0).to(device=DEVICE)
    gt = item[1].float().to(device=DEVICE)
    output = model(data)
    print(output[0].detach().cpu().numpy().transpose((1, 2, 0)).shape, gt.shape)
    cv2.imshow('output', output[0].detach().cpu().numpy().transpose((1, 2, 0)))
    cv2.imshow('GT', gt.cpu().numpy().transpose((1, 2, 0)))
    cv2.waitKey(5000)
    output = output.squeeze()
    print(output.shape)

if __name__ == '__main__':
    test()

