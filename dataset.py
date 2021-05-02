import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

def create_io_pairs(path):
    
    pairs = []
    assert os.path.exists(path), 'Path does not exist'
    for directory in [os.path.join(path,x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
        for file in os.listdir(directory):
            #add files to list
            if 'mask' not in file:
                result = 0
                img_path = os.path.join(directory, file)
                mask_path = os.path.join(directory, file[:file.find('.tif')]+'_mask.tif')
                    
                #check if mask is nonempty
                if np.max(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)) > 0:
                    result = 1
                
                pairs.append([img_path, mask_path, result])

    print('Len of total samples', len(pairs))
    return pairs


class Brain_Segmentation_Dataset(Dataset):
    def __init__(self, inputs, transform=None):
        self.inputs = inputs
        self.transform = transform
        self.input_dtype = torch.float32
        self.target_dtype = torch.float32
        self.val_transforms = Transforms.Compose(
        [
            Transforms.Normalize(
                mean=[0.0],
                std=[1.0]
            )
        ],
    )
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        img_path = self.inputs[index][0]
        mask_path = self.inputs[index][1]
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        x = torch.from_numpy(np.transpose(np.array(cv2.imread(img_path)), (2,0,1))).type(self.input_dtype)
        y = torch.from_numpy(np.resize(np.array(mask_img)/255., (1,256,256))).type(self.target_dtype)
        
        if self.transform is not None:
            x = self.transform(x)
            y = self.val_transforms(y)
        
        return x,y


if __name__ == "__main__":
    pairs = create_io_pairs('D:/U-Net/unet/data/kaggle_3m')
    dataset = Brain_Segmentation_Dataset(pairs)
    item = dataset.__getitem__(2)
    print(item)
