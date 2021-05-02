from utils import load_checkpoint, save_checkpoint, get_loaders
from dataset import create_io_pairs, Brain_Segmentation_Dataset
import torchvision.transforms as Transforms



if __name__ == '__main__':
    import requests
    path = "D:/U-Net/unet/data/kaggle_3m"
    pairs = create_io_pairs(path)
    transforms = Transforms.Compose(
        [
            Transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            )
        ],
    )
    dataset = Brain_Segmentation_Dataset(pairs, transforms)
    item = dataset.__getitem__(2)
    data = {
        'data': item[0].numpy().tolist()
    }
    url = 'https://brain-segmap-app.herokuapp.com/predict'
    r = requests.post(url, json=data)
    r.text.strip()
    print(r.text.strip())