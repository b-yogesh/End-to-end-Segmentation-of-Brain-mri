from flask import Flask, request, jsonify
from utils import load_checkpoint, save_checkpoint, get_loaders
from dataset import create_io_pairs, Brain_Segmentation_Dataset
import torchvision.transforms as Transforms
import torch
from model import Unet
import numpy as np

app = Flask('app')

@app.route('/predict', methods=['POST'])
def test():
    data = request.get_json()
    print('here', np.array(data['data']).shape)
    model_path = "D:/U-Net/my_checkpoint.pth.tar"
    DEVICE = "cpu"
    model = Unet()
    load_checkpoint(torch.load(model_path), model)
    model.double().to(DEVICE)
    data = torch.from_numpy(np.array(data['data'])).double().unsqueeze(0).to(device=DEVICE)
    output = inference(data, model)
    result = {
        'segmap': output.detach().numpy().tolist()
    }
    return jsonify(result)

def inference(data, model):
    output = model(data)
    print(output.shape)
    return output

# if __name__ == '__main__':
#     app.run(debug=True)