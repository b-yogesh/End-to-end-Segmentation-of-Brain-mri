import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as Transforms
from loss import bce_dice_loss
from metric import iou
from dataset import create_io_pairs, Brain_Segmentation_Dataset
from model import Unet
from utils import load_checkpoint, save_checkpoint, get_loaders

# Config params
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
LOAD_MODEL = False
DATA_DIR = "D:/U-Net/unet/data/kaggle_3m"


def train(loader, val_loader, model, optimizer, loss_fn, metric):
    loop = tqdm(loader)
    model.train()
    loss, score = run_model(loop, model, optimizer, loss_fn, metric)

    model.eval()
    loop = tqdm(val_loader)
    val_loss, val_score = run_model(loop, model, optimizer, loss_fn, metric)

    return loss, score, val_loss, val_score
    

def run_model(loop, model, optimizer, loss_fn, metric):
    losses = []
    scores = []
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.float().to(device=DEVICE)
        predictions = model(data)
        loss = loss_fn(predictions, target)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        score = metric(predictions.detach(), target)
        scores.append(score)
    loss = sum(losses) / len(losses)
    scores = sum(scores) / len(scores)
    
    return loss, score


def main():
   
    transforms = Transforms.Compose(
        [
            Transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            )
        ],
    )

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    
    print('Model Initialized')
    
    loss_fn = bce_dice_loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    metric = iou

    train_loader, val_loader = get_loaders(
        DATA_DIR,
        BATCH_SIZE,
        transforms
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    print('Starting training')
    
    for epoch in range(NUM_EPOCHS):
        loss, score, val_loss, val_score = train(train_loader, val_loader, model, optimizer, loss_fn, metric)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        print(f'Epoch: {epoch}, Loss: {loss}, score: {score}, Val_loss: {val_loss}, Val_Score: {val_score}')



if __name__ == "__main__":
    main()