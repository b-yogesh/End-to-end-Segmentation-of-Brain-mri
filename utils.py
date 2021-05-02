import torch
import torchvision
from dataset import Brain_Segmentation_Dataset, create_io_pairs
from torch.utils.data import DataLoader
from torch.utils import data
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    data_dir,
    batch_size,
    transform
):

    pairs = create_io_pairs(data_dir)
    dataset = Brain_Segmentation_Dataset(pairs, transform)
    valid_size = int(0.3 * len(dataset))
    train_set, val_set = data.random_split(dataset, [len(dataset)-valid_size, valid_size])

    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def visualize_segmentation(model, data_loader, num_samples=5, device='cuda'):
    #visualize segmentation on unseen samples
    fig, axs = plt.subplots(num_samples, 3, figsize=(60,60))

    for ax, col in zip(axs[0], ['MRI', 'Ground Truth', 'Predicted Mask']):
        ax.set_title(col)

    index = 0
    for i,batch in enumerate(data_loader):
            img = batch[0].to(device)
            msk = batch[1].to(device)

            output = model(img)

            for j in range(batch[0].size()[0]): #iterate over batchsize
                axs[index,0].imshow(np.transpose(img[j].detach().cpu().numpy(), (1,2,0)).astype(np.uint8), cmap='bone', interpolation='none')

                axs[index,1].imshow(np.transpose(img[j].detach().cpu().numpy(), (1,2,0)).astype(np.uint8), cmap='bone', interpolation='none')
                axs[index,1].imshow(torch.squeeze(msk[j]).detach().cpu().numpy(), cmap='Blues', interpolation='none', alpha=0.5)

                axs[index,2].imshow(np.transpose(img[j].detach().cpu().numpy(), (1,2,0)).astype(np.uint8), cmap='bone', interpolation='none')
                axs[index,2].imshow(torch.squeeze(output[j]).detach().cpu().numpy(), cmap='Greens', interpolation='none', alpha=0.5)

                index += 1

            if index >= num_samples:
                break

    plt.tight_layout()