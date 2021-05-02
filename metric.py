import torch

def iou(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5):
    # IoU Metric from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  
    labels = labels.squeeze(1)
    
    bin_out = torch.where(outputs > threshold, 1, 0).type(torch.int16)
    labels = labels.type(torch.int16)
    
    intersection = (bin_out & labels).float().sum((1, 2))  
    union = (bin_out | labels).float().sum((1, 2))       
    
    iou = (intersection + SMOOTH) / (union + SMOOTH) 
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10 
    
    return thresholded.mean()