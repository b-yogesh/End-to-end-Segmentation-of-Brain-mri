import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels
        ):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    
class Upsample(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels
        ):
        super(Upsample, self).__init__()

        self._up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self._up(x)

class Unet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512]
    ):
        super(Unet, self).__init__()

        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()

        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling
        for feature in features:
            self.downsample.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upsampling
        for feature in reversed(features):
            self.upsample.append(Upsample(feature*2, feature))
            self.upsample.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.reduction = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_conn = []

        for down in self.downsample:
            x = down(x)
            skip_conn.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        
        skip_conn = skip_conn[::-1] # reverse

        for i in range(0, len(self.upsample), 2):
            x = self.upsample[i](x)
            skip_c = skip_conn[i//2]

            ## since we need to concatenate input features with upsampled output, they must have same shape
            ## we enforce this here
            if x.shape != skip_c.shape:
                x = TF.resize(x, size=skip_c.shape[2:])

            skip_concat = torch.cat((skip_c, x), dim=1)
            x = self.upsample[i+1](skip_concat)

        return self.reduction(x)


if __name__ == '__main__':
    x = torch.randn((3, 1, 162, 165))
    model = Unet(1, 1)
    preds = model(x)
    print(preds.shape, x.shape)