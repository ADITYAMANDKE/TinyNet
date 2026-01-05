import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
             super().__init__()
             self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1,bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 16, 3,padding=1,groups=16,bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 58, 1, bias=False),
                nn.BatchNorm2d(58),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
             )
             self.fc = nn.Linear(58, 10)
         def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x,1)
            x = self.fc(x)
            return x
    return TinyNet()
