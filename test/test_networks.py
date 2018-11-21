import torch
from arch import Discriminator

if __name__ == '__main__':
    img = torch.randn(1,3,128,128)
    d = Discriminator(in_dim=3)
    output = d(img)
    print(output.shape)