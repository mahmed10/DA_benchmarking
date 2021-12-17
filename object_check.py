import data
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
target_loader = data.setup_loaders('semantickitti', './dataset/SemanticKitti/vallist.txt', 1)
# target_loader = data.setup_loaders('rellis3d', './dataset/Rellis3d/trainlist.txt', 1)
targetloader_iter = enumerate(target_loader)
for i in range(6000):
    _, batch = targetloader_iter.__next__()
    x, y= batch
    y1 = y==18
    if torch.sum(y1)>5000:
        x1 = x * y1
        trans = transforms.ToPILImage()
        plt.figure()
        plt.imshow(trans(x[0]))
        plt.figure()
        plt.imshow(trans(x1[0]))
        break