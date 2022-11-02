from model import Net

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import FashionMNIST 
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test = FashionMNIST(root='.', train=False, download=True,
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
)
test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False)

md = Net(in_channels=1, num_class=10).to(device)
md.load_state_dict(torch.load('best.pt'))
md.eval()

y_hat = []

with torch.inference_mode():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_hat.append(md(X).softmax(0).argmax(1))

y_hat = torch.cat(y_hat).to('cpu')
y = test.targets

fig, ax = plot_confusion_matrix(
    ConfusionMatrix(10)(preds=y_hat, target=y).numpy(),
    class_names=test.classes
)
plt.show()

