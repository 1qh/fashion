from model import Net
from icecream import ic

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fashion(batch_size = 64):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ]
    )
    train = FashionMNIST(
        root='.',
        train=True,
        download=True,
        transform=transform,
        target_transform=None
    )
    test = FashionMNIST(
        root='.',
        train=False,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, test_loader

def acc_fn(y, y_hat):
    return torch.eq(y, y_hat).sum().item() / len(y) * 100

def train_epoch(model, loader, opti, loss):
    model.train()
    los, acc = 0.0, 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opti.zero_grad()
        y_hat = model(X)
        l = loss(y_hat, y)
        l.backward()
        opti.step()
        los += l.item()
        acc += acc_fn(y, y_hat.argmax(1))

    los /= len(loader)
    acc /= len(loader)

    return los, acc

def eval_model(model, loader, loss):
    model.eval()
    los, acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            los += loss(y_hat, y).item()
            acc += acc_fn(y, y_hat.argmax(1))
        
        los /= len(loader)
        acc /= len(loader)
    
    return los, acc

def go(model, opti, loss, epochs, loaders):
    train, test = loaders
    
    for e in range(epochs):
        print(f'Epoch {e}:\n')
        
        train_loss, train_acc = train_epoch(model, train, opti, loss)
        ic(train_loss, train_acc)
        if e % 5 == 0:
            test_loss, test_acc = eval_model(model, test, loss)
            ic(test_loss, test_acc)

    return model.state_dict()

if __name__ == '__main__':

    epochs = 3
    learning_rate = 0.001
    cnn = Net(
        in_channels=1,
        num_class=10
    ).to(device)
    loss = nn.CrossEntropyLoss()
    opti = optim.Adam(cnn.parameters(), lr=learning_rate)
    loaders = fashion()

    best = go(cnn, opti, loss, epochs, loaders)
    torch.save(best, 'best.pt')