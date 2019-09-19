import torch
import torchvision
import numpy as np
import d2l
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython import display

def load_data_fashion_mnist(batch_size, resize=None, num_workers=0):
    tranform_list = []
    if resize:
        tranform_list.append(torchvision.transforms.Resize(resize))
    tranform_list.append(transforms.ToTensor())
    transform = transforms.Compose(tranform_list)
    mnist_train = datasets.FashionMNIST('~/datasets/F_MNIST/',
                                 download=True,
                                 train=True,
                                 transform=transform)

    mnist_test = datasets.FashionMNIST('~/datasets/F_MNIST/',
                                     download=True,
                                     train=False,
                                    transform=transform)
    
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, 
                                              shuffle=False)
    return train_iter, test_iter

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).sum()


class Accumulator(object):
    """Sum a list of numbers over time"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a+b for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def evaluate_accuracy(net, data_iter):
    if hasattr(net, 'training'):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), len(y))
    return float(metric[0])/metric[1]


def sgd(params, lr):
    for param in params:
        param.data -= lr * param.grad.data
        param.grad.data.zero_()

def train_epoch(net, train_iter, loss, updater):
    if hasattr(net, 'training'):
        net.train()
    metric = Accumulator(3) # train_loss_sum, train_acc_sum, num_examples
    for X, y in train_iter:
        y_hat = net(X)
        # compute gradients and update parameters
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.SGD):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.backward()
            updater()
        l_sum = float(l)*len(y)
        metric.add(l_sum, float(accuracy(y_hat, y)), len(y))
    return metric[0]/metric[2], metric[1]/metric[2]


class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : d2l.set_axes(self.axes[0], xlabel, ylabel,
                                                 xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    trains, test_accs = [], []
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], 
                        ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
        
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.squeeze(0).numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
            
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']
    return [text_labels[int(i)] for i in labels]


def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y.numpy())
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1).numpy())
    titles = [true+'\n'+ pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n,28,28)), 1, n, titles=titles[0:n])