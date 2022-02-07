from copy import deepcopy
import torch
from models import LipNN, Unconstrained
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from constants import N_MODELS, BATCH_SIZE, EPOCHS, PRE_EPOCHS, N
torch.manual_seed(0)

N_TRAIN = int(N * 0.8)
LR = 0.02
PLOT = False

X = torch.linspace(0, 2 * torch.pi, N).view(-1, 1)
Y = torch.sin(X).view(-1, 1)

train_set, test_set = random_split(TensorDataset(X, Y), [N_TRAIN, N - N_TRAIN])
torch.save(train_set, 'data/train_set.pt')
torch.save(test_set, 'data/test_set.pt')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)


def init_plot():
    fig, ax = plt.subplots()
    x_plot = X.numpy().flatten()
    y_plot = model(X).detach().numpy().flatten()

    plt.scatter(x_plot, Y.numpy().flatten().flatten(), s=1, c='k', label='True', )
    line, = plt.plot(x_plot, y_plot, label='Prediction')
    text = plt.text(0., 1.01, 'text', transform=ax.transAxes)

    plt.legend()
    return fig, ax, [line, text]


def animate(i, *fargs):
    line = fargs[0]
    text = fargs[1]
    metrics = train(verbose=False)
    x_plot = X.numpy().flatten()
    y_plot = model(X).detach().numpy().flatten()
    text.set_text(
        f'{i}/ train: {metrics["loss_train"]:.3e}, test: {metrics["loss_test"]:.3e}')
    line.set_data(x_plot, y_plot)
    plt.pause(0.001)
    return [line, text]


def train(model, verbose=True):
    metrics = {}
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = (y_pred - y)**2
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        metrics['loss_train'] = loss.item()
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            y_pred = model(x)
            loss_test = (y_pred - y)**2
            loss_test = loss_test.mean()
            metrics['loss_test'] = loss_test.item()
        if verbose:
            pbar.set_description(
                f'Loss: {loss.item():.4f} Test Loss: {loss_test.item():.4f}')
    return metrics


if __name__ == "__main__":
    model = LipNN(1)
    suffix = '_lipnn'
    # model = Unconstrained()
    # suffix = ''
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    if PLOT:
        fig, ax, lines = init_plot()
        anim = FuncAnimation(fig, animate, blit=False,
                             frames=EPOCHS, fargs=lines, repeat=False)
        plt.show()
        print("saving animation...")
        gif_name = f'animation{suffix}.gif'
        anim.save(gif_name, writer='imagemagick')
        print("done!")
    else:
        # pretraining
        print("pretraining...")
        pbar = tqdm(range(PRE_EPOCHS))
        for epoch in pbar:
            train(model)
        model_name = f'model_init{suffix}.pt'
        torch.save(model.state_dict(), f'models/{model_name}.pt')
        for i in range(N_MODELS):
            print(f"Training model_copy {i}")
            pbar = tqdm(range(EPOCHS))
            model_copy = deepcopy(model)
            optimizer = torch.optim.SGD(model_copy.parameters(), lr=LR)
            for epoch in pbar:
                train(model=model_copy)
                torch.save(model_copy.state_dict(), f'models/model{suffix}{i}-{epoch}.pt')
