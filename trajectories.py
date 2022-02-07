from copy import deepcopy
import torch
from models import Unconstrained, LipNN
from matplotlib import pyplot as plt
from constants import N_MODELS, BATCH_SIZE, EPOCHS
index_init = 0
index_final = 0

# model = Unconstrained(1)
# suffix = ''
model = LipNN(1)
suffix = '_lipnn'

model.load_state_dict(torch.load(f'models/model{suffix}{index_final}-{0}.pt'))
train_set = torch.load('data/train_set.pt')
test_set = torch.load('data/test_set.pt')
x, y = train_set.dataset.tensors

model_final = deepcopy(model)
model_final.load_state_dict(torch.load(
    f'models/model{suffix}{index_final}-{EPOCHS-1}.pt'))
dots = []
angles = []
angles_top5 = []
dots_top5 = []
losses = []
norms = []
diff_norm = []
param_f = torch.concat([p.view(-1) for p in model_final.parameters()])

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for i in range(0, EPOCHS - 1):
    model.load_state_dict(torch.load(f'models/model{suffix}{index_init}-{i}.pt'))
    optimizer.zero_grad()
    y_pred = model(x)
    loss = (y_pred - y)**2
    loss = loss.mean()
    loss.backward()
    param = torch.concat([p.view(-1) for p in model.parameters()])
    param_grad = torch.concat([p.grad.view(-1) for p in model.parameters()])
    _, indices = torch.sort(param_grad)
    dot = (-param_grad * (param_f - param)).sum().item()
    angle = (dot / torch.norm(param_grad) / torch.norm(param_f - param)).item()
    dot_top5 = (-param_grad[indices[:2]] *
                (param_f[indices[:2]] - param[indices[:2]])).sum().item()
    angle_top5 = (dot_top5 / torch.norm(param_grad[indices[: 2]]) / torch.norm(
        param_f[indices[: 2]] - param[indices[: 2]])).item()

    dots.append(dot)
    angles.append(angle)
    losses.append(loss.item())
    norms.append(torch.norm(param).item())
    diff_norm.append(torch.norm(param_f - param).item())

    angles_top5.append(angle_top5)
    dots_top5.append(dot_top5)


# Plotting
fig, axes = plt.subplots(2, 2, sharex=True)
colors = ['tab:blue', 'gray', 'tab:red']

for ax1, line, label in zip(axes.flatten(), [dots, angles], ["dot product", "cos angle"]):
    ax1.plot(line, c=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.set_ylabel(label, c=colors[0])
    # ax1.set_yscale('symlog')
    ax1.axhline(0, c=colors[2])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(losses, c=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.set_ylabel('losses', c=colors[1])
    # ax2.set_yscale('log')
for ax in axes.flatten()[-2:]:
    ax.set_xlabel('Epoch')

ax = axes.flatten()[2]
ax.plot(norms, c=colors[0])
ax.set_ylabel('norms', c=colors[0])
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(diff_norm, c=colors[1])
ax2.set_ylabel('diff norm', c=colors[1])
# ax2.set_yscale('log')

ax = axes.flatten()[3]
ax.plot(angles_top5, c=colors[0])
ax.set_ylabel('cosangle topX', c=colors[0])
# ax.set_yscale('symlog')
ax.axhline(0, c=colors[2])
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(dots_top5, c=colors[1])
ax2.set_ylabel('dot topX', c=colors[1])
ax2.axhline(0, c=colors[2])


fig.suptitle(f'{index_init}-{index_final}{suffix}')
plt.tight_layout()
plt.savefig(f'figures/dot_angle{index_init}-{index_final}{suffix}.png')
plt.show()
