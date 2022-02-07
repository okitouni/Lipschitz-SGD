from torch import nn
from monotonenorm import direct_norm, GroupSort


def Unconstrained(input_size=1):
    model = nn.Sequential(nn.Linear(input_size, 32),
                          nn.ReLU(),
                          nn.Linear(32, 32),
                          nn.ReLU(),
                          nn.Linear(32, 1)
                          )
    return model


def LipNN(input_size=1):
    model = nn.Sequential(direct_norm(nn.Linear(input_size, 128)),
                          GroupSort(2),
                          direct_norm(nn.Linear(128, 128)),
                          GroupSort(2),
                          nn.Linear(128, 1))
    return model
