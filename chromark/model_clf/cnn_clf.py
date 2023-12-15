import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_VAE_clf(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2,nclasses):
        super(CNN_VAE_clf, self).__init__()
 
        self.encoder = nn.Sequential(
            # input is nc x imsize x imsize
            nn.Conv2d(nc, hidden1, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden1) x imsize/stride^2
            nn.Conv2d(hidden1, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden2, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden3, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden4, hidden5, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc=nn.Sequential(
            nn.Linear(fc1, fc2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc2, fc2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc2,nclasses)
        )
 
 
    def forward(self, x):
        h=self.encoder(x)
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fc(h)