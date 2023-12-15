import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.mutualInfo import MutualInformation

class CNN_VAE(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
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
        self.encoder=nn.DataParallel(self.encoder)
 
        self.fcE1 = nn.DataParallel(nn.Linear(fc1, fc2))
        self.fcE2 = nn.DataParallel(nn.Linear(fc1,fc2))
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder=nn.DataParallel(self.decoder)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1=nn.DataParallel(self.fcD1)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
    
class CNN_VAE_dropout(nn.Module):
#adapted from https://github.com/uhlerlab/cross-modal-autoencoders
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_dropout, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
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
        self.encoder=nn.DataParallel(self.encoder)
 
        self.fcE1 = nn.DataParallel(nn.Linear(fc1, fc2))
        self.fcE2 = nn.DataParallel(nn.Linear(fc1,fc2))
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder=nn.DataParallel(self.decoder)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1=nn.DataParallel(self.fcD1)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x)
#         print(h.size())
#         if torch.isnan(torch.sum(h)):
#             print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        dropout=nn.Dropout()
        h=dropout(h)
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
    
class CNN_VAE_hook(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_hook, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
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
 
        self.fcE1 = nn.Linear(fc1, fc2)
        self.fcE2 = nn.Linear(fc1,fc2)
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
#         self.bn_mean = nn.BatchNorm1d(fc2)
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def encode(self, x):
        h = self.encoder(x)
        hook = h.register_hook(self.activations_hook)
#         print(h.size())
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
    
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self, x):
        return self.encoder(x)

class CNN_VAE_split(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5,sharedChannels, fc1_shared,fc1_d,fc2_shared,fc2_d):
        super(CNN_VAE_split, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1_shared+fc1_d
        self.fc2=fc2_shared+fc2_d
        self.sharedChannels=sharedChannels
 
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
        self.encoder=nn.DataParallel(self.encoder)
 
        self.fcE1_shared = nn.DataParallel(nn.Linear(fc1_shared, fc2_shared))
        self.fcE2_shared = nn.DataParallel(nn.Linear(fc1_shared, fc2_shared))
        
        self.fcE1_d = nn.DataParallel(nn.Linear(fc1_d, fc2_d))
        self.fcE2_d = nn.DataParallel(nn.Linear(fc1_d, fc2_d))
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder=nn.DataParallel(self.decoder)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(self.fc2, self.fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1=nn.DataParallel(self.fcD1)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x)
        h_shared=h[:,:self.sharedChannels]
        h_d=h[:,self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1]*h_shared.size()[2]*h_shared.size()[3])
        h_d=h_d.view(-1,h_d.size()[1]*h_d.size()[2]*h_d.size()[3])
        h_shared_e1=self.fcE1_shared(h_shared)
        h_shared_e2=self.fcE2_shared(h_shared)
        h_d_e1=self.fcE1_d(h_d)
        h_d_e2=self.fcE2_d(h_d)
        return torch.cat((h_shared_e1,h_d_e1),dim=1), torch.cat((h_shared_e2,h_d_e2),dim=1)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar

class CNN_VAE_decode(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_decode, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder=nn.DataParallel(self.decoder)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1=nn.DataParallel(self.fcD1)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, z):
        res = self.decode(z)
        return res
    
class CNN_VAE_split_MI(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5,sharedChannels, fc1_shared,fc1_d,fc2_shared,fc2_d):
        super(CNN_VAE_split_MI, self).__init__()
        self.MI= MutualInformation(num_bins=64, sigma=0.1, normalize=True,device=1).cuda(1)
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1_shared+fc1_d
        self.fc2=fc2_shared+fc2_d
        self.sharedChannels=sharedChannels
 
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
        self.encoder=nn.DataParallel(self.encoder)
 
        self.fcE1_shared = nn.DataParallel(nn.Linear(fc1_shared, fc2_shared))
        self.fcE2_shared = nn.DataParallel(nn.Linear(fc1_shared, fc2_shared))
        
        self.fcE1_d = nn.DataParallel(nn.Linear(fc1_d, fc2_d))
        self.fcE2_d = nn.DataParallel(nn.Linear(fc1_d, fc2_d))
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder=nn.DataParallel(self.decoder)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(self.fc2, self.fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1=nn.DataParallel(self.fcD1)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x)
        h_shared=h[:,:self.sharedChannels]
        h_d=h[:,self.sharedChannels:]
        #MI
        h=h.cuda(1)
        mi=0
        for s_channel_idx in range(self.sharedChannels):
            for d_channel_idx in range(self.sharedChannels,self.hidden5):
#                 torch.cuda.empty_cache()
                mi+=torch.mean(self.MI(h[:,[s_channel_idx]],h[:,[d_channel_idx]]))
        mi=mi/(self.sharedChannels*(self.hidden5-self.sharedChannels))
        mi=mi.cuda(0)
        h_shared = h_shared.view(-1, h_shared.size()[1]*h_shared.size()[2]*h_shared.size()[3])
        h_d=h_d.view(-1,h_d.size()[1]*h_d.size()[2]*h_d.size()[3])
        h_shared_e1=self.fcE1_shared(h_shared)
        h_shared_e2=self.fcE2_shared(h_shared)
        h_d_e1=self.fcE1_d(h_d)
        h_d_e2=self.fcE2_d(h_d)
        
#         #MI
#         mi=0
#         for sidx in range(h_shared_e1.shape[1]):
#             for didx in range(h_d_e1.shape[1]):
#                 mi+=self.MI(h_shared_e1[:,sidx].reshape(1,1,-1,1),h_d_e1[:,didx].reshape(1,1,-1,1))
#                 mi+=self.MI(h_shared_e2[:,sidx].reshape(1,1,-1,1),h_d_e2[:,didx].reshape(1,1,-1,1))
        return torch.cat((h_shared_e1,h_d_e1),dim=1), torch.cat((h_shared_e2,h_d_e2),dim=1),mi.cuda(0)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar,mi = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar,mi    
    
class CNN_VAE_sharded(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_sharded, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
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
        self.encoder.cuda(0)
 
        self.fcE1 = nn.Linear(fc1, fc2)
        self.fcE2 = nn.Linear(fc1,fc2)
        self.fcE1.cuda(3)
        self.fcE2.cuda(3)
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder.cuda(0)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1.cuda(0)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x.cuda(0).float())
#         print(h.size())
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        h=h.cuda(3)
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z.cuda(0))
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
    
class FC_l3(nn.Module):
    def __init__(self,inputdim,fcdim1,fcdim2,fcdim3, num_classes,dropout=0.5):
        super(FC_l3, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim1, fcdim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim2, fcdim3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim3, num_classes),            
        )

    def forward(self, x):
        x = self.classifier(x)
        return x   
    
class FC_l2(nn.Module):
    def __init__(self,fcdim1,fcdim2,fcdim3, num_classes,dropout=0.5):
        super(FC_l2, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(fcdim1, fcdim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim3, num_classes),            
        )

    def forward(self, x):
        x = self.classifier(x)
        return x   