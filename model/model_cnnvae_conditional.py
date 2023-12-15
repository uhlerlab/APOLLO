import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_VAE_split_pIDemb(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, sharedChannels,
                 fc1_shared, fc1_d, fc2_shared, fc2_d,nProt,pID_type,pIDemb_size):
        super(CNN_VAE_split_pIDemb, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared + fc1_d
        self.fc2 = fc2_shared + fc2_d
        self.sharedChannels = sharedChannels

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

        self.fcE1_shared = nn.Linear(fc1_shared+pIDemb_size, fc2_shared)
        self.fcE2_shared = nn.Linear(fc1_shared+pIDemb_size, fc2_shared)

        self.fcE1_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)
        self.fcE2_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)

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
            nn.Linear(self.fc2+pIDemb_size, self.fc1),
            nn.ReLU(inplace=True),
        )
        #         self.bn_mean = nn.BatchNorm1d(fc2)

    def encode(self, x,pIDemb_batch):
        h = self.encoder(x)
        h_shared = h[:, :self.sharedChannels]
        h_d = h[:, self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1] * h_shared.size()[2] * h_shared.size()[3])
        h_d = h_d.view(-1, h_d.size()[1] * h_d.size()[2] * h_d.size()[3])
        #concatenate protein ID
        h_shared = torch.cat((h_shared,pIDemb_batch),dim=1)
        h_d = torch.cat((h_d,pIDemb_batch),dim=1)
        h_shared_e1 = self.fcE1_shared(h_shared)
        h_shared_e2 = self.fcE2_shared(h_shared)
        h_d_e1 = self.fcE1_d(h_d)
        h_d_e2 = self.fcE2_d(h_d)
        return torch.cat((h_shared_e1, h_d_e1), dim=1), torch.cat((h_shared_e2, h_d_e2), dim=1)

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
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1 / self.hidden5)), int(np.sqrt(self.fc1 / self.hidden5)))
        return self.decoder(h)

    def forward(self, x,pID):
        pIDemb_batch= self.pIDemb(pID)
        mu, logvar = self.encode(x,pIDemb_batch)
        z = self.reparameterize(mu, logvar)
        res = self.decode(torch.cat((z,pIDemb_batch),dim=1))
        return res, z, mu, logvar
    
class CNN_VAE_split_encode_pIDemb(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, sharedChannels,
                 fc1_shared, fc1_d, fc2_shared, fc2_d,nProt,pID_type,pIDemb_size):
        super(CNN_VAE_split_encode_pIDemb, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared + fc1_d
        self.fc2 = fc2_shared + fc2_d
        self.sharedChannels = sharedChannels

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

        self.fcE1_shared = nn.Linear(fc1_shared+pIDemb_size, fc2_shared)

        self.fcE1_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
        
    def encode(self, x,pIDemb_batch,gethook):
        h = self.encoder(x)
        if gethook:
            hook = h.register_hook(self.activations_hook)
        h_shared = h[:, :self.sharedChannels]
        h_d = h[:, self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1] * h_shared.size()[2] * h_shared.size()[3])
        h_d = h_d.view(-1, h_d.size()[1] * h_d.size()[2] * h_d.size()[3])
        #concatenate protein ID
        h_shared = torch.cat((h_shared,pIDemb_batch),dim=1)
        h_d = torch.cat((h_d,pIDemb_batch),dim=1)
        h_shared_e1 = self.fcE1_shared(h_shared)
        h_d_e1 = self.fcE1_d(h_d)
        return h_shared_e1, h_d_e1

    def forward(self, x,pID,gethook=False):
        pIDemb_batch= self.pIDemb(pID)
        h_shared_e1, h_d_e1 = self.encode(x,pIDemb_batch,gethook)
        return h_shared_e1, h_d_e1
    
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self, x):
        return self.encoder(x)
    
class CNN_VAE_split_encode_pIDemb_hook4(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, sharedChannels,
                 fc1_shared, fc1_d, fc2_shared, fc2_d,nProt,pID_type,pIDemb_size):
        super(CNN_VAE_split_encode_pIDemb_hook4, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared + fc1_d
        self.fc2 = fc2_shared + fc2_d
        self.sharedChannels = sharedChannels

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

        self.fcE1_shared = nn.Linear(fc1_shared+pIDemb_size, fc2_shared)

        self.fcE1_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
        
    def encode(self, x,pIDemb_batch,gethook):
        if gethook:
            h=x
            for i in range(11):
                h=self.encoder[i](h)
            hook = h.register_hook(self.activations_hook)
            for i in range(11,14):
                h=self.encoder[i](h)
        else:
            h = self.encoder(x)
        h_shared = h[:, :self.sharedChannels]
        h_d = h[:, self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1] * h_shared.size()[2] * h_shared.size()[3])
        h_d = h_d.view(-1, h_d.size()[1] * h_d.size()[2] * h_d.size()[3])
        #concatenate protein ID
        h_shared = torch.cat((h_shared,pIDemb_batch),dim=1)
        h_d = torch.cat((h_d,pIDemb_batch),dim=1)
        h_shared_e1 = self.fcE1_shared(h_shared)
        h_d_e1 = self.fcE1_d(h_d)
        return h_shared_e1, h_d_e1

    def forward(self, x,pID,gethook=False):
        pIDemb_batch= self.pIDemb(pID)
        h_shared_e1, h_d_e1 = self.encode(x,pIDemb_batch,gethook)
        return h_shared_e1, h_d_e1
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        h=x
        for i in range(11):
            h=self.encoder[i](h)
        return h
    
class CNN_VAE_split_encode_pIDemb_hook3(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, sharedChannels,
                 fc1_shared, fc1_d, fc2_shared, fc2_d,nProt,pID_type,pIDemb_size):
        super(CNN_VAE_split_encode_pIDemb_hook3, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared + fc1_d
        self.fc2 = fc2_shared + fc2_d
        self.sharedChannels = sharedChannels

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

        self.fcE1_shared = nn.Linear(fc1_shared+pIDemb_size, fc2_shared)

        self.fcE1_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
        
    def encode(self, x,pIDemb_batch,gethook):
        if gethook:
            h=x
            for i in range(8):
                h=self.encoder[i](h)
            hook = h.register_hook(self.activations_hook)
            for i in range(8,14):
                h=self.encoder[i](h)
        else:
            h = self.encoder(x)
        h_shared = h[:, :self.sharedChannels]
        h_d = h[:, self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1] * h_shared.size()[2] * h_shared.size()[3])
        h_d = h_d.view(-1, h_d.size()[1] * h_d.size()[2] * h_d.size()[3])
        #concatenate protein ID
        h_shared = torch.cat((h_shared,pIDemb_batch),dim=1)
        h_d = torch.cat((h_d,pIDemb_batch),dim=1)
        h_shared_e1 = self.fcE1_shared(h_shared)
        h_d_e1 = self.fcE1_d(h_d)
        return h_shared_e1, h_d_e1

    def forward(self, x,pID,gethook=False):
        pIDemb_batch= self.pIDemb(pID)
        h_shared_e1, h_d_e1 = self.encode(x,pIDemb_batch,gethook)
        return h_shared_e1, h_d_e1
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        h=x
        for i in range(8):
            h=self.encoder[i](h)
        return h
    
class CNN_VAE_split_encode(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, sharedChannels,
                 fc1_shared, fc1_d, fc2_shared, fc2_d):
        super(CNN_VAE_split_encode, self).__init__()

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared + fc1_d
        self.fc2 = fc2_shared + fc2_d
        self.sharedChannels = sharedChannels

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

        self.fcE1_shared = nn.Linear(fc1_shared, fc2_shared)

        self.fcE1_d = nn.Linear(fc1_d, fc2_d)

    def encode(self, x):
        h = self.encoder(x)
        h_shared = h[:, :self.sharedChannels]
        h_d = h[:, self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1] * h_shared.size()[2] * h_shared.size()[3])
        h_d = h_d.view(-1, h_d.size()[1] * h_d.size()[2] * h_d.size()[3])
        h_shared_e1 = self.fcE1_shared(h_shared)
        h_d_e1 = self.fcE1_d(h_d)
        return h_shared_e1, h_d_e1

    def forward(self, x):
        h_shared_e1, h_d_e1 = self.encode(x)
        return h_shared_e1, h_d_e1
    
class CNN_VAE_encode(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5,
                 fc1_shared, fc2_shared):
        super(CNN_VAE_encode, self).__init__()


        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared 
        self.fc2 = fc2_shared 

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

        self.fcE1_shared = nn.Linear(fc1_shared, fc2_shared)


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, h.size()[1] * h.size()[2] * h.size()[3])
        #concatenate protein ID
        h_e1 = self.fcE1_shared(h)
        return h_e1

    def forward(self, x):
        h_e1= self.encode(x)
        return h_e1
    
class CNN_VAE_encode_pIDemb(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5,
                 fc1_shared, fc2_shared,nProt,pID_type,pIDemb_size):
        super(CNN_VAE_encode_pIDemb, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared 
        self.fc2 = fc2_shared 

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

        self.fcE1_shared = nn.Linear(fc1_shared+pIDemb_size, fc2_shared)


    def encode(self, x,pIDemb_batch):
        h = self.encoder(x)
        h = h.view(-1, h.size()[1] * h.size()[2] * h.size()[3])
        #concatenate protein ID
        h= torch.cat((h,pIDemb_batch),dim=1)
        h_e1 = self.fcE1_shared(h)
        return h_e1

    def forward(self, x,pID):
        pIDemb_batch= self.pIDemb(pID)
        h_e1= self.encode(x,pIDemb_batch)
        return h_e1


class CNN_VAE_decode_pIDemb(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1, fc2,pIDemb_size,applySigmoid=True):
        super(CNN_VAE_decode_pIDemb, self).__init__()

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1
        self.fc2 = fc2

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
        )
        self.act=nn.Sigmoid()
        self.applySigmoid=applySigmoid

        self.fcD1 = nn.Sequential(
            nn.Linear(fc2+pIDemb_size, fc1),
            nn.ReLU(inplace=True),
        )

    #         self.bn_mean = nn.BatchNorm1d(fc2)

    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1 / self.hidden5)), int(np.sqrt(self.fc1 / self.hidden5)))
        if self.applySigmoid:
            return self.act(self.decoder(h))
        else:
            return self.decoder(h)

    def forward(self, z,pIDemb_batch):
        res = self.decode(torch.cat((z,pIDemb_batch),dim=1))
        return res

class CNN_VAE_split_pIDemb_dOnly(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, sharedChannels,
                 fc1_shared, fc1_d, fc2_shared, fc2_d,nProt,pID_type,pIDemb_size):
        super(CNN_VAE_split_pIDemb_dOnly, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1_shared + fc1_d
        self.fc2 = fc2_shared + fc2_d
        self.sharedChannels = sharedChannels

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

        self.fcE1_shared = nn.Linear(fc1_shared, fc2_shared)
        self.fcE2_shared = nn.Linear(fc1_shared, fc2_shared)

        self.fcE1_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)
        self.fcE2_d = nn.Linear(fc1_d+pIDemb_size, fc2_d)

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
            nn.Linear(self.fc2+pIDemb_size, self.fc1),
            nn.ReLU(inplace=True),
        )
        #         self.bn_mean = nn.BatchNorm1d(fc2)

    def encode(self, x,pIDemb_batch):
        h = self.encoder(x)
        h_shared = h[:, :self.sharedChannels]
        h_d = h[:, self.sharedChannels:]
        h_shared = h_shared.view(-1, h_shared.size()[1] * h_shared.size()[2] * h_shared.size()[3])
        h_d = h_d.view(-1, h_d.size()[1] * h_d.size()[2] * h_d.size()[3])
        #concatenate protein ID
        h_d = torch.cat((h_d,pIDemb_batch),dim=1)
        h_shared_e1 = self.fcE1_shared(h_shared)
        h_shared_e2 = self.fcE2_shared(h_shared)
        h_d_e1 = self.fcE1_d(h_d)
        h_d_e2 = self.fcE2_d(h_d)
        return torch.cat((h_shared_e1, h_d_e1), dim=1), torch.cat((h_shared_e2, h_d_e2), dim=1)

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
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1 / self.hidden5)), int(np.sqrt(self.fc1 / self.hidden5)))
        return self.decoder(h)

    def forward(self, x,pID):
        pIDemb_batch= self.pIDemb(pID)
        mu, logvar = self.encode(x,pIDemb_batch)
        z = self.reparameterize(mu, logvar)
        res = self.decode(torch.cat((z,pIDemb_batch),dim=1))
        return res, z, mu, logvar
    
class CNN_VAE_pIDemb(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5,fc1, fc2, nProt,pID_type,pIDemb_size):
        super(CNN_VAE_pIDemb, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1
        self.fc2 = fc2

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

        self.fcE1 = nn.Linear(fc1+pIDemb_size, fc2)
        self.fcE2 = nn.Linear(fc1+pIDemb_size, fc2)


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
            nn.Linear(self.fc2+pIDemb_size, self.fc1),
            nn.ReLU(inplace=True),
        )
        #         self.bn_mean = nn.BatchNorm1d(fc2)

    def encode(self, x,pIDemb_batch):
        h = self.encoder(x)
        h = h.view(-1, h.size()[1] * h.size()[2] * h.size()[3])
        #concatenate protein ID
        h = torch.cat((h,pIDemb_batch),dim=1)
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
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1 / self.hidden5)), int(np.sqrt(self.fc1 / self.hidden5)))
        return self.decoder(h)

    def forward(self, x,pID):
        pIDemb_batch= self.pIDemb(pID)
        mu, logvar = self.encode(x,pIDemb_batch)
        z = self.reparameterize(mu, logvar)
        res = self.decode(torch.cat((z,pIDemb_batch),dim=1))
        return res, z, mu, logvar
    
class CNN_AE_pIDemb(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5,fc1, fc2, nProt,pID_type,pIDemb_size):
        super(CNN_AE_pIDemb, self).__init__()

        if pID_type == 'randInit':
            self.pIDemb = torch.nn.Embedding(nProt, pIDemb_size)

        self.nc = nc
        self.hidden5 = hidden5
        self.fc1 = fc1
        self.fc2 = fc2

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

        self.fcE1 = nn.Linear(fc1+pIDemb_size, fc2)


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
            nn.Linear(self.fc2, self.fc1),
            nn.ReLU(inplace=True),
        )
        #         self.bn_mean = nn.BatchNorm1d(fc2)

    def encode(self, x,pIDemb_batch):
        h = self.encoder(x)
        h = h.view(-1, h.size()[1] * h.size()[2] * h.size()[3])
        #concatenate protein ID
        h = torch.cat((h,pIDemb_batch),dim=1)
        return self.fcE1(h)


    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1 / self.hidden5)), int(np.sqrt(self.fc1 / self.hidden5)))
        return self.decoder(h)

    def forward(self, x,pID):
        pIDemb_batch= self.pIDemb(pID)
        z = self.encode(x,pIDemb_batch)
        res = self.decode(z)
        return res, z
    


