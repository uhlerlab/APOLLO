import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class fc_decode_l4(nn.Module):   
    def __init__(self, input_feat_dim, latentSize,hidden_decoder, dropout,batchnorm=True):
        super(fc_decode_l4, self).__init__()
        self.fc0 = layers.FC(latentSize, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = batchnorm)
        self.fc1 = layers.FC(hidden_decoder, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = batchnorm)
        self.fc2 = layers.FC(hidden_decoder, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = batchnorm)
        self.fc3 = layers.FC(hidden_decoder, input_feat_dim, dropout, act = lambda x: x, batchnorm = False)

    def decode_X(self,z):
        output = self.fc3(self.fc2(self.fc1(self.fc0(z))))
        return output
    
    def forward(self, z):
        return self.decode_X(z)
    
class fc_encode_l4(nn.Module):   
    def __init__(self, input_feat_dim,hidden_1,sharedHidden,dHidden,sharedLatent,dLatent, dropout):
        super(fc_encode_l4, self).__init__()
        self.fc0 = layers.FC(input_feat_dim, hidden_1, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc1 = layers.FC(hidden_1, hidden_1, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc2_shared = layers.FC(hidden_1,sharedHidden, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc2_d=layers.FC(hidden_1,dHidden, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc3_shared = layers.FC(sharedHidden,sharedLatent, dropout, act = lambda x:x, batchnorm = False)
        self.fc3_d=layers.FC(dHidden,dLatent, dropout, act = lambda x:x, batchnorm = False)
        

    def encode_X(self,x):
        h=self.fc1(self.fc0(x))
        h_shared=self.fc2_shared(h)
        h_d=self.fc2_d(h)
        return self.fc3_shared(h_shared),self.fc3_d(h_d)
    
    def forward(self, x):
        return self.encode_X(x)