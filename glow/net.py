import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .actnorm import ActNorm


# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

class NN(nn.Module):
    def __init__(self, channels_in, channels_out, device, init_last_zeros=False):
        super(NN, self).__init__()
        self.conv1    = nn.Conv2d(channels_in,512,kernel_size=(3,3),stride=1,padding=1,bias=True)
        self.actnorm1 = ActNorm(512,device)
                                
        self.conv2    = nn.Conv2d(512,512,kernel_size=(1,1),stride=1,padding=0,bias=True)
        self.actnorm2 = ActNorm(512,device)
        
        self.conv3    = nn.Conv2d(512,channels_out,kernel_size=(3,3),stride=1,padding=1,bias=True)        
        self.logs     = nn.Parameter(torch.zeros(channels_out, 1, 1))
        
        
        
        # initializing
        with torch.no_grad():
            
            nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv1.bias)
            
            nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv2.bias) 
            
            if init_last_zeros:
                nn.init.zeros_(self.conv3.weight) # last layer initialized with zeros
            else:
                nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv3.bias)
            
        # to device
        self.to(device)
            
    def forward(self, x):
        x = self.conv1(x)
        x, _, _ = self.actnorm1(x, logdet=0, reverse=False) 
        x = F.relu(x)
        x = self.conv2(x)
        x, _, _ = self.actnorm2(x, logdet=0, reverse=False) 
        x = F.relu(x)
        x = self.conv3(x)
        # x = x * torch.exp(self.logs * 3)
        logs_c = torch.clamp(self.logs, -2., 2.)   # 软约束
        x = x * torch.exp(logs_c * 3)

        return x
    
    
    
