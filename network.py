import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import torch.optim as optim

class Model(nn.Module):
    def __init__(self,):
        super(Model,self).__init__()
        #self.arr_drop = arr_drop
        self.lay1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=64,out_channels=32,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2,padding=0,stride=1),
                                 nn.Dropout(0.25)
                                 )
        self.lay2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=32,out_channels=16,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2,padding=0,stride=1),
                                 nn.Dropout(0.25)
                                 )
        self.num_fea = 16*14*14
        self.fc1 = nn.Sequential(nn.Linear(in_features=self.num_fea,out_features=128,bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(in_features=128,out_features=10,bias=False))
    def forward(self, x):
        x1 = self.lay1(x)
        x2 = self.lay2(x1)
        x3 = x2.view(-1,self.num_fea)
        x4 = self.fc1(x3)
        return F.softmax(x4, dim=1)
