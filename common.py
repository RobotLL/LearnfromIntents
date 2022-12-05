import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.fc1 = nn.Linear(15, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5)
        
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, images):
        images = F.relu(self.conv1x1(images))
        images = images.mean((2,3))
        
        x = torch.cat([x,images],1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# batch = 128
# C=1
# H=W=60
# images = torch.rand((batch,C,H,W))
# x = torch.rand((batch,14))
# a = ClassifierNet()
# a.forward(x, images)
# class ClassifierNet(nn.Module):
#     def __init__(self):
#         super(ClassifierNet, self).__init__()
#         self.fc1 = nn.Linear(13, 256)
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, 5)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x