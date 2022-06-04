import torch
import torch.nn as nn
import torch.nn.functional as F
class encoder1(nn.Module):
    def __init__(self, num_layers=10, num_features=64, out_num=2):
        super(encoder1, self).__init__()
        
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)
        self.layers2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers4=nn.Sequential(nn.Linear(65536, 512), nn.ReLU(), nn.Linear(512, 256), 
                              nn.ReLU(), nn.Linear(256, out_num), nn.Sigmoid())

    

    def forward(self, inputs):
        output1 = self.layers(inputs)
        output1 = self.layers2(output1)
        output1 = self.layers3(output1)
        output2 = output1.reshape(output1.size(0), -1)
        output2 = self.layers4(output2)
        
        return output2