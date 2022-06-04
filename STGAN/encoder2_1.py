import torch
import torch.nn as nn
import torch.nn.functional as F
class encoder2(nn.Module):
    def __init__(self, num_layers=10, num_features=64, out_num=1):
        super(encoder2, self).__init__()
        
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)
        self.layer_2=nn.Sequential(nn.Conv2d(num_features, out_num, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(out_num),
                                        nn.ReLU(inplace=True))
        
    

    def forward(self, inputs):
        output = self.layers(inputs)
        output = self.layer_2(output)
        
        return output