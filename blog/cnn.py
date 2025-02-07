import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Net(nn.Module):
    def __init__(self, model_name, global_pool = "avg", num_classes = 10, pretrained = True):
        super (Net, self).__init__()
        if global_pool == False:
            pooling = ""
        else:
            pooling = "avg"
        self.model = timm.create_model(model_name,
                                       num_classes = num_classes,
                                       pretrained = pretrained,
                                       in_chans = 1,
                                       global_pool = pooling
                                      )
        
    def forward(self, input):
        out = self.model(input)
        return out
