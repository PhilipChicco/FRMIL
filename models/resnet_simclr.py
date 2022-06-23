import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimCLR(nn.Module):
    
    def __init__(self,args):
        super(SimCLR, self).__init__()
        
        model     = models.resnet18(pretrained=True)
        num_feats = model.fc.in_features
        model.fc  = nn.Identity()
        
        self.features = nn.Sequential(model)
        self.head = nn.Sequential(
            nn.Linear(num_feats, num_feats),
            nn.ReLU(),
            nn.Linear(num_feats, args.out_dim)
        )
        self.mode = 0
        
    def forward(self, x):
        out  = self.features(x)
        if self.mode == 1: return out
        out = self.head(out)
        return out 