import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Grad_CAM:
    def __init__(self, model, model_path, target_layer):
        model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model = model
        self.model.eval()
        self.feature_map = None
        self.gradients = None

        def forward_hook(model, input, output):
            self.feature_map = output.to('cpu').detach()
            
        def backward_hook(model, grad_inputs, grad_outputs):
            self.gradients = grad_outputs[0].to('cpu').detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, label:int = None):
        b, _, h, w = input.size()
        if b > 1:
            input = input[0:1]
            
        logit = self.model(input)
        self.model.zero_grad()

        if label is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, label].squeeze()
            
        score.backward()
        b_, c_, _, _ = self.gradients.size()
        alpha = self.gradients.view(b_, c_, -1).mean(2)
        weights = alpha.view(b_, c_, 1, 1)
        
        saliency_map = F.relu((weights * self.feature_map).sum(1, keepdim = True))
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_map = saliency_map.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        cmap = plt.get_cmap('jet')
        heatmap = (cmap(saliency_map))[:, :, :, :3].squeeze(2)

        img = input.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        grad_cam_img = img * 0.5 + heatmap * 0.5

        return grad_cam_img

    def __call__(self, input, label = None):
        return self.forward(input, label)
