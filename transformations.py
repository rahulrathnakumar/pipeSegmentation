import torch
from torchvision.transforms import functional as TF

class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, rgb, depth, mean_curvature, gaussian_curvature, normal, label):
        # Apply the same flip to all images
        if torch.rand(1).item() > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
            mean_curvature = TF.hflip(mean_curvature)
            gaussian_curvature = TF.hflip(gaussian_curvature)
            normal = TF.hflip(normal)
            label = TF.hflip(label)

        if torch.rand(1).item() > 0.5:
            rgb = TF.vflip(rgb)
            depth = TF.vflip(depth)
            mean_curvature = TF.vflip(mean_curvature)
            gaussian_curvature = TF.vflip(gaussian_curvature)
            normal = TF.vflip(normal)
            label = TF.vflip(label)

        return rgb, depth, mean_curvature, gaussian_curvature, normal, label

class Normalize():
    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def __call__(self, tensor):
        return (tensor - self.min) / (self.max - self.min)
