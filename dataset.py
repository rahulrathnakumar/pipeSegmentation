import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transformations import RandomFlip, Normalize

class PipelineDataset(Dataset):
    '''
    Create dataset class using train.txt and/or val.txt
    '''
    def __init__(self, root_dir, num_classes, image_set, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.num_classes = num_classes
        self.image_set = image_set
        self.txt = os.path.join(self.root_dir, self.image_set)
        
        # custom transform
        self.random_flip = RandomFlip() if self.image_set == 'train.txt' else None
        
        self.rgb = []
        self.labels = []
        self.depth = []
        self.mean_curvature = []
        self.gaussian_curvature = []
        self.normal = []
        
        self.read_image_list()
        
        # now we have the list of images, labels, depth, mean_curvature, gaussian_curvature, and normals
        
    
    def __len__(self):
        return len(self.rgb)
    
    def read_image_list(self):
        with open(self.txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # the txt file has the format: i.png, where i is the index 
                # of the image
                index = line.strip()
                self.rgb.append(os.path.join(self.root_dir, 'rgb', index))
                self.labels.append(os.path.join(self.root_dir, 'labels', index))
                self.depth.append(os.path.join(self.root_dir, 'depth', index))
                self.mean_curvature.append(os.path.join(self.root_dir, 'mean_curvature', index))
                self.gaussian_curvature.append(os.path.join(self.root_dir, 'gaussian_curvature', index))
                self.normal.append(os.path.join(self.root_dir, 'normal', index))
                
    def __getitem__(self, idx):
        # read the images
        rgb = Image.open(self.rgb[idx]).convert('RGB')
        labels = Image.open(self.labels[idx]).convert('L')
        depth = Image.open(self.depth[idx])
        mean_curvature = Image.open(self.mean_curvature[idx])
        gaussian_curvature = Image.open(self.gaussian_curvature[idx])
        normal = Image.open(self.normal[idx])
        
        # first apply the flipping transformations across all images
        if self.random_flip is not None:
            rgb, labels, depth, mean_curvature, gaussian_curvature, normal = self.random_flip(rgb, labels, depth, mean_curvature, gaussian_curvature, normal)
        
        
        # apply other transformations
        if self.transform:
            rgb = self.transform['rgb'](rgb)
            labels = self.transform['labels'](labels)
            depth = self.transform['depth'](depth)
            mean_curvature = self.transform['mean_curvature'](mean_curvature)
            gaussian_curvature = self.transform['gaussian_curvature'](gaussian_curvature)
            normal = self.transform['normal'](normal)
            
        # one hot encode the labels - labels is  (1, H, W) long tensor
        # convert labels to long tensor
        labels = labels.long().squeeze(0)
        
        
        labels = torch.nn.functional.one_hot(labels, 
                                             num_classes = self.num_classes).permute(2,0,1)
        
        
        
        
        
        return rgb, labels, depth, mean_curvature, gaussian_curvature, normal
                

if __name__ == '__main__':
    # Initialize your dataset with any required transformations (excluding normalization)
    
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        Normalize(min=0.0, max=1.0)
    ])
    
    dataset = PipelineDataset(root_dir='data/', image_set='train.txt', transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Function to calculate mean and std
    def calculate_mean_std(dataloader):
        mean_rgb = torch.zeros(3)
        std_rgb = torch.zeros(3)
        
        # for mean and gaussian curvature, they should be 1 channel and normalized between 0 and 1
        
        for images in dataloader:
            for i in range(3):  # Iterate over each channel for rgb
                mean[i] += images[0][:, i, :, :].mean()
                std[i] += images[0][:, i, :, :].std()

            
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std

    mean, std = calculate_mean_std(dataloader)
    print(f'Mean: {mean}')
    print(f'Std: {std}')
