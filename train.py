import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torchvision

from sklearn.model_selection import KFold


from torch.utils.tensorboard import SummaryWriter

from networks.network import *

from dataset import PipelineDataset

from losses.focal_loss import FocalLoss
from losses.lovasz_softmax import LovaszSoftmaxV1
from losses.activeboundaryloss import ABL

from config import configDict

from utils import Metrics, median_frequency_balancing

import os

root_dir = configDict['root_dir']
train_txt = configDict['train_txt']
val_txt = configDict['val_txt']

batch_size = configDict['batch_size']
num_workers = configDict['num_workers']
num_epochs = configDict['num_epochs']
learning_rate = configDict['learning_rate']
weight_decay = configDict['weight_decay']
momentum = configDict['momentum']
scheduler_step_size = configDict['scheduler_step_size']
scheduler_gamma = configDict['scheduler_gamma']
experiment_name = configDict['experiment_name']

# Create directories for saving the model and logs

model_dir = 'models/' 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

import datetime 
now = datetime.datetime.now()
experiment_name = experiment_name + now.strftime("_%Y-%m-%d_%H-%M-%S")
experiment_dir = model_dir + experiment_name

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# create logs directory
logs_dir = experiment_dir + '/logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# create tensorboard writer
writer = SummaryWriter(logs_dir)


# set the seed
seed = 42
torch.manual_seed(seed)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the model
model = BayesianSegmentationNetwork(num_classes=3)
import torchsummary as summary
x_rgb = torch.randn(1, 3, 102, 180)
x_depth = torch.randn(1, 1, 102, 180)
x_normal = torch.randn(1, 3, 102, 180)
x_mean_curvature = torch.randn(1, 1, 102, 180)
x_gaussian_curvature = torch.randn(1, 1, 102, 180)

model_summary = summary.summary(model, input_size=[x_rgb.shape, [x_depth.shape, x_normal.shape, x_mean_curvature.shape, x_gaussian_curvature.shape]])
model_summary_string = str(model_summary)
# summary_string
# print this summary to config.txt too
with open(experiment_dir + '/config.txt', 'a', encoding = 'utf-8') as f:
    f.write(model_summary_string)

# move the model to the device
model.to(device)

# create the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# create the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

# create the dataset
train_transform = {
    'rgb': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4935, 0.4934, 0.4971], std=[0.1189, 0.1203, 0.1278]),
]),
    'depth': transforms.Compose([
    transforms.ToTensor(),
    # Normalize(min=0.0, max = 1.0)
]),
    'mean_curvature': transforms.Compose([
    transforms.ToTensor(),
]),
    'gaussian_curvature': transforms.Compose([
    transforms.ToTensor(),
]),
    'normal': transforms.Compose([
    transforms.ToTensor(),
]),
    'labels': transforms.Compose([
    transforms.PILToTensor()
])
}

train_dataset = PipelineDataset(root_dir = root_dir, image_set = train_txt, num_classes = 3, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = PipelineDataset(root_dir = root_dir, image_set = val_txt, num_classes = 3, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

dataset = {
    'train': train_dataset,
    'val': val_dataset
}

dataloader = {
    'train': train_loader,
    'val': val_loader
}

# weight is median frequency balancing
weights = median_frequency_balancing(dataset['train'])
weights = weights.to(device)
# broadcast the weights to match [B, C, H, W]

# focal loss for segmentation
sup_loss = nn.CrossEntropyLoss(weight=weights)
# iou loss
iou_loss = LovaszSoftmaxV1()
# abl loss
abl_loss = ABL(device = device)



# init metrics object
metrics = Metrics(num_classes = 3).to(device)

# training loop
best_loss = float('inf')
best_f1_defectsOnly = 0.0
for epoch in range(num_epochs):
    model.train()
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        for i, data in enumerate(dataloader[phase]):
            # this expt only uses rgb
            rgb = data[0].to(device)
            labels = data[1].to(device)
            labels = labels.float()
            labels_unhot = torch.argmax(labels, dim=1).long()
            output = model(rgb)
            
            sup_ = sup_loss(output, labels_unhot)
            iou_ = iou_loss(output, labels_unhot)[0]
            abl_ = abl_loss(output, labels_unhot)
            
            if abl_ is None:
                loss = sup_ + iou_
            else:
                loss = sup_ + iou_ + abl_
            
            # metrics
            softmax = nn.Softmax(dim=1)
            y_pred = softmax(output)
            y_pred = torch.argmax(y_pred, dim=1)
            metrics.update(y_pred, labels_unhot.long())
            
            
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataset[phase])
        
        # write the loss to tensorboard based on phase
        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        
        # write the metrics to tensorboard
        metrics_dict = metrics.compute()
        for key, value in metrics_dict.items():
            if key == 'class_f1':
                for i, class_f1 in enumerate(value):
                    # add scalar with key, class_{i} and phase
                    writer.add_scalar(f'{key + str(i)}/{phase}', class_f1, epoch)
            else:
                writer.add_scalar(f'{key}/{phase}', value, epoch)

        # write the image, prediction and ground truth to tensorboard
        # select 3 random images
        if phase == 'train':
            indices = torch.randint(0, len(rgb), (3,))
            gray = rgb[:, 0, :, :].unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)
            labels_unhot = labels_unhot.unsqueeze(1)
            # make a grid of image, prediction and ground truth
            for idx in indices:
                # rgb to grayscale
                grid = torchvision.utils.make_grid([gray[idx].float(), y_pred[idx].float(), labels_unhot[idx].float()], nrow=3)
                writer.add_image(f'Image/{idx}', grid, epoch)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Phase: {phase}, Loss: {epoch_loss}, Metrics: {metrics_dict}')
        writer.flush()
        metrics.reset()

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), experiment_dir + '/best_model.pth')
        # save the best_model based on f1 score for defects only
        if phase == 'val' and torch.mean(metrics_dict['class_f1'][1:]) > best_f1_defectsOnly:
            best_f1_defectsOnly = torch.mean(metrics_dict['class_f1'][1:]) 
            torch.save(model.state_dict(), experiment_dir + '/best_model_f1_defectsOnly.pth')
        
    scheduler.step()

