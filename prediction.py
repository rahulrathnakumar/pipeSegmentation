import torch
import torchvision
from torchvision import transforms
import os
from dataset import PipelineDataset
from networks.network import *
from networks.network_fusionv1 import *
from networks.network_fusionv2 import *
from torch.utils.data import DataLoader
from config import configDict
from utils import visualize_segmentation

from utils import Metrics

# Configuration and Model Loading
root_dir = configDict['root_dir']
val_txt = configDict['val_txt']
num_classes = 3
batch_size = configDict['batch_size']
num_workers = configDict['num_workers']
# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(network_name, num_classes, dim_sources):
    if network_name == 'BayesianFusionNetworkV1':
        model = BayesianFusionNetworkV1(num_classes=num_classes, dim_sources=dim_sources)
    elif network_name == 'BayesianFusionNetworkV2':
        model = BayesianFusionNetworkV2(num_classes=num_classes, dim_sources=dim_sources, is_BBB = False)
    elif network_name == 'BayesianFusionNetworkV2EdgeLayer':
        model = BayesianFusionNetworkV2EdgeLayer(num_classes=num_classes, dim_sources=dim_sources)
    elif network_name == 'BayesianSegmentationNetwork':
        model = BayesianSegmentationNetwork(num_classes=num_classes)
    else:
        raise ValueError(f'Network {network_name} not recognized')
    return model

# Create model and load weights
network_name = 'BayesianFusionNetworkV2'
model = create_model(network_name, num_classes, dim_sources= [3, 1, 3, 1, 1])
model.to(device)

# Load the best model
experiment_dir = 'models/rgbdncSegmentation_BayesianFusionNetworkV2_MCD_2024-03-24_18-44-19'
model_path = experiment_dir + '/best_model.pth'
model.load_state_dict(torch.load(model_path))

# Validation Dataset and DataLoader
val_transform = {
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

val_dataset = PipelineDataset(root_dir=root_dir, image_set=val_txt, num_classes=num_classes, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Prediction and Saving Results
results_dir = 'results/' + experiment_dir + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# model.eval()
# metrics = Metrics(num_classes=num_classes).to(device)
# with torch.no_grad():
#     for i, data in enumerate(val_loader):
#         rgb, labels, depth, mean_curvature, gaussian_curvature, normal = [d.to(device) for d in data]
        
#         labels_unhot = torch.argmax(labels, dim=1)
#         if network_name == 'BayesianSegmentationNetwork':
#             output = model(rgb)
#         else:
#             x3d = [depth]
#             output = model(rgb, x3d)

#         softmax = torch.nn.Softmax(dim=1)
#         y_pred = softmax(output)
#         y_pred = torch.argmax(y_pred, dim=1)
#         metrics.update(y_pred, labels_unhot.long())
        
#         metrics_dict = metrics.compute()
#         print(f'Batch {i+1}/{len(val_loader)}: {metrics_dict}')
        
#         # Loop through batch
#         for j in range(rgb.size(0)):
#             img = rgb[j].cpu()
#             pred = y_pred[j].cpu()
#             gt = torch.argmax(labels[j], dim=0).cpu()
#             # Visualization function similar to utils.visualize_segmentation
#             visualized = visualize_segmentation(img, pred, gt)
#             torchvision.utils.save_image(visualized, results_dir + f'image_{i * batch_size + j}.png')

# print("Prediction and saving completed.")

# Prediction with Uncertainty Estimation
model.eval()  # Set the model to evaluation mode
# Ensure dropout layers are active during evaluation for MC Dropout
for module in model.modules():
    if isinstance(module, nn.Dropout2d):
        module.train()

metrics = Metrics(num_classes=num_classes).to(device)
uncertainty = []  # List to store uncertainty values for each pixel

# Define the number of stochastic forward passes
num_forward_passes = 10

for i, data in enumerate(val_loader):
    rgb, labels, depth, mean_curvature, gaussian_curvature, normal = [d.to(device) for d in data]
    labels_unhot = torch.argmax(labels, dim=1)

    # Store the softmax probabilities from each forward pass
    softmax_outputs = []

    for _ in range(num_forward_passes):
        if network_name == 'BayesianSegmentationNetwork':
            output = model(rgb)
        else:
            x3d = [depth, normal, mean_curvature, gaussian_curvature]
            output = model(rgb, x3d)
        
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(output)
        softmax_outputs.append(y_pred.unsqueeze(0))

    # Stack and compute mean and variance
    softmax_stack = torch.cat(softmax_outputs, dim=0)  # Shape: [num_forward_passes, batch_size, num_classes, H, W]
    mean_softmax = torch.mean(softmax_stack, dim=0)  # Shape: [batch_size, num_classes, H, W]
    variance_softmax = torch.var(softmax_stack, dim=0)  # Shape: [batch_size, num_classes, H, W]
    
    # The predictive uncertainty can be estimated as the variance across the MC samples
    uncertainty.append(variance_softmax)
    
    # Use mean of softmax probabilities to compute metrics
    y_pred = torch.argmax(mean_softmax, dim=1)
    metrics.update(y_pred, labels_unhot.long())
    
    
    metrics_dict = metrics.compute()
    print(f'Batch {i+1}/{len(val_loader)}: {metrics_dict}')
    # print uncertainty
    print(f'Uncertainty: {variance_softmax.mean()}')
    
    # Save the mean prediction and uncertainty maps
    for j in range(rgb.size(0)):
        img = rgb[j].cpu()
        pred = y_pred[j].cpu()
        gt = torch.argmax(labels[j], dim=0).cpu()
        unc = variance_softmax[j].cpu()  # Uncertainty map for the j-th image in the batch
        # Visualization function similar to utils.visualize_segmentation
        visualized = visualize_segmentation(img, pred, gt)  # Modify this function as needed to include uncertainty
        torchvision.utils.save_image(visualized, results_dir + f'pred_{i * batch_size + j}.png')
        torchvision.utils.save_image(unc, results_dir + f'unc_{i * batch_size + j}.png')

print("Prediction and saving completed.")
