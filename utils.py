import torch
import torchmetrics

    

class Metrics(torchmetrics.MetricCollection):
    def __init__(self, num_classes):
        super(Metrics, self).__init__({
            'f1': torchmetrics.F1Score(task = "multiclass", num_classes=num_classes, average = 'macro'),
            'class_f1': torchmetrics.F1Score(task = "multiclass", num_classes=num_classes, average = None)
        }
        )
        
    def update(self, y_pred, y_true):
        self['f1'](y_pred, y_true)
        self['class_f1'](y_pred, y_true)
    
    def compute(self):
        return {
            'f1': self['f1'].compute(),
            'class_f1': self['class_f1'].compute()
        }
    def reset(self) -> None:
        return super().reset()


def sigmoid_ramp_up(current_epoch, sigmoid_ramp_up_length=100):
    """
    Sigmoid ramp-up function for gradually increasing the weight of the ABL-IOU losses.

    :param current_epoch: Current epoch
    :param sigmoid_ramp_up_length: Length of the sigmoid ramp-up
    :return: Weight of the ABL-IOU losses
    """
    if current_epoch < sigmoid_ramp_up_length:
        return 1.0 - torch.exp(torch.tensor(-5.0 * (1.0 - current_epoch / sigmoid_ramp_up_length) ** 2))
    else:
        return 1.0

def median_frequency_balancing(dataset):
    # compute the class frequencies
    class_frequencies = torch.zeros(dataset.num_classes)
    for i, data in enumerate(dataset):
        # remove one hot encoding of data[1]
        class_frequencies += torch.bincount(torch.argmax(data[1], dim=0).reshape(-1), minlength=dataset.num_classes)
        
    # compute the median frequency
    total_samples = class_frequencies
    median_frequency = torch.median(class_frequencies)
    
    # compute the class weights
    class_weights = median_frequency / total_samples
    return class_weights

import torch
import torchvision.transforms.functional as TF

def visualize_segmentation(rgb, pred, gt, num_classes=3):
    """
    Visualizes the RGB image, prediction and ground truth.

    :param rgb: Tensor of the RGB image
    :param pred: Tensor of the predicted segmentation
    :param gt: Tensor of the ground truth segmentation
    :param num_classes: Number of classes in the segmentation
    :return: Tensor of the concatenated image
    """
    # Normalize the RGB image to [0, 1] for display
    rgb_normalized = TF.normalize(rgb, mean=[-0.4935/0.1189, -0.4934/0.1203, -0.4971/0.1278], std=[1/0.1189, 1/0.1203, 1/0.1278])

    # Convert predictions and ground truths to color maps
    color_mapping = torch.tensor([
        [0, 0, 0],       # Background
        [255, 0, 0],     # Class 1 (e.g., Crack)
        [0, 255, 0]      # Class 2 (e.g., Corrosion)
    ]) / 255.0  # Normalize to [0, 1]

    pred_color = color_mapping[pred]
    gt_color = color_mapping[gt]
    
    pred_color = pred_color.permute(2, 0, 1)
    gt_color = gt_color.permute(2, 0, 1)

    # Concatenate images horizontally
    concatenated = torch.cat([rgb_normalized, pred_color, gt_color], dim=2)  # Concatenate along width

    return concatenated

if __name__ == '__main__':
    # test the metrics class on a dummy dataset
    
    metrics = Metrics(num_classes = 3)
    
    # dataset is an image dataset with 3 classes, and we need to predict pixelwise
    # labels for each image
    # y_pred is the predicted labels, and y_true is the ground truth labels

    x = torch.randn(10, 3, 100, 100)
    y_true = torch.randint(0, 3, (10, 100, 100))
    
    dataset = torch.utils.data.TensorDataset(x, y_true)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    epochs = 10
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            y_pred, y_true = data
            metrics.update(y_pred, y_true)
            
        print('Epoch:', epoch, metrics.compute())
        metrics.reset()