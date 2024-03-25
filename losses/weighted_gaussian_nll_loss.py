import torch
import torch.nn as nn

class WeightedGaussianNLLLoss(nn.Module):
    def __init__(self, class_weights, reduction = 'mean'):
        super(WeightedGaussianNLLLoss, self).__init__()
        self.class_weights = class_weights
        self.num_classes = len(class_weights)
        self.reduction = reduction
    
    def forward(self, mean, var, target):
        # Split input into mean and variance
        # one-hot encode the target
        target = torch.nn.functional.one_hot(target, num_classes = self.num_classes).permute(0, 3, 1, 2)
        
        # Ensure positive variance
        var = torch.exp(var)  
        
        # sample from the Gaussian distribution
        eps = torch.randn_like(mean)
        sampled_logits = mean + eps * var
        
        # softmax
        probs = torch.nn.functional.softmax(sampled_logits, dim=1)
        
        nll_loss = -torch.log(probs.gather(1, target)) + 1e-9
        
        # # Calculate Gaussian NLL
        # Apply class weights
        for i in range(self.num_classes):
            nll_loss[:, i] *= self.class_weights[i]
        
        if self.reduction == 'mean':
            nll_loss = nll_loss.mean()
        elif self.reduction == 'sum':
            nll_loss = nll_loss.sum()        
        print(nll_loss)
        
        return nll_loss

if __name__ == '__main__':
    # Test the loss function
    class_weights = [1, 1, 1]
    loss_fn = WeightedGaussianNLLLoss(class_weights)

    mean = torch.randn(1, 3, 102, 180)
    var = torch.randn(1, 3, 102, 180)
    target = torch.randint(0, 3, (1, 102, 180))

    # mean, var with gradients
    mean = nn.Parameter(mean, requires_grad=True)
    var = nn.Parameter(var, requires_grad=True)
    

    for i in range(100):
        loss = loss_fn(mean, var, target)
        loss.backward()
        mean.data -= 0.01 * mean.grad
        var.data -= 0.01 * var.grad
    
    
    print(loss)