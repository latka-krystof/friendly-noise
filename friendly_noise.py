import torch
import torch.nn.functional as F
import random
import numpy as np

def generate_friendly_noise(
        model,
        trainloader,
        device,
        friendly_epochs,
        mu,
        friendly_lr,
        friendly_momentum=0.9,
        nesterov=True,
        friendly_steps=None,
        clamp_min=-32/255,
        clamp_max=32/255,
        return_preds=False,
        loss_fn='KL'):


    if loss_fn == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_fn == 'KL':
        criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    else:
        raise ValueError("No such loss fn")

    model.eval()
    transform = trainloader.dataset.transform
    trainloader.dataset.transform = None

    dataset_size = len(trainloader.dataset)
    friendly_noise = torch.zeros((dataset_size, 3, 32, 32))
    if return_preds:
        preds = torch.zeros((dataset_size, 10))

    for batch_idx, (inputs, target, p, idx) in enumerate(trainloader):
        inputs = inputs.cuda()
        init = (torch.rand(*(inputs.shape)) - 0.5) * 2 * 8/255
        eps = torch.autograd.Variable(init.to(device), requires_grad=True)
        optimizer = torch.optim.SGD([eps], lr=friendly_lr, momentum=friendly_momentum, nesterov=nesterov)
        
        # Clear any previous optimizer state to save memory
        optimizer.zero_grad(set_to_none=True)

        if friendly_steps is None:
            friendly_steps = [friendly_epochs // 2, friendly_epochs // 4 * 3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, friendly_steps)

        images_normalized = torch.stack([transform(x) for x in inputs], dim=0).to(device)
        output_original = model(images_normalized)
        if loss_fn == 'KL':
            output_original = F.log_softmax(output_original, dim=1).detach()
        else:
            output_original = output_original.detach()

        for friendly_epoch in range(friendly_epochs):
            eps_clamp = torch.clamp(eps, clamp_min, clamp_max)
            perturbed = torch.clamp(inputs + eps_clamp, 0, 1)
            perturbed_normalized = torch.stack(
                [transform(p) for p in perturbed], dim=0)
            output_perturb = model(perturbed_normalized)
            if loss_fn == 'KL':
                output_perturb = F.log_softmax(output_perturb, dim=1)

            emp_risk, constraint = friendly_loss(output_perturb, output_original, eps_clamp, criterion)
            loss = emp_risk - mu * constraint

            optimizer.zero_grad()
            loss.backward()
            model.zero_grad()
            optimizer.step()

            print(f"Friendly noise gen --- Batch {batch_idx} / {len(trainloader)}  "
                  f"Epoch: {friendly_epoch}  -- Max: {torch.max(eps):.5f}  Min: {torch.min(eps):.5f}  "
                  f"Mean (abs): {torch.abs(eps).mean():.5f}  Mean: {torch.mean(eps):.5f}  "
                  f"Mean (abs) Clamp: {torch.abs(eps_clamp).mean():.5f}  Mean Clamp: {torch.mean(eps_clamp):.5f}  "
                  f"emp_risk: {emp_risk:.3f}  constraint: {constraint:.3f}", end='\r', flush=True)
            
            # Clean up intermediate tensors from this iteration
            del perturbed, perturbed_normalized, output_perturb, eps_clamp, emp_risk, constraint, loss

        friendly_noise[idx] = eps.cpu().detach()
        if return_preds:
           preds[idx] = output_original.cpu()
        
        # Clean up batch-level tensors to free GPU memory
        del eps, optimizer, scheduler
        del inputs, init, images_normalized, output_original
        
        # Periodically clear cache every 10 batches to prevent accumulation
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    friendly_noise = torch.clamp(friendly_noise, clamp_min, clamp_max)

    trainloader.dataset.transform = transform

    if return_preds:
        return friendly_noise, preds
    else:
        return friendly_noise


def friendly_loss(output, target, eps, criterion):
    emp_risk = criterion(output, target)
    constraint = torch.mean(torch.square(eps))
    return emp_risk, constraint


def generator_loss(model, generator, inputs_normalized, targets, criterion, lambda_mag, device):
    """
    Compute loss for training the noise generator.
    
    The generator loss aims to:
    1. Minimize the difference between model predictions on clean vs noisy inputs
    2. Encourage non-zero noise through a negative magnitude constraint
    
    Args:
        model: The classifier model (frozen during generator training)
        generator: The noise generator network
        inputs_normalized: Already-transformed input images (normalized + augmented from dataloader)
        targets: True labels
        criterion: Loss function (e.g., KLDivLoss or MSELoss)
        lambda_mag: Weight for the negative magnitude constraint term
        device: Device to run computations on
        
    Returns:
        loss: Total generator loss
        emp_risk: Empirical risk component
        constraint: Magnitude constraint component
    """
    model.eval()  # Ensure model is in eval mode
    generator.train()
    
    # Inputs are already normalized and augmented from the dataloader
    inputs_normalized = inputs_normalized.to(device)
    
    # Get original model output (detached)
    with torch.no_grad():
        output_original = model(inputs_normalized)
        if isinstance(criterion, torch.nn.KLDivLoss):
            output_original = F.log_softmax(output_original, dim=1).detach()
        else:
            output_original = output_original.detach()
    
    # Generate noise from generator (pass normalized inputs)
    noise = generator(inputs_normalized)
    
    # Apply noise to normalized inputs
    inputs_noisy = torch.clamp(inputs_normalized + noise, -3, 3)  # Reasonable bounds for normalized images
    
    # Get model output on noisy inputs
    output_noisy = model(inputs_noisy)
    if isinstance(criterion, torch.nn.KLDivLoss):
        output_noisy = F.log_softmax(output_noisy, dim=1)
    
    # Compute empirical risk (difference between clean and noisy predictions)
    emp_risk = criterion(output_noisy, output_original)
    
    # Compute magnitude constraint (negative to encourage non-zero noise)
    magnitude = torch.mean(torch.square(noise))
    
    # Total loss: empirical risk - lambda * magnitude
    loss = emp_risk - lambda_mag * magnitude
    
    return loss, emp_risk, magnitude


def model_loss_with_generator(model, generator, inputs_normalized, targets, loss_fn, device):
    """
    Compute loss for training the model with generated noise.
    
    The model is trained ONLY on noisy inputs to be robust to the generated noise.
    
    Args:
        model: The classifier model
        generator: The noise generator network (frozen during model training)
        inputs_normalized: Already-transformed input images (normalized + augmented from dataloader)
        targets: True labels
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        device: Device to run computations on
        
    Returns:
        loss: Model loss on noisy inputs only
    """
    model.train()
    generator.eval()  # Ensure generator is in eval mode
    
    # Inputs are already normalized and augmented from the dataloader
    inputs_normalized = inputs_normalized.to(device)
    
    # Generate noise (no gradients through generator)
    with torch.no_grad():
        noise = generator(inputs_normalized)
    
    # Apply noise to normalized inputs
    inputs_noisy = torch.clamp(inputs_normalized + noise, -3, 3)
    
    # Loss on noisy inputs only
    output_noisy = model(inputs_noisy)
    loss = loss_fn(output_noisy, targets.to(device)).mean()
    
    return loss


class UniformNoise(object):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, tensor):
        out = tensor + torch.rand(tensor.size()) * self.eps * 2 -self.eps
        return out
        return torch.clamp(out, 0, 1)


class GaussianNoise(object):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, tensor):
        out = tensor + torch.randn(tensor.size()) * self.eps
        return out
        return torch.clamp(out, 0, 1)


class BernoulliNoise(object):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, tensor):
        noise = (torch.rand(tensor.size()) > 0.5).float() * 2 - 1
        out = tensor + noise * self.eps
        return out
        return torch.clamp(out, 0, 1)

