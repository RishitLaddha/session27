"""
This module implements custom layer functions using basic PyTorch tensor operations.
No high-level torch.nn layers are used.
Each function is written with detailed comments to explain the steps.
"""

import torch
import math

def pad2d(input_tensor, pad):
    """
    Manually applies 2D zero-padding to an input tensor.
    
    Parameters:
      input_tensor (Tensor): Input tensor of shape (N, C, H, W)
      pad (int): Number of zeros to add to each side
      
    Returns:
      Tensor: Padded tensor of shape (N, C, H+2*pad, W+2*pad)
    """
    if pad == 0:
        return input_tensor
    N, C, H, W = input_tensor.shape
    H_new, W_new = H + 2 * pad, W + 2 * pad
    # Create a new tensor filled with zeros on the same device and dtype
    padded = torch.zeros((N, C, H_new, W_new), device=input_tensor.device, dtype=input_tensor.dtype)
    # Copy the original tensor into the center of the padded tensor
    padded[:, :, pad:pad+H, pad:pad+W] = input_tensor
    return padded

def conv2d_custom(x, weight, bias, stride=1, padding=0):
    """
    Custom implementation of a 2D convolution layer.
    Uses explicit loops over the output spatial dimensions.
    
    Parameters:
      x (Tensor): Input tensor of shape (N, C_in, H, W)
      weight (Tensor): Convolution kernel of shape (C_out, C_in, kH, kW)
      bias (Tensor): Bias tensor of shape (C_out,)
      stride (int): Stride of the convolution
      padding (int): Zero-padding to add to each side of the input
      
    Returns:
      Tensor: Convolved output of shape (N, C_out, H_out, W_out)
    """
    # Apply padding manually
    x_padded = pad2d(x, padding)
    N, C_in, H, W = x_padded.shape
    C_out, _, kH, kW = weight.shape
    # Calculate the spatial dimensions of the output
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    # Create output tensor
    out = torch.zeros((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    # Loop over batch, output channels, and spatial positions
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    # Extract patch from input and perform element-wise multiplication and sum
                    patch = x_padded[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    out[n, c, i, j] = torch.sum(patch * weight[c]) + bias[c]
    return out

def relu(x):
    """
    Custom ReLU activation function.
    
    Parameters:
      x (Tensor): Input tensor.
      
    Returns:
      Tensor: Output tensor with negative elements set to 0.
    """
    return torch.clamp(x, min=0)

def max_pool2d_custom(x, kernel_size=2, stride=2):
    """
    Custom implementation of a 2D max pooling layer.
    Uses explicit loops to select the maximum value in each window.
    
    Parameters:
      x (Tensor): Input tensor of shape (N, C, H, W)
      kernel_size (int): Size of the pooling window
      stride (int): Stride for the pooling operation
      
    Returns:
      Tensor: Pooled output of shape (N, C, H_out, W_out)
    """
    N, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    out = torch.zeros((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x[n, c, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    out[n, c, i, j] = torch.max(patch)
    return out

def flatten(x):
    """
    Flattens a tensor except for the first (batch) dimension.
    
    Parameters:
      x (Tensor): Input tensor.
      
    Returns:
      Tensor: Flattened tensor of shape (N, -1)
    """
    return x.view(x.shape[0], -1)

def linear_custom(x, weight, bias):
    """
    Custom implementation of a linear (fully connected) layer.
    
    Parameters:
      x (Tensor): Input tensor of shape (N, in_features)
      weight (Tensor): Weight matrix of shape (in_features, out_features)
      bias (Tensor or int): Bias tensor of shape (out_features,). If bias is 0, no bias is added.
      
    Returns:
      Tensor: Output tensor of shape (N, out_features)
    """
    if isinstance(bias, int) and bias == 0:
        return x @ weight
    return x @ weight + bias

def softmax(x, dim):
    """
    Custom softmax function that applies the exponential and normalizes.
    
    Parameters:
      x (Tensor): Input tensor.
      dim (int): Dimension along which to apply softmax.
      
    Returns:
      Tensor: Softmax-normalized tensor.
    """
    # Subtract max for numerical stability
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def gelu(x):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    
    Parameters:
      x (Tensor): Input tensor.
    
    Returns:
      Tensor: Output tensor after applying GELU.
    """
    return x * 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, eps=1e-5):
    """
    Custom layer normalization that normalizes over the last dimension.
    
    Parameters:
      x (Tensor): Input tensor of arbitrary shape.
      eps (float): A small number for numerical stability.
      
    Returns:
      Tensor: The layer-normalized tensor.
    """
    # Compute the mean and variance along the last dimension.
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    # Normalize the input tensor.
    return (x - mean) / torch.sqrt(var + eps)
