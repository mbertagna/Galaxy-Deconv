import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def transform_tensor_batched(tensor):
    """
    Transform a batch of tensors to match the format of the function that loads PNGs.
    Input shape can be (B, C, H, W) or (B, H, W)
    Output shape will be (B, H, W) with values in [0, 1] and gradients enabled.
    """
    if tensor.dim() == 3:  # (B, H, W)
        transformed_tensor = tensor.clone()
    elif tensor.dim() == 4:  # (B, C, H, W)
        # Use RGB weights for grayscale conversion
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device)
        transformed_tensor = torch.einsum('bchw,c->bhw', tensor, rgb_weights)
    
    # Normalize if needed
    if transformed_tensor.max() > 1.0:
        transformed_tensor = transformed_tensor / 255.0

    # Min-max normalization per batch
    min_val = transformed_tensor.view(transformed_tensor.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    max_val = transformed_tensor.view(transformed_tensor.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    valid_range = (max_val > min_val).float()
    transformed_tensor = valid_range * (transformed_tensor - min_val) / (max_val - min_val + 1e-8) + (1 - valid_range) * transformed_tensor
    
    transformed_tensor.requires_grad_(True)
    return transformed_tensor

def sigmoid_mask_batched(x: torch.Tensor, 
                        peak_pos: float = 0.4, 
                        sharpness: float = 20.0) -> torch.Tensor:
    """
    Batched version of sigmoid mask.
    Input shape: (B, H, W)
    Output shape: (B, H, W)
    """
    scaled_x = sharpness * (x - peak_pos)
    return x * torch.sigmoid(scaled_x) * (1 - torch.sigmoid(scaled_x - 2.0))

def mask_to_points_and_weights_batched(mask):
    """
    Converts a batch of masked images into points with associated weights.
    Input shape: (B, H, W)
    Output shapes: points (B, H*W, 2), weights (B, H*W)
    """
    B, H, W = mask.shape
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=mask.device), 
                                      torch.arange(W, device=mask.device))
    
    # Create points grid (H, W, 2)
    points_grid = torch.stack((y_coords, x_coords), dim=-1).float()
    
    # Expand points for batch dimension (B, H, W, 2)
    points = points_grid.unsqueeze(0).expand(B, H, W, 2)
    
    # Reshape to (B, H*W, 2)
    points = points.reshape(B, H*W, 2)
    
    # Reshape weights to (B, H*W)
    weights = mask.reshape(B, H*W)
    
    return points, weights

def weighted_ellipse_fit_batched(points, weights):
    """
    Batched version of weighted ellipse fit.
    
    Parameters:
        points (Tensor): (B, N, 2) tensor of (x, y) points
        weights (Tensor): (B, N) tensor of weights
    
    Returns:
        params (Tensor): (B, 6) tensor of ellipse parameters [A, B, C, D, E, F]
    """
    B, N, _ = points.shape
    
    # Extract x and y coordinates
    x = points[..., 0]  # (B, N)
    y = points[..., 1]  # (B, N)
    
    # Construct the design matrix (B, N, 6)
    D = torch.stack((x**2, x*y, y**2, x, y, torch.ones_like(x)), dim=-1)
    
    # Apply weights to design matrix
    D_weighted = D * weights.unsqueeze(-1)  # (B, N, 6)
    
    # Solve using SVD for each batch
    # We can use torch.svd on the whole batch at once
    U, S, V = torch.svd(D_weighted)
    
    # Get the last column of V for each batch
    params = V[..., -1]  # (B, 6)
    
    # Normalize parameters
    norm = torch.norm(params, dim=-1, keepdim=True)
    params = params / (norm + 1e-8)
    
    return params

def ellipse_params_batched(image_tensor):
    """
    Compute ellipse parameters for a batch of images.
    Input shape: (B, H, W)
    Output shape: (B, 5) containing [cx, cy, theta, a, b] for each image
    """
    masked_image = sigmoid_mask_batched(image_tensor)
    points, weights = mask_to_points_and_weights_batched(masked_image)
    params = weighted_ellipse_fit_batched(points, weights)
    
    # Extract parameters
    A, B, C, D, E, F = params.unbind(-1)
    
    # Calculate ellipse center
    denominator = 4*A*C - B**2
    cx = (B*E - 2*C*D) / (denominator + 1e-8)
    cy = (B*D - 2*A*E) / (denominator + 1e-8)
    
    # Calculate rotation angle
    theta = 0.5 * torch.atan2(B, A - C)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Calculate semi-axes lengths
    expr1 = A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F
    a_squared = -2 * expr1 / (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2 + 1e-8)
    b_squared = -2 * expr1 / (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2 + 1e-8)
    
    a = torch.sqrt(torch.abs(a_squared))
    b = torch.sqrt(torch.abs(b_squared))
    
    return torch.stack([cx, cy, theta, a, b], dim=-1)

def ellipse_loss_batched(output_params, target_params, center_weight=1.0, angle_weight=1.0, axis_weight=1.0):
    """
    Batched version of ellipse loss.
    Input shapes: (B, 5) for both output_params and target_params
    Returns: scalar loss
    """
    # Unpack parameters
    cx_out, cy_out, theta_out, a_out, b_out = output_params.unbind(-1)
    cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params.unbind(-1)
    
    # Center loss
    center_coords_out = torch.stack([cx_out, cy_out], dim=-1)
    center_coords_tgt = torch.stack([cx_tgt, cy_tgt], dim=-1)
    
    # Normalize by maximum axis length per batch
    coord_scale = torch.max(torch.stack([a_tgt, b_tgt], dim=-1), dim=-1)[0].unsqueeze(-1)
    normalized_center_loss = F.mse_loss(
        center_coords_out / (coord_scale.unsqueeze(-1) + 1e-8),
        center_coords_tgt / (coord_scale.unsqueeze(-1) + 1e-8)
    )
    
    # Angle loss using vector representation
    angle_vec_out = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=-1)
    angle_vec_tgt = torch.stack([torch.cos(theta_tgt), torch.sin(theta_tgt)], dim=-1)
    normalized_angle_loss = 0.5 * F.mse_loss(angle_vec_out, angle_vec_tgt)
    
    # Axis loss
    axis_scale = torch.max(torch.stack([a_tgt, b_tgt], dim=-1), dim=-1)[0].unsqueeze(-1)
    normalized_axis_loss = 0.5 * (
        F.l1_loss(a_out / (axis_scale + 1e-8), a_tgt / (axis_scale + 1e-8)) +
        F.l1_loss(b_out / (axis_scale + 1e-8), b_tgt / (axis_scale + 1e-8))
    )
    
    # Combine losses
    total_loss = (
        center_weight * normalized_center_loss +
        angle_weight * normalized_angle_loss +
        axis_weight * normalized_axis_loss
    )
    
    return total_loss

def eloss(output_batch, target_batch):
    transformed_output = transform_tensor_batched(output_batch)
    transformed_target = transform_tensor_batched(target_batch)
    
    output_params = ellipse_params_batched(transformed_output)
    target_params = ellipse_params_batched(transformed_target)
    
    return ellipse_loss_batched(output_params, target_params)

def plot_batch_with_ellipses(images, ellipse_params, num_cols=4, figsize=(15, 15)):
    """
    Plot a batch of images with their fitted ellipses overlaid.
    """
    # Convert images to numpy and ensure they're grayscale
    if images.dim() == 4:  # (B, C, H, W)
        images = images.mean(dim=1)
    images_np = images.detach().cpu().numpy()
    
    # Convert ellipse parameters to numpy
    params_np = ellipse_params.detach().cpu().numpy()
    
    batch_size = images.shape[0]
    num_rows = (batch_size + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    t = np.linspace(0, 2*np.pi, 100)
    
    for idx in range(batch_size):
        row, col = idx // num_cols, idx % num_cols
        ax = axes[row, col]
        
        # Plot the image with extent to ensure correct aspect ratio
        height, width = images_np[idx].shape
        ax.imshow(images_np[idx], cmap='gray', extent=[0, width, height, 0])
        
        # Extract ellipse parameters
        cx, cy, theta, a, b = params_np[idx]
        
        # Generate ellipse points
        x = a * np.cos(t)
        y = b * np.sin(t)
        
        # Rotate and translate the ellipse
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        points = np.dot(np.stack([x, y], axis=1), R.T)
        points[:, 0] += cx
        points[:, 1] += cy
        
        # Plot the ellipse
        ax.plot(points[:, 1], points[:, 0], 'r-', linewidth=2)
        ax.plot(cy, cx, 'r+', markersize=10)
        
        ax.set_title(f'Image {idx}')
        ax.set_aspect('equal')  # This ensures square pixels
        
    # Hide empty subplots
    for idx in range(batch_size, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, axes

# Example usage:
"""
# Assuming you have:
images = torch.randn(8, 64, 64)  # Your batch of images
params = ellipse_params_batched(transform_tensor_batched(images))

# Plot the results
fig, axes = plot_batch_with_ellipses(images, params)
plt.show()
"""

# import torch

# def transform_tensor(tensor):
#     """
#     Transform a loaded tensor to match the format of the function that loads PNGs.
#     Ensures shape (H, W) with values in [0, 1] and gradients enabled.
#     """
#     print(tensor.shape)
#     transformed_tensor = tensor
#     if tensor.dim() == 2:  # Already grayscale
#         transformed_tensor = tensor.clone()
#     elif tensor.dim() == 3 and tensor.shape[0] in {1, 3}:  # Convert to single channel
#         transformed_tensor = tensor.mean(dim=0)  # Convert to grayscale
    
#     # Normalize if needed
#     if transformed_tensor.max() > 1.0:
#         transformed_tensor = transformed_tensor / 255.0

#     # Min-max normalization
#     min_val, max_val = transformed_tensor.min(), transformed_tensor.max()
#     if max_val > min_val:  # Avoid division by zero
#         transformed_tensor = (transformed_tensor - min_val) / (max_val - min_val)
    
#     transformed_tensor.requires_grad_(True)
#     return transformed_tensor

# def sigmoid_mask(x: torch.Tensor, 
#                    peak_pos: float = 0.4, 
#                    sharpness: float = 20.0) -> torch.Tensor:
#     """
#     Differentiable nonlinear transformation for galaxy edge processing.
#     Maintains low values, amplifies mid-range, suppresses highs.
#     """
#     scaled_x = sharpness * (x - peak_pos)
#     return x * torch.sigmoid(scaled_x) * (1 - torch.sigmoid(scaled_x - 2.0))

# def mask_to_points_and_weights_full(mask):
#     """
#     Converts a masked image tensor into points with associated weights.
#     """
#     H, W = mask.shape
#     x_coords, y_coords = torch.meshgrid(torch.arange(H), torch.arange(W))
#     points = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).float()
#     weights = mask.flatten()
#     return points, weights

# def weighted_ellipse_fit(points, weights):
#     """
#     Fit an ellipse to weighted points using SVD, avoiding in-place operations.
    
#     Parameters:
#         points (Tensor): Nx2 tensor of (x, y) points
#         weights (Tensor): N-element tensor of weights
    
#     Returns:
#         params (Tensor): The ellipse parameters [A, B, C, D, E, F]
#     """
#     # Extract x and y coordinates
#     x = points[:, 0]
#     y = points[:, 1]
    
#     # Construct the design matrix
#     D = torch.stack((x**2, x*y, y**2, x, y, torch.ones_like(x)), dim=1)
    
#     # Create diagonal weight matrix and apply weights
#     W = torch.diag_embed(weights)
#     D_weighted = torch.matmul(W, D)
    
#     # Solve using SVD
#     U, S, V = torch.svd(D_weighted)
#     params = V[:, -1]
    
#     # Normalize parameters WITHOUT in-place operation
#     norm = torch.norm(params)
#     params = params / (norm + 1e-8)  # Add small epsilon for numerical stability
    
#     return params

# def ellipse_params(image_tensor):
#     masked_image = sigmoid_mask(image_tensor)
#     points, weights = mask_to_points_and_weights_full(masked_image)
#     params = weighted_ellipse_fit(points, weights)

#     # Assuming params are already tensors, if not, convert them in a differentiable manner
#     A, B, C, D, E, F = [param.to(image_tensor.device) for param in params]

#     # Calculate ellipse center
#     denominator = 4*A*C - B**2
#     cx = (B*E - 2*C*D) / denominator
#     cy = (B*D - 2*A*E) / denominator
    
#     # Calculate rotation angle and semi-axes
#     theta = 0.5 * torch.atan2(B, A - C)
#     cos_t = torch.cos(theta)
#     sin_t = torch.sin(theta)
    
#     # Calculate semi-axes lengths
#     a_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / \
#                 (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2)
#     b_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / \
#                 (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2)
    
#     a = torch.sqrt(torch.abs(a_squared))
#     b = torch.sqrt(torch.abs(b_squared))

#     return torch.stack([cx, cy, theta, a, b])

# def ellipse_loss(output_params, target_params, center_weight=1.0, angle_weight=1.0, axis_weight=1.0):
#     """
#     Computes normalized loss between output and target ellipse parameters where each
#     component (center, angle, axes) contributes equally to the total loss.
    
#     Args:
#     - output_params: (cx_out, cy_out, theta_out, a_out, b_out)
#     - target_params: (cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt)
#     - center_weight: Weight for center loss
#     - angle_weight: Weight for angle loss
#     - axis_weight: Weight for axis loss
    
#     Returns:
#     - total_loss: Combined normalized loss
#     """
#     # Unpack parameters
#     cx_out, cy_out, theta_out, a_out, b_out = output_params
#     cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params
    
#     # Center loss (normalized by image size)
#     center_coords_out = torch.stack([cx_out, cy_out])
#     center_coords_tgt = torch.stack([cx_tgt, cy_tgt])
    
#     # Normalize center coordinates by the maximum of target axes
#     coord_scale = torch.max(torch.stack([a_tgt, b_tgt]))
#     normalized_center_loss = F.mse_loss(
#         center_coords_out / (coord_scale + 1e-8),
#         center_coords_tgt / (coord_scale + 1e-8)
#     )
    
#     # Angle loss (normalized to be between 0 and 1)
#     # Convert angles to normalized direction vectors to handle periodicity
#     def angle_to_vector(theta):
#         return torch.stack([torch.cos(theta), torch.sin(theta)])
    
#     angle_vec_out = angle_to_vector(theta_out)
#     angle_vec_tgt = angle_to_vector(theta_tgt)
#     normalized_angle_loss = 0.5 * torch.nn.functional.mse_loss(angle_vec_out, angle_vec_tgt)
    
#     # Axis loss (normalized by the larger target axis)
#     axis_scale = torch.max(torch.stack([a_tgt, b_tgt]))
#     normalized_axis_loss = 0.5 * (
#         torch.nn.functional.l1_loss(a_out / (axis_scale + 1e-8), a_tgt / (axis_scale + 1e-8)) +
#         torch.nn.functional.l1_loss(b_out / (axis_scale + 1e-8), b_tgt / (axis_scale + 1e-8))
#     )
    
#     # Combine losses with weights
#     total_loss = (
#         center_weight * normalized_center_loss +
#         angle_weight * normalized_angle_loss +
#         axis_weight * normalized_axis_loss
#     )

#     return total_loss

# def eloss(output_i, target_i):
#     return ellipse_loss(ellipse_params(transform_tensor(output_i)), ellipse_params(transform_tensor(target_i)))