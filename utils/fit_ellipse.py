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
                        peak_pos: float = 0.5, 
                        sharpness: float = 0.1) -> torch.Tensor:
    """
    Batched version of sigmoid mask.
    Input shape: (B, H, W)
    Output shape: (B, H, W)
    """
    return 1 / torch.exp(((x - peak_pos) / sharpness) ** 2)

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

def weighted_samsons_distance_batched(points, coeffs, weights):
    """
    Compute weighted Samson's distance for batches of points given batches of ellipse coefficients.
    
    Args:
        points: Tensor of shape (B, N, 2), where B is batch size, N is number of points, 
                and each point is (y, x) following your convention.
        coeffs: Tensor of shape (B, 6), where coeffs = (A, B, C, D, E, F) for each batch.
        weights: Tensor of shape (B, N), containing the weight for each point in each batch.
    
    Returns:
        Tensor of shape (B, N) with weighted Samson's distances for each point in each batch.
    """
    y, x = points[..., 0], points[..., 1]  # Using your coordinate convention
    
    # Extract coefficients for each batch (B, 6) -> (B, 1) for broadcasting
    A = coeffs[:, 0:1]
    B = coeffs[:, 1:2]
    C = coeffs[:, 2:3]
    D = coeffs[:, 3:4]
    E = coeffs[:, 4:5]
    F = coeffs[:, 5:6]

    # Compute the algebraic distance for each point in each batch
    algebraic_dist = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F  # (B, N)

    # Compute the gradient magnitude normalization term
    # Using the correct gradient computation
    grad_x = 2*A*x + B*y + D  # (B, N)
    grad_y = B*x + 2*C*y + E  # (B, N)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # (B, N)

    # Compute Samson's distance
    samsons_dist = torch.abs(algebraic_dist) / (grad_magnitude + 1e-8)  # (B, N)
    
    # Apply weights
    weighted_samsons_dist = samsons_dist * weights  # (B, N)

    return weighted_samsons_dist

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

def ellipse_params_batched(image_tensor, peak_pos: float = 0.5, sharpness: float = 0.1):
    """
    Compute ellipse parameters and confidence for a batch of images.
    Input shape: (B, H, W)
    Output shapes: 
        params: (B, 5) containing [cx, cy, theta, a, b]
        samsons_dist: (B,) containing confidence scores
    """
    masked_image = sigmoid_mask_batched(image_tensor, peak_pos=peak_pos, sharpness=sharpness)
    points, weights = mask_to_points_and_weights_batched(masked_image)
    params = weighted_ellipse_fit_batched(points, weights)
    weighted_samsons_dist = weighted_samsons_distance_batched(points, params, weights)  # (B, H*W)
    
    # Extract parameters
    A, B, C, D, E, F = params.unbind(-1)
    
    # Calculate ellipse parameters as before...
    denominator = 4*A*C - B**2
    cx = (B*E - 2*C*D) / (denominator + 1e-8)
    cy = (B*D - 2*A*E) / (denominator + 1e-8)
    theta = 0.5 * torch.atan2(B, A - C)
    
    # Calculate semi-axes lengths
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    expr1 = A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F
    a_squared = -2 * expr1 / (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2 + 1e-8)
    b_squared = -2 * expr1 / (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2 + 1e-8)
    
    a = torch.sqrt(torch.abs(a_squared))
    b = torch.sqrt(torch.abs(b_squared))
    
    total_weighted_dist = torch.sum(weighted_samsons_dist, dim=1)  # (B,)
    total_weight = torch.sum(weights, dim=1)  # (B,)
    mean_samsons_dist = total_weighted_dist / (total_weight + 1e-8)  # (B,)
    
    return torch.stack([cx, cy, theta, a, b], dim=-1), mean_samsons_dist

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

# def eloss(output_batch, target_batch):
#     transformed_output = transform_tensor_batched(output_batch)
#     transformed_target = transform_tensor_batched(target_batch)
    
#     output_params, _ = ellipse_params_batched(transformed_output)
#     target_params, _ = ellipse_params_batched(transformed_target)
    
#     return ellipse_loss_batched(output_params, target_params)

def plot_batch_with_ellipses(images, ellipses_params, num_cols=2, figsize=None):
    """
    Plot a batch of images with their fitted ellipses overlaid.
    Always shows a 40x40 unit view regardless of ellipse size.
    """
    # Convert images to numpy and ensure they're grayscale
    if images.dim() == 4:  # (B, C, H, W)
        images = images.mean(dim=1)
    images_np = images.detach().cpu().numpy()
    
    batch_size = images.shape[0]
    num_rows = (batch_size + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    t = np.linspace(0, 2*np.pi, 100)
    
    for img_idx in range(batch_size):
        row, col = img_idx // num_cols, img_idx % num_cols
        ax = axes[row, col]
        
        # Plot the image with fixed extent for 40x40 units
        ax.imshow(images_np[img_idx], cmap='gray', extent=[0, 40, 40, 0])

        # Set fixed limits for 40x40 view
        ax.set_xlim(0, 40)
        ax.set_ylim(40, 0)  # Inverted y-axis to match image coordinates
        
        color_codes = ['r', 'g', 'b', 'c', 'm', 'y']
        
        # Loop over different ellipse parameters (from different peak positions)
        for param_idx, ellipse_params in enumerate(ellipses_params):
            params_np = ellipse_params.detach().cpu().numpy()
            cx, cy, theta, a, b = params_np[img_idx]
            
            # Scale the coordinates to match the 40x40 view
            height, width = images_np[img_idx].shape
            scale_x = 40 / width
            scale_y = 40 / height
            
            cx = cx * scale_y  # Scale center x (note: x uses scale_y because of image coordinate system)
            cy = cy * scale_x  # Scale center y (note: y uses scale_x because of image coordinate system)
            a = a * scale_y    # Scale semi-major axis
            b = b * scale_x    # Scale semi-minor axis
            
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
            ax.plot(points[:, 1], points[:, 0], color_codes[param_idx%len(color_codes)]+'-', linewidth=2)
            ax.plot(cy, cx, color_codes[param_idx%len(color_codes)]+'+', markersize=10)
        
        ax.set_title(f'Image {img_idx}')
        ax.grid(True)  # Add grid for better visibility of units
        
    # Hide empty subplots
    for idx in range(batch_size, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, axes

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