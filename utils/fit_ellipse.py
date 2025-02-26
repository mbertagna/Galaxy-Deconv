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

    # Min-max normalization per batch with improved numerical stability
    tensor_reshaped = transformed_tensor.view(transformed_tensor.shape[0], -1)
    min_val = tensor_reshaped.min(dim=1, keepdim=True)[0].unsqueeze(-1)
    max_val = tensor_reshaped.max(dim=1, keepdim=True)[0].unsqueeze(-1)
    
    # Check for valid range with improved stability
    range_diff = max_val - min_val
    valid_range = (range_diff > 1e-6).float()
    
    # Use where to avoid division by zero
    normalized = torch.where(
        valid_range > 0,
        (transformed_tensor - min_val) / (range_diff + 1e-8),
        torch.zeros_like(transformed_tensor)
    )
    
    # Replace any NaN or Inf values with zeros
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    
    normalized.requires_grad_(True)
    return normalized

def sigmoid_mask_batched(x: torch.Tensor, 
                        peak_pos: float = 0.5, 
                        sharpness: float = 0.1) -> torch.Tensor:
    """
    Batched version of sigmoid mask.
    Input shape: (B, H, W)
    Output shape: (B, H, W)
    """
    exp_term = torch.clamp(((x - peak_pos) / sharpness) ** 2, max=50)  # Clamp to prevent overflow
    return 1 / torch.exp(exp_term)

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
    
    # Reshape weights to (B, H*W) and handle potential nan/inf values
    weights = mask.reshape(B, H*W)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    grad_x = 2*A*x + B*y + D  # (B, N)
    grad_y = B*x + 2*C*y + E  # (B, N)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)  # Add epsilon inside sqrt for stability
    
    # Handle potential division by zero
    samsons_dist = torch.abs(algebraic_dist) / grad_magnitude
    
    # Clean any NaN/Inf values
    samsons_dist = torch.nan_to_num(samsons_dist, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply weights
    weighted_samsons_dist = samsons_dist * weights  # (B, N)

    return weighted_samsons_dist

def weighted_ellipse_fit_batched(points, weights, max_retries=3):
    """
    Batched version of weighted ellipse fit with robust handling of numerical issues.
    
    Parameters:
        points (Tensor): (B, N, 2) tensor of (x, y) points
        weights (Tensor): (B, N) tensor of weights
        max_retries (int): Maximum number of retry attempts with adjusted weights
    
    Returns:
        params (Tensor): (B, 6) tensor of ellipse parameters [A, B, C, D, E, F]
    """
    B, N, _ = points.shape
    device = points.device
    
    # Extract x and y coordinates
    x = points[..., 0]  # (B, N)
    y = points[..., 1]  # (B, N)
    
    # Construct the design matrix (B, N, 6)
    D = torch.stack((x**2, x*y, y**2, x, y, torch.ones_like(x)), dim=-1)
    
    # Apply weights to design matrix
    D_weighted = D * weights.unsqueeze(-1)  # (B, N, 6)
    
    # Check for nonfinite values and replace them
    D_weighted = torch.nan_to_num(D_weighted, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Initialize default parameters if SVD fails completely
    default_params = torch.zeros((B, 6), device=device)
    default_params[:, 0] = 1.0  # A = 1
    default_params[:, 2] = 1.0  # C = 1
    
    # Try SVD with progressively stronger regularization
    params = None
    success_mask = torch.zeros(B, dtype=torch.bool, device=device)
    
    for attempt in range(max_retries):
        # Skip batches that already succeeded
        batch_indices = torch.where(~success_mask)[0]
        if len(batch_indices) == 0:
            break
            
        # Select only the batch elements that haven't succeeded yet
        D_to_process = D_weighted[batch_indices]
        
        try:
            # Add small random noise to help with numerical stability (increases with each attempt)
            noise_scale = 1e-7 * (10**attempt)
            noisy_D = D_to_process + noise_scale * torch.randn_like(D_to_process)
            
            # Try to run SVD
            U, S, V = torch.svd(noisy_D)
            
            # Get the last column of V for each batch
            batch_params = V[..., -1]  # (Remaining_B, 6)
            
            # Normalize parameters
            norm = torch.norm(batch_params, dim=-1, keepdim=True)
            batch_params = batch_params / (norm + 1e-8)
            
            # Update the success mask for these batches
            success_mask[batch_indices] = True
            
            # Initialize params tensor if not already done
            if params is None:
                params = default_params.clone()
                
            # Update params for the successful batches
            params[batch_indices] = batch_params
            
        except RuntimeError as e:
            if "algorithm failed to converge" in str(e).lower():
                # Continue to the next attempt with stronger regularization
                continue
            else:
                # For other errors, break and use what we have so far
                break
    
    # If all attempts failed, return the default parameters
    if params is None:
        params = default_params
    
    # Force the parameters to be ellipses by ensuring A*C > B^2/4
    A, B, C = params[:, 0], params[:, 1], params[:, 2]
    B_sq_over_4 = B**2 / 4
    condition = (A * C <= B_sq_over_4)
    
    # Adjust C where needed to ensure it's an ellipse
    C_adjusted = torch.where(
        condition,
        B_sq_over_4 / (A + 1e-8) + 0.01,  # Make C slightly larger than needed
        C
    )
    params[:, 2] = C_adjusted
    
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
    
    # Calculate ellipse parameters with numerical safeguards
    denominator = 4*A*C - B**2
    valid_denom = (torch.abs(denominator) > 1e-8).float()
    safe_denominator = denominator + 1e-8
    
    # Compute center coordinates
    cx = torch.where(
        valid_denom > 0, 
        (B*E - 2*C*D) / safe_denominator,
        torch.zeros_like(denominator)
    )
    
    cy = torch.where(
        valid_denom > 0,
        (B*D - 2*A*E) / safe_denominator,
        torch.zeros_like(denominator)
    )
    
    # Calculate orientation
    theta = 0.5 * torch.atan2(B, A - C + 1e-10)
    
    # Calculate semi-axes lengths with stability checks
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    expr1 = A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F
    
    # Compute denominators for a and b with stability checks
    a_denom = A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2
    b_denom = A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2
    
    # Ensure positive denominators (they should be positive for valid ellipses)
    a_denom = torch.clamp(a_denom, min=1e-8)
    b_denom = torch.clamp(b_denom, min=1e-8)
    
    # Calculate squared semi-axes
    a_squared = -2 * expr1 / a_denom
    b_squared = -2 * expr1 / b_denom
    
    # Handle negative values (which shouldn't occur for valid ellipses)
    a_squared = torch.clamp(a_squared, min=1e-8)
    b_squared = torch.clamp(b_squared, min=1e-8)
    
    # Calculate semi-axes
    a = torch.sqrt(a_squared)
    b = torch.sqrt(b_squared)
    
    # Ensure a ≥ b (by convention, a is the semi-major axis)
    a_temp = torch.max(a, b)
    b = torch.min(a, b)
    a = a_temp
    
    # Reshape a for proper broadcasting: (B,) -> (B, 1)
    a_expanded = a.unsqueeze(1)  # Now shape is (B, 1)
    
    # Normalize by semi-major axis with stability check
    normalized_dist = weighted_samsons_dist / (a_expanded + 1e-8)
    
    # Calculate weighted mean with stability check
    total_weighted_dist = torch.sum(normalized_dist * weights, dim=1)
    total_weight = torch.sum(weights, dim=1) + 1e-8  # Avoid division by zero
    mean_normalized_samsons_dist = total_weighted_dist / total_weight
    
    # Clean any remaining NaN values from the final output
    params_output = torch.stack([cx, cy, theta, a, b], dim=-1)
    params_output = torch.nan_to_num(params_output, nan=0.0, posinf=0.0, neginf=0.0)
    mean_normalized_samsons_dist = torch.nan_to_num(mean_normalized_samsons_dist, nan=1.0, posinf=1.0, neginf=1.0)
    
    return params_output, mean_normalized_samsons_dist

def safe_ellipse_params_batched(image_tensor, peak_pos=0.5, sharpness=0.1):
    """
    A wrapper around ellipse_params_batched that handles exceptions at the batch element level
    and detaches gradients for problematic samples.
    """
    B = image_tensor.shape[0]
    device = image_tensor.device
    
    # Initialize output tensors
    all_params = torch.zeros((B, 5), device=device)
    all_confidence = torch.zeros(B, device=device)
    
    # Process each batch element individually
    for i in range(B):
        try:
            # Process single image
            single_image = image_tensor[i:i+1]  # Keep batch dimension
            params, confidence = ellipse_params_batched(single_image, peak_pos, sharpness)
            
            # Check for NaN or Inf values
            if (torch.isnan(params).any() or torch.isinf(params).any() or 
                torch.isnan(confidence).any() or torch.isinf(confidence).any()):
                raise ValueError("NaN or Inf values detected in output")
                
            all_params[i] = params[0]  # First (only) element of batch
            all_confidence[i] = confidence[0]
            
        except Exception as e:
            # Log the error for debugging
            print(f"Error in ellipse fitting for batch element {i}: {str(e)}")
            
            # Default values (detached from computation graph)
            default_params = torch.tensor([
                image_tensor.shape[2] / 2,  # cx = width/2
                image_tensor.shape[1] / 2,  # cy = height/2
                0.0,                        # theta = 0
                10.0,                       # a = 10
                10.0                        # b = 10
            ], device=device).detach()
            
            all_params[i] = default_params
            all_confidence[i] = torch.tensor(1.0, device=device).detach()
    
    return all_params, all_confidence

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