import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def transform_tensor_batched(tensor):
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
    return 1 / torch.exp(((x - peak_pos) / sharpness) ** 2)

def mask_to_points_and_weights_batched(mask):
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
    masked_image = sigmoid_mask_batched(image_tensor, peak_pos=peak_pos, sharpness=sharpness)
    points, weights = mask_to_points_and_weights_batched(masked_image)
    params = weighted_ellipse_fit_batched(points, weights)
    weighted_samsons_dist = weighted_samsons_distance_batched(points, params, weights)  # (B, H*W)
    
    # Extract parameters
    A, B, C, D, E, F = params.unbind(-1)
    
    # Calculate ellipse parameters
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

    # Reshape a for proper broadcasting: (B,) -> (B, 1)
    a_expanded = a.unsqueeze(1)  # Now shape is (B, 1)
    
    # Normalize by semi-major axis
    normalized_dist = weighted_samsons_dist / (a_expanded + 1e-8)
    
    # Calculate weighted mean
    total_weighted_dist = torch.sum(normalized_dist * weights, dim=1)
    total_weight = torch.sum(weights, dim=1)
    mean_normalized_samsons_dist = total_weighted_dist / (total_weight + 1e-8)
    
    return torch.stack([cx, cy, theta, a, b], dim=-1), mean_normalized_samsons_dist

def safe_ellipse_params_batched(image_tensor, peak_pos=0.5, sharpness=0.1):
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

def plot_batch_with_ellipses(images, ellipses_params, num_cols=2, figsize=None):
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

def ellipse_fit_metric(image_tensor, ellipse_params):
    """
    Computes a normalized metric (0 to 1) indicating how well an ellipse fits a galaxy.
    Higher values indicate better fit (more intensity inside ellipse, less outside).
    
    Parameters:
    -----------
    image_tensor : torch.Tensor
        The input image tensor with shape (B, H, W) or (B, C, H, W)
    ellipse_params : torch.Tensor
        Tensor of ellipse parameters with shape (B, 5) containing:
        [center_y, center_x, theta, a, b] for each image in the batch
    
    Returns:
    --------
    torch.Tensor
        A tensor of shape (B,) with values between 0 and 1 representing 
        the normalized fit metric for each image
    """
    # Ensure image is grayscale (B, H, W)
    if image_tensor.dim() == 4:  # (B, C, H, W)
        # Use RGB weights for grayscale conversion if needed
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device)
        image = torch.einsum('bchw,c->bhw', image_tensor, rgb_weights)
    else:
        image = image_tensor
    
    # Extract batch size and image dimensions
    B, H, W = image.shape
    device = image.device
    dtype = image.dtype
    
    # Extract ellipse parameters
    cy = ellipse_params[:, 0]  # center_y (first parameter in your coordinate system)
    cx = ellipse_params[:, 1]  # center_x
    theta = ellipse_params[:, 2]  # rotation angle
    a = ellipse_params[:, 3]  # semi-major axis
    b = ellipse_params[:, 4]  # semi-minor axis
    
    # Create coordinate grids for all images in batch
    y_indices = torch.arange(H, dtype=dtype, device=device)
    x_indices = torch.arange(W, dtype=dtype, device=device)
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Expand grids to match batch dimension (B, H, W)
    y_grid = y_grid.unsqueeze(0).expand(B, H, W)
    x_grid = x_grid.unsqueeze(0).expand(B, H, W)
    
    # Prepare results tensor
    results = torch.zeros(B, dtype=dtype, device=device)
    
    # Total image area
    total_area = H * W
    
    for i in range(B):
        # Translate coordinates to center of ellipse
        x_trans = x_grid[i] - cx[i]
        y_trans = y_grid[i] - cy[i]
        
        # Rotate coordinates
        cos_theta = torch.cos(theta[i])
        sin_theta = torch.sin(theta[i])
        x_rot = x_trans * cos_theta + y_trans * sin_theta
        y_rot = -x_trans * sin_theta + y_trans * cos_theta
        
        # Create elliptical mask using smooth approximation for differentiability
        ellipse_eq = (x_rot / a[i])**2 + (y_rot / b[i])**2
        sharpness = torch.tensor(20.0, dtype=dtype, device=device)  # Adjust for desired edge sharpness
        ellipse_mask = torch.sigmoid(-sharpness * (ellipse_eq - 1.0))
        
        # Sum of intensity inside ellipse (using soft mask)
        intensity_inside = torch.sum(image[i] * ellipse_mask)
        
        # Area of ellipse (sum of mask values for soft boundary)
        area_inside = torch.sum(ellipse_mask)
        
        # Sum of intensity outside ellipse
        outside_mask = 1.0 - ellipse_mask
        intensity_outside = torch.sum(image[i] * outside_mask)
        
        # Area outside ellipse
        area_outside = torch.sum(outside_mask)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Calculate densities
        inside_density = intensity_inside / (area_inside + eps)
        outside_density = intensity_outside / (area_outside + eps)
        
        # Calculate contrast ratio
        contrast_ratio = inside_density / (outside_density + eps)
        
        # Normalize to range [0, 1]
        normalized_score = contrast_ratio / (1.0 + contrast_ratio)
        
        results[i] = normalized_score
    
    return results