import torch

def transform_tensor(tensor):
    """
    Transform a loaded tensor to match the format of the function that loads PNGs.
    Ensures shape (H, W) with values in [0, 1] and gradients enabled.
    """
    if tensor.dim() == 2:  # Already grayscale
        transformed_tensor = tensor.clone()
    elif tensor.dim() == 3 and tensor.shape[0] in {1, 3}:  # Convert to single channel
        transformed_tensor = tensor.mean(dim=0)  # Convert to grayscale
    
    # Normalize if needed
    if transformed_tensor.max() > 1.0:
        transformed_tensor = transformed_tensor / 255.0

    # Min-max normalization
    min_val, max_val = transformed_tensor.min(), transformed_tensor.max()
    if max_val > min_val:  # Avoid division by zero
        transformed_tensor = (transformed_tensor - min_val) / (max_val - min_val)
    
    transformed_tensor.requires_grad_(True)
    return transformed_tensor

def sigmoid_mask(x: torch.Tensor, 
                   peak_pos: float = 0.4, 
                   sharpness: float = 20.0) -> torch.Tensor:
    """
    Differentiable nonlinear transformation for galaxy edge processing.
    Maintains low values, amplifies mid-range, suppresses highs.
    """
    scaled_x = sharpness * (x - peak_pos)
    return x * torch.sigmoid(scaled_x) * (1 - torch.sigmoid(scaled_x - 2.0))

def mask_to_points_and_weights_full(mask):
    """
    Converts a masked image tensor into points with associated weights.
    """
    H, W = mask.shape
    x_coords, y_coords = torch.meshgrid(torch.arange(H), torch.arange(W))
    points = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).float()
    weights = mask.flatten()
    return points, weights

def weighted_ellipse_fit(points, weights):
    """
    Fit an ellipse to weighted points using SVD, avoiding in-place operations.
    
    Parameters:
        points (Tensor): Nx2 tensor of (x, y) points
        weights (Tensor): N-element tensor of weights
    
    Returns:
        params (Tensor): The ellipse parameters [A, B, C, D, E, F]
    """
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Construct the design matrix
    D = torch.stack((x**2, x*y, y**2, x, y, torch.ones_like(x)), dim=1)
    
    # Create diagonal weight matrix and apply weights
    W = torch.diag_embed(weights)
    D_weighted = torch.matmul(W, D)
    
    # Solve using SVD
    U, S, V = torch.svd(D_weighted)
    params = V[:, -1]
    
    # Normalize parameters WITHOUT in-place operation
    norm = torch.norm(params)
    params = params / (norm + 1e-8)  # Add small epsilon for numerical stability
    
    return params

def ellipse_params(image_tensor):
    masked_image = sigmoid_mask(image_tensor)
    points, weights = mask_to_points_and_weights_full(masked_image)
    params = weighted_ellipse_fit(points, weights)

    # Assuming params are already tensors, if not, convert them in a differentiable manner
    A, B, C, D, E, F = [param.to(image_tensor.device) for param in params]

    # Calculate ellipse center
    denominator = 4*A*C - B**2
    cx = (B*E - 2*C*D) / denominator
    cy = (B*D - 2*A*E) / denominator
    
    # Calculate rotation angle and semi-axes
    theta = 0.5 * torch.atan2(B, A - C)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Calculate semi-axes lengths
    a_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / \
                (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2)
    b_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / \
                (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2)
    
    a = torch.sqrt(torch.abs(a_squared))
    b = torch.sqrt(torch.abs(b_squared))

    return torch.stack([cx, cy, theta, a, b])

def ellipse_loss(output_params, target_params, center_weight=1.0, angle_weight=1.0, axis_weight=1.0):
    """
    Computes normalized loss between output and target ellipse parameters where each
    component (center, angle, axes) contributes equally to the total loss.
    
    Args:
    - output_params: (cx_out, cy_out, theta_out, a_out, b_out)
    - target_params: (cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt)
    - center_weight: Weight for center loss
    - angle_weight: Weight for angle loss
    - axis_weight: Weight for axis loss
    
    Returns:
    - total_loss: Combined normalized loss
    """
    # Unpack parameters
    cx_out, cy_out, theta_out, a_out, b_out = output_params
    cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params
    
    # Center loss (normalized by image size)
    center_coords_out = torch.stack([cx_out, cy_out])
    center_coords_tgt = torch.stack([cx_tgt, cy_tgt])
    
    # Normalize center coordinates by the maximum of target axes
    coord_scale = torch.max(torch.stack([a_tgt, b_tgt]))
    normalized_center_loss = F.mse_loss(
        center_coords_out / (coord_scale + 1e-8),
        center_coords_tgt / (coord_scale + 1e-8)
    )
    
    # Angle loss (normalized to be between 0 and 1)
    # Convert angles to normalized direction vectors to handle periodicity
    def angle_to_vector(theta):
        return torch.stack([torch.cos(theta), torch.sin(theta)])
    
    angle_vec_out = angle_to_vector(theta_out)
    angle_vec_tgt = angle_to_vector(theta_tgt)
    normalized_angle_loss = 0.5 * torch.nn.functional.mse_loss(angle_vec_out, angle_vec_tgt)
    
    # Axis loss (normalized by the larger target axis)
    axis_scale = torch.max(torch.stack([a_tgt, b_tgt]))
    normalized_axis_loss = 0.5 * (
        torch.nn.functional.l1_loss(a_out / (axis_scale + 1e-8), a_tgt / (axis_scale + 1e-8)) +
        torch.nn.functional.l1_loss(b_out / (axis_scale + 1e-8), b_tgt / (axis_scale + 1e-8))
    )
    
    # Combine losses with weights
    total_loss = (
        center_weight * normalized_center_loss +
        angle_weight * normalized_angle_loss +
        axis_weight * normalized_axis_loss
    )

    return total_loss

def eloss(output_i, target_i):
    return ellipse_loss(ellipse_params(transform_tensor(output_i)), ellipse_params(transform_tensor(target_i)))