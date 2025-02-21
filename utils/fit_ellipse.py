import torch
import torch.nn.functional as F

def transform_tensor(tensor):
    """
    Transform a batch of tensors to match the format of the function that loads PNGs.
    Ensures shape (B, H, W) with values in [0, 1] and gradients enabled.
    """
    B = tensor.shape[0]
    if tensor.dim() == 3:  # Single-channel images (B, H, W) already
        transformed_tensor = tensor.clone()
    elif tensor.dim() == 4 and tensor.shape[1] in {1, 3}:  # Convert RGB/Grayscale to single-channel
        transformed_tensor = tensor.mean(dim=1)  # Convert to grayscale (B, H, W)

    # Normalize if needed
    if transformed_tensor.max() > 1.0:
        transformed_tensor = transformed_tensor / 255.0

    # Min-max normalization per image in batch
    min_val = transformed_tensor.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
    max_val = transformed_tensor.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
    transformed_tensor = (transformed_tensor - min_val) / (max_val - min_val + 1e-8)

    transformed_tensor.requires_grad_(True)
    return transformed_tensor

def sigmoid_mask(x, peak_pos=0.4, sharpness=20.0):
    """
    Batch-wise differentiable nonlinear transformation for galaxy edge processing.
    """
    scaled_x = sharpness * (x - peak_pos)
    return x * torch.sigmoid(scaled_x) * (1 - torch.sigmoid(scaled_x - 2.0))

def mask_to_points_and_weights_full(mask):
    """
    Converts a batch of masked image tensors into points with associated weights.
    """
    B, H, W = mask.shape
    x_coords, y_coords = torch.meshgrid(torch.arange(H, device=mask.device), 
                                        torch.arange(W, device=mask.device), indexing='ij')
    
    points = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).float()  # (H*W, 2)
    points = points.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)

    weights = mask.view(B, -1)  # (B, H*W)
    
    return points, weights

def weighted_ellipse_fit(points, weights):
    """
    Fit an ellipse to weighted points using SVD, adapted for batch processing.
    """
    B, N, _ = points.shape  # (B, N, 2)
    
    x = points[:, :, 0]
    y = points[:, :, 1]
    
    # Construct design matrix (B, N, 6)
    D = torch.stack((x**2, x*y, y**2, x, y, torch.ones_like(x)), dim=2)
    
    # Create diagonal weight matrices (B, N, N)
    W = torch.diag_embed(weights)

    # Weighted design matrix multiplication
    D_weighted = torch.bmm(W, D)  # (B, N, 6)

    # Solve using SVD
    U, S, V = torch.svd(D_weighted)
    params = V[:, :, -1]  # (B, 6)

    # Normalize parameters per batch
    norm = torch.norm(params, dim=1, keepdim=True)
    params = params / (norm + 1e-8)

    return params

def ellipse_params(image_tensor):
    """
    Compute ellipse parameters for a batch of images.
    """
    masked_image = sigmoid_mask(image_tensor)
    points, weights = mask_to_points_and_weights_full(masked_image)
    params = weighted_ellipse_fit(points, weights)

    A, B, C, D, E, F = [params[:, i] for i in range(6)]

    denominator = 4*A*C - B**2
    cx = (B*E - 2*C*D) / denominator
    cy = (B*D - 2*A*E) / denominator

    theta = 0.5 * torch.atan2(B, A - C)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    a_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / \
                (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2)
    b_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / \
                (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2)

    a = torch.sqrt(torch.abs(a_squared))
    b = torch.sqrt(torch.abs(b_squared))

    return torch.stack([cx, cy, theta, a, b], dim=1)  # (B, 5)

def ellipse_loss(output_params, target_params, center_weight=1.0, angle_weight=1.0, axis_weight=1.0):
    """
    Computes normalized loss for a batch of ellipses.
    """
    # Unpack parameters
    cx_out, cy_out, theta_out, a_out, b_out = output_params.T
    cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params.T

    # Normalize center coordinates
    coord_scale = torch.max(torch.stack([a_tgt, b_tgt]), dim=1)[0].unsqueeze(1)
    normalized_center_loss = F.mse_loss(
        torch.stack([cx_out, cy_out], dim=1) / (coord_scale + 1e-8),
        torch.stack([cx_tgt, cy_tgt], dim=1) / (coord_scale + 1e-8)
    )

    # Angle loss using direction vectors
    def angle_to_vector(theta):
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    angle_vec_out = angle_to_vector(theta_out)
    angle_vec_tgt = angle_to_vector(theta_tgt)
    normalized_angle_loss = 0.5 * F.mse_loss(angle_vec_out, angle_vec_tgt)

    # Axis loss (normalized by larger target axis)
    axis_scale = torch.max(torch.stack([a_tgt, b_tgt]), dim=1)[0].unsqueeze(1)
    normalized_axis_loss = 0.5 * (
        F.l1_loss(a_out / (axis_scale + 1e-8), a_tgt / (axis_scale + 1e-8)) +
        F.l1_loss(b_out / (axis_scale + 1e-8), b_tgt / (axis_scale + 1e-8))
    )

    total_loss = (
        center_weight * normalized_center_loss +
        angle_weight * normalized_angle_loss +
        axis_weight * normalized_axis_loss
    )

    return total_loss

def eloss(output_batch, target_batch):
    """
    Compute ellipse loss for a batch of images.
    """
    return ellipse_loss(
        ellipse_params(transform_tensor(output_batch)),
        ellipse_params(transform_tensor(target_batch))
    )