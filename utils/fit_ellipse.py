import torch
import torch.nn.functional as F

def transform_tensor(tensor):
    if tensor.dim() == 3:
        transformed_tensor = tensor.clone()
    elif tensor.dim() == 4 and tensor.shape[1] in {1, 3}: 
        transformed_tensor = tensor.mean(dim=1)
    
    if transformed_tensor.max() > 1.0:
        transformed_tensor = transformed_tensor / 255.0
    
    min_val, max_val = transformed_tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0], \
                       transformed_tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    transformed_tensor = (transformed_tensor - min_val) / (max_val - min_val + 1e-8)
    transformed_tensor.requires_grad_(True)
    return transformed_tensor

def mask_to_points_and_weights_full(mask):
    B, H, W = mask.shape
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=mask.device), torch.arange(W, device=mask.device), indexing='ij')
    points = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).float().unsqueeze(0).repeat(B, 1, 1)
    weights = mask.view(B, -1)
    return points, weights

def ellipse_params(image_tensor):
    masked_image = sigmoid_mask(image_tensor)
    points, weights = mask_to_points_and_weights_full(masked_image)
    params = weighted_ellipse_fit(points, weights)

    A, B, C, D, E, F = params.unbind(dim=1)
    denominator = 4*A*C - B**2
    cx = (B*E - 2*C*D) / (denominator + 1e-8)
    cy = (B*D - 2*A*E) / (denominator + 1e-8)
    theta = 0.5 * torch.atan2(B, A - C)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    
    a_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2 + 1e-8)
    b_squared = -2 * (A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F) / (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2 + 1e-8)
    
    a = torch.sqrt(torch.abs(a_squared))
    b = torch.sqrt(torch.abs(b_squared))
    return torch.stack([cx, cy, theta, a, b], dim=1)

def ellipse_loss(output_params, target_params, center_weight=1.0, angle_weight=1.0, axis_weight=1.0):
    cx_out, cy_out, theta_out, a_out, b_out = output_params.unbind(dim=1)
    cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params.unbind(dim=1)
    
    coord_scale = torch.max(torch.stack([a_tgt, b_tgt], dim=1), dim=1, keepdim=True)[0]
    normalized_center_loss = F.mse_loss(
        torch.stack([cx_out, cy_out], dim=1) / (coord_scale + 1e-8),
        torch.stack([cx_tgt, cy_tgt], dim=1) / (coord_scale + 1e-8),
        reduction='none'
    ).sum(dim=1)
    
    angle_vec_out = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=1)
    angle_vec_tgt = torch.stack([torch.cos(theta_tgt), torch.sin(theta_tgt)], dim=1)
    normalized_angle_loss = 0.5 * F.mse_loss(angle_vec_out, angle_vec_tgt, reduction='none').sum(dim=1)
    
    axis_scale = torch.max(torch.stack([a_tgt, b_tgt], dim=1), dim=1, keepdim=True)[0]
    normalized_axis_loss = 0.5 * (
        F.l1_loss(a_out / (axis_scale + 1e-8), a_tgt / (axis_scale + 1e-8), reduction='none') +
        F.l1_loss(b_out / (axis_scale + 1e-8), b_tgt / (axis_scale + 1e-8), reduction='none')
    ).sum(dim=1)
    
    total_loss = (
        center_weight * normalized_center_loss +
        angle_weight * normalized_angle_loss +
        axis_weight * normalized_axis_loss
    )
    return total_loss.mean()

def eloss(output_i, target_i):
    return ellipse_loss(
        ellipse_params(transform_tensor(output_i)),
        ellipse_params(transform_tensor(target_i))
    )