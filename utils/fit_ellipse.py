import torch

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

def plot_galaxy_ellipse(image: torch.Tensor, params: torch.Tensor, ax=None):
    """
    Plot the galaxy image with fitted ellipse overlay.
    
    Parameters:
        image (torch.Tensor): Original galaxy image
        params (torch.Tensor): Fitted ellipse parameters [A, B, C, D, E, F]
        ax (matplotlib.axes.Axes, optional): The axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the galaxy image
    ax.imshow(image.cpu().numpy(), cmap='gray', origin='lower')
    
    # Convert parameters to tensors
    A, B, C, D, E, F = [torch.tensor(x) for x in params.tolist()]
    
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
    
    # Generate ellipse points
    t = torch.linspace(0, 2*np.pi, 200)
    x_circle = a * torch.cos(t)
    y_circle = b * torch.sin(t)
    
    # Rotate and translate
    R = torch.tensor([[cos_t, -sin_t],
                     [sin_t, cos_t]])
    points = torch.stack([x_circle, y_circle])
    rotated_points = R @ points
    x = rotated_points[0] + cx
    y = rotated_points[1] + cy
    
    # Plot ellipse overlay
    ax.plot(y.numpy(), x.numpy(), 'r-', label='Fitted Ellipse', linewidth=2)
    ax.scatter(cy.item(), cx.item(), color='yellow', marker='+', s=100, label='Center')
    
    # Customize plot
    ax.set_title('Galaxy Image with Fitted Ellipse')
    ax.legend()
    return ax

def ellipse_params(image_tensor):
    gray_image = torch.einsum('chw,c->hw', image_tensor.detach(), torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device))
    masked_image = sigmoid_mask(gray_image)
    points, weights = mask_to_points_and_weights_full(masked_image)
    params = weighted_ellipse_fit(points, weights)

    # Convert parameters to tensors
    A, B, C, D, E, F = [torch.tensor(x) for x in params.tolist()]
    
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

    return (cx, cy), theta, a, b

def ellipse_loss(output_params, target_params, center_weight=1.0, angle_weight=1.0, axis_weight=1.0, epsilon=1e-6):
    """
    Computes normalized loss between output and target ellipse parameters.
    
    Args:
    - output_params: Tuple (cx_out, cy_out, theta_out, a_out, b_out)
    - target_params: Tuple (cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt)
    - center_weight: Weight for center loss
    - angle_weight: Weight for angle loss
    - axis_weight: Weight for axis loss
    - epsilon: Small constant for numerical stability
    
    Returns:
    - Total normalized loss (scalar tensor)
    """
    # Unpack parameters
    (cx_out, cy_out, theta_out, a_out, b_out) = output_params
    (cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt) = target_params
    
    # Compute absolute differences
    center_diff = torch.abs(torch.stack([cx_out, cy_out]) - torch.stack([cx_tgt, cy_tgt]))
    angle_diff = torch.abs(theta_out - theta_tgt)
    axis_diff = torch.abs(torch.stack([a_out, b_out]) - torch.stack([a_tgt, b_tgt]))

    # Normalize each term by its range to make them comparable
    norm_center = center_diff / (torch.abs(cx_tgt) + torch.abs(cy_tgt) + epsilon)
    norm_angle = angle_diff / (torch.abs(theta_tgt) + epsilon)
    norm_axis = axis_diff / (torch.abs(a_tgt) + torch.abs(b_tgt) + epsilon)

    # Compute loss terms
    center_loss = torch.mean(norm_center)
    angle_loss = torch.mean(norm_angle)
    axis_loss = torch.mean(norm_axis)

    # Weighted sum of losses
    total_loss = center_weight * center_loss + angle_weight * angle_loss + axis_weight * axis_loss
    
    return total_loss