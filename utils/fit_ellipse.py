import torch
import matplotlib.pyplot as plt
import numpy as np
import math

def transform_tensor_batched(tensor):
    if tensor.dim() == 3: 
        transformed_tensor = tensor.clone()
    elif tensor.dim() == 4: 
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device)
        transformed_tensor = torch.einsum('bchw,c->bhw', tensor, rgb_weights)
    
    if transformed_tensor.max() > 1.0:
        transformed_tensor = transformed_tensor / 255.0

    min_val = transformed_tensor.view(transformed_tensor.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    max_val = transformed_tensor.view(transformed_tensor.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    valid_range = (max_val > min_val).float()
    transformed_tensor = valid_range * (transformed_tensor - min_val) / (max_val - min_val + 1e-8) + (1 - valid_range) * transformed_tensor
    
    transformed_tensor.requires_grad_(True)
    return transformed_tensor

def mask_batched(x: torch.Tensor, 
                        peak_pos: float = 0.5, 
                        sharpness: float = 0.1) -> torch.Tensor:
    return 1 / torch.exp(((x - peak_pos) / sharpness) ** 2)

def mask_to_points_and_weights_batched(mask):
    B, H, W = mask.shape
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=mask.device), 
                                      torch.arange(W, device=mask.device))
    
    points_grid = torch.stack((y_coords, x_coords), dim=-1).float()
    
    points = points_grid.unsqueeze(0).expand(B, H, W, 2)
    
    points = points.reshape(B, H*W, 2)
    
    weights = mask.reshape(B, H*W)
    
    return points, weights

def weighted_samsons_distance_batched(points, coeffs, weights):
    y, x = points[..., 0], points[..., 1] 
    
    A = coeffs[:, 0:1]
    B = coeffs[:, 1:2]
    C = coeffs[:, 2:3]
    D = coeffs[:, 3:4]
    E = coeffs[:, 4:5]
    F = coeffs[:, 5:6]

    algebraic_dist = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F  # (B, N)

    grad_x = 2*A*x + B*y + D  # (B, N)
    grad_y = B*x + 2*C*y + E  # (B, N)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # (B, N)

    samsons_dist = torch.abs(algebraic_dist) / (grad_magnitude + 1e-8)  # (B, N)
    
    weighted_samsons_dist = samsons_dist * weights  # (B, N)

    return weighted_samsons_dist

def weighted_ellipse_fit_batched(points, weights):
    B, N, _ = points.shape
    
    x = points[..., 0]
    y = points[..., 1]
    
    D = torch.stack((x**2, x*y, y**2, x, y, torch.ones_like(x)), dim=-1)
    
    D_weighted = D * weights.unsqueeze(-1)
    
    U, S, V = torch.svd(D_weighted)
    
    params = V[..., -1]
    
    norm = torch.norm(params, dim=-1, keepdim=True)
    params = params / (norm + 1e-8)
    
    return params

def ellipse_params_batched(image_tensor, peak_pos: float = 0.5, sharpness: float = 0.1):
    masked_image = mask_batched(image_tensor, peak_pos=peak_pos, sharpness=sharpness)
    points, weights = mask_to_points_and_weights_batched(masked_image)
    params = weighted_ellipse_fit_batched(points, weights)
    weighted_samsons_dist = weighted_samsons_distance_batched(points, params, weights)  # (B, H*W)
    
    A, B, C, D, E, F = params.unbind(-1)
    
    denominator = 4*A*C - B**2
    cx = (B*E - 2*C*D) / (denominator + 1e-8)
    cy = (B*D - 2*A*E) / (denominator + 1e-8)
    theta = 0.5 * torch.atan2(B, A - C)
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    expr1 = A*cx**2 + C*cy**2 + B*cx*cy + D*cx + E*cy + F
    a_squared = -2 * expr1 / (A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2 + 1e-8)
    b_squared = -2 * expr1 / (A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2 + 1e-8)
    
    a = torch.sqrt(torch.abs(a_squared))
    b = torch.sqrt(torch.abs(b_squared))

    a_expanded = a.unsqueeze(1)
    
    normalized_dist = weighted_samsons_dist / (a_expanded + 1e-8)
    
    total_weighted_dist = torch.sum(normalized_dist * weights, dim=1)
    total_weight = torch.sum(weights, dim=1)
    mean_normalized_samsons_dist = total_weighted_dist / (total_weight + 1e-8)
    
    return torch.stack([cx, cy, theta, a, b], dim=-1), mean_normalized_samsons_dist

def safe_ellipse_params_batched(image_tensor, peak_pos=0.5, sharpness=0.1):
    B = image_tensor.shape[0]
    device = image_tensor.device
    
    all_params = torch.zeros((B, 5), device=device)
    all_confidence = torch.zeros(B, device=device)
    
    for i in range(B):
        try:
            single_image = image_tensor[i:i+1]
            params, confidence = ellipse_params_batched(single_image, peak_pos, sharpness)
            
            if (torch.isnan(params).any() or torch.isinf(params).any() or 
                torch.isnan(confidence).any() or torch.isinf(confidence).any()):
                raise ValueError("NaN or Inf values detected in output")
                
            all_params[i] = params[0]
            all_confidence[i] = confidence[0]
            
        except Exception as e:
            print(f"Error in ellipse fitting for batch element {i}: {str(e)}")
            
            default_params = torch.tensor([
                image_tensor.shape[2] / 2,
                image_tensor.shape[1] / 2, 
                0.0,
                10.0,
                10.0
            ], device=device).detach()
            
            all_params[i] = default_params
            all_confidence[i] = torch.tensor(1.0, device=device).detach()
    
    return all_params, all_confidence

def plot_batch_with_ellipses(images, ellipses_params, num_cols=2, figsize=None):
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
        
        ax.imshow(images_np[img_idx], cmap='gray', extent=[0, 40, 40, 0])

        ax.set_xlim(0, 40)
        ax.set_ylim(40, 0) 
        
        color_codes = ['r', 'g', 'b', 'c', 'm', 'y']
        
        for param_idx, ellipse_params in enumerate(ellipses_params):
            params_np = ellipse_params.detach().cpu().numpy()
            cx, cy, theta, a, b = params_np[img_idx]
            
            height, width = images_np[img_idx].shape
            scale_x = 40 / width
            scale_y = 40 / height
            
            cx = cx * scale_y  
            cy = cy * scale_x  
            a = a * scale_y    
            b = b * scale_x    
            
            x = a * np.cos(t)
            y = b * np.sin(t)
            
            R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
            points = np.dot(np.stack([x, y], axis=1), R.T)
            points[:, 0] += cx
            points[:, 1] += cy
            
            ax.plot(points[:, 1], points[:, 0], color_codes[param_idx%len(color_codes)]+'-', linewidth=2)
            ax.plot(cy, cx, color_codes[param_idx%len(color_codes)]+'+', markersize=10)
        
        ax.set_title(f'Image {img_idx}')
        ax.grid(True) 
        
    for idx in range(batch_size, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, axes
    
def ellipse_fit_metric(image_tensor, ellipse_params):
    """
    Computes a normalized metric (0 to 1) indicating how well an ellipse fits a galaxy.
    Higher values indicate better fit (more intensity inside ellipse, less outside).
    Uses whole pixel counting and distance-weighted intensity for better center rewards.
    Only considers pixels within the image boundaries.
    
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
    if image_tensor.dim() == 4:
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device)
        image = torch.einsum('bchw,c->bhw', image_tensor, rgb_weights)
    else:
        image = image_tensor
        
    B, H, W = image.shape
    device, dtype = image.device, image.dtype
        
    cy = ellipse_params[:, 0].view(B, 1, 1)
    cx = ellipse_params[:, 1].view(B, 1, 1)
    theta = ellipse_params[:, 2].view(B, 1, 1)
    a = ellipse_params[:, 3].view(B, 1, 1)
    b = ellipse_params[:, 4].view(B, 1, 1)
        
    y_indices, x_indices = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    y_indices = y_indices.unsqueeze(0).expand(B, -1, -1)
    x_indices = x_indices.unsqueeze(0).expand(B, -1, -1)
    
    valid_mask = torch.ones_like(y_indices, dtype=torch.bool)
        
    x_trans = x_indices - cx
    y_trans = y_indices - cy
        
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = x_trans * cos_theta + y_trans * sin_theta
    y_rot = -x_trans * sin_theta + y_trans * cos_theta
        
    ellipse_eq = (x_rot / a)**2 + (y_rot / b)**2
        
    inside_mask = (ellipse_eq <= 1.0).float() * valid_mask.float()
    
    outside_mask = valid_mask.float() - inside_mask
    
    distance = torch.sqrt(((x_indices - cx) / a)**2 + ((y_indices - cy) / b)**2)
    
    distance_weight = torch.clamp(1.0 - distance * 0.5, min=0.5, max=1.0)
    
    weighted_inside = image * inside_mask * distance_weight
    weighted_inside_sum = torch.sum(weighted_inside, dim=(1, 2))
    
    inside_count = torch.sum(inside_mask, dim=(1, 2))
    outside_count = torch.sum(outside_mask, dim=(1, 2))
    
    outside_intensity = torch.sum(image * outside_mask, dim=(1, 2))
    
    eps = 1e-8 
    
    zero_inside = inside_count < eps
    zero_outside = outside_count < eps
    
    inside_density = weighted_inside_sum / (inside_count + eps)
    outside_density = outside_intensity / (outside_count + eps)
    
    contrast_ratio = torch.zeros_like(inside_density)
    
    normal_case = (~zero_inside & ~zero_outside)
    contrast_ratio[normal_case] = inside_density[normal_case] / (outside_density[normal_case] + eps)
    
    contrast_ratio[~zero_inside & zero_outside] = 10.0 
    
    contrast_ratio[zero_inside] = 0.0
    
    normalized_score = contrast_ratio / (1.0 + contrast_ratio)
    return normalized_score

def normalize_images(batch):
    """
    Scale pixel values in each image of the batch to range [0, 1].
    
    Args:
        batch: Tensor of shape [batch_size, channels, height, width]
        
    Returns:
        Normalized tensor with same shape but values scaled to [0, 1]
    """
    if len(batch.shape) == 3:
        batch = batch.unsqueeze(1)

    batch_size, channels, height, width = batch.shape
    reshaped = batch.view(batch_size, channels, -1)
    
    min_vals = reshaped.min(dim=2, keepdim=True)[0]
    max_vals = reshaped.max(dim=2, keepdim=True)[0]
    
    min_vals = min_vals.unsqueeze(-1) 
    max_vals = max_vals.unsqueeze(-1)  
    
    divisor = torch.maximum(max_vals - min_vals, torch.ones_like(max_vals) * 1e-8)
    
    normalized = (batch - min_vals) / divisor
    
    return normalized

def compute_moments(image_tensor):
    """
    Compute image moments up to order 3 for a batch of images.
    
    Args:
        image_tensor: Tensor of shape [B, C, H, W]
        
    Returns:
        moments_dict: Dictionary containing moment values for each image in the batch
    """
    image_tensor = normalize_images(image_tensor)
    
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    batch_moments = []
    
    for i in range(B):
        img = image_tensor[i]
        
        m00 = torch.sum(img) + 1e-8
            
        m10 = torch.sum(img * x_coords)
        m01 = torch.sum(img * y_coords)
        
        cx = m10 / m00
        cy = m01 / m00
        
        mu20 = torch.sum(img * (x_coords - cx)**2) / m00
        mu11 = torch.sum(img * (x_coords - cx) * (y_coords - cy)) / m00
        mu02 = torch.sum(img * (y_coords - cy)**2) / m00
        
        mu30 = torch.sum(img * (x_coords - cx)**3) / m00
        mu21 = torch.sum(img * (x_coords - cx)**2 * (y_coords - cy)) / m00
        mu12 = torch.sum(img * (x_coords - cx) * (y_coords - cy)**2) / m00
        mu03 = torch.sum(img * (y_coords - cy)**3) / m00
        
        moments = {
            'm00': m00,
            
            'cx': cx,
            'cy': cy,
            
            'mu20': mu20,
            'mu11': mu11,
            'mu02': mu02,
            
            'mu30': mu30,
            'mu21': mu21,
            'mu12': mu12,
            'mu03': mu03
        }
        
        batch_moments.append(moments)
    
    return batch_moments

def ellipse_params_from_moments(image_tensor):
    """
    Calculate ellipse parameters from image moments for a batch of images.
    
    Args:
        image_tensor: Tensor of shape [B, C, H, W]
        
    Returns:
        ellipse_params: Tensor of shape [B, 5] containing [cy, cx, theta, a, b] for each image
    """
    batch_moments = compute_moments(image_tensor)
    
    B = len(batch_moments)
    device = image_tensor.device
    
    ellipse_params = torch.zeros((B, 5), device=device)
    
    for i in range(B):
        moments = batch_moments[i]
        
        cy = moments['cy']
        cx = moments['cx']
        mu20 = moments['mu20']
        mu11 = moments['mu11']
        mu02 = moments['mu02']
        
        delta = mu20 - mu02
        theta = 0.5 * torch.atan2(2 * mu11, delta + 1e-8)
        
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        disc = torch.sqrt(trace * trace - 4 * det + 1e-8)
        
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)
        
        lambda1 = torch.maximum(lambda1, torch.tensor(1e-6, device=device))
        lambda2 = torch.maximum(lambda2, torch.tensor(1e-6, device=device))
        
        a = torch.sqrt(lambda1)
        b = torch.sqrt(lambda2)
        
        ellipse_params[i] = torch.tensor([cy, cx, theta, a, b], device=device)
    
    return ellipse_params

def laguerre_torch(n, x):
    if n == 2:
        return (x**2 - 4 * x + 2) / 2.0
    elif n == 4:
        return (x**4 - 16 * x**3 + 72 * x**2 - 96 * x + 24) / 24.0
    else:
        raise NotImplementedError("Only n=2 and 4 are supported.")

def laguerre_shapelet(n, x):
    """
    Compute the Laguerre shapelet function.
    
    Args:
        n: Order of the Laguerre polynomial
        x: Input tensor
        
    Returns:
        Laguerre shapelet function value
    """
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    
    if torch.isnan(x).any():
        print(f"NaN detected in input to laguerre_shapelet for n={n}")
        return torch.zeros_like(x)
        
    L_n = laguerre_torch(n, x)
    
    if torch.isnan(L_n).any():
        print(f"NaN detected in Laguerre polynomial L_{n} output")
        return torch.zeros_like(x)
    
    factorial_n = math.factorial(n)
    result = (1 / math.sqrt(factorial_n)) * torch.exp(-x / 2) * L_n
    
    if torch.isnan(result).any():
        print(f"NaN detected in laguerre_shapelet final result for n={n}")
        print(f"Factor: {1 / math.sqrt(factorial_n)}")
        print(f"Min/max of x: {x.min().item()}, {x.max().item()}")
        return torch.zeros_like(x)
        
    return result

def compute_shapelet_moments(image_tensor, max_order=4):
    """
    Compute Laguerre shapelet moments up to specified order for a batch of images.
    
    Args:
        image_tensor: Tensor of shape [B, C, H, W]
        max_order: Maximum order of moments to compute (default=4)
        
    Returns:
        shapelet_moments_dict: Dictionary containing shapelet moment values for each image
    """
    if torch.isnan(image_tensor).any():
        print("NaN detected in input to compute_shapelet_moments")
        for b in range(image_tensor.shape[0]):
            if torch.isnan(image_tensor[b]).any():
                print(f"  NaN found in batch element {b}")
    
    image_tensor = normalize_images(image_tensor)
    
    if torch.isnan(image_tensor).any():
        print("NaN detected after normalize_images in compute_shapelet_moments")
    
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    if C == 1:
        image_tensor = image_tensor.squeeze(1)
    else:
        image_tensor = image_tensor.mean(dim=1)
    
    if torch.isnan(image_tensor).any():
        print("NaN detected after channel handling in compute_shapelet_moments")
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    batch_centroids = []
    for i in range(B):
        img = image_tensor[i]
        
        if torch.isnan(img).any():
            print(f"NaN detected in image {i} before centroid calculation")
            
        m00 = torch.sum(img) + 1e-8
        
        if m00 < 1e-6 or torch.isnan(m00):
            print(f"Warning: Very small or NaN m00 ({m00.item()}) for image {i}")
            
        m10 = torch.sum(img * x_coords)
        m01 = torch.sum(img * y_coords)
        
        cx = m10 / m00
        cy = m01 / m00
        
        if torch.isnan(cx) or torch.isnan(cy):
            print(f"NaN detected in centroids for image {i}: cx={cx.item()}, cy={cy.item()}")
            print(f"  m00={m00.item()}, m10={m10.item()}, m01={m01.item()}")
            cx = H / 2
            cy = W / 2
        
        mu20 = torch.sum(img * (x_coords - cx)**2) / m00
        mu02 = torch.sum(img * (y_coords - cy)**2) / m00
        
        if torch.isnan(mu20) or torch.isnan(mu02):
            print(f"NaN detected in moments for image {i}: mu20={mu20.item()}, mu02={mu02.item()}")
            mu20 = torch.tensor(1.0, device=device)
            mu02 = torch.tensor(1.0, device=device)
            
        radius = 2 * torch.sqrt(mu20 + mu02)
        
        if radius < 1e-6 or torch.isnan(radius):
            print(f"Warning: Very small or NaN radius ({radius.item()}) for image {i}")
            radius = torch.tensor(1.0, device=device)
            
        batch_centroids.append((cx, cy, radius))
    
    batch_shapelet_moments = []
    
    for i in range(B):
        img = image_tensor[i]
        cx, cy, radius = batch_centroids[i]
        
        moments = {}
        
        r_squared = ((x_coords - cx)**2 + (y_coords - cy)**2) / (radius**2 + 1e-8)
        
        if torch.isnan(r_squared).any():
            print(f"NaN detected in r_squared for image {i}")
            print(f"  cx={cx.item()}, cy={cy.item()}, radius={radius.item()}")
            continue
            
        r_squared.requires_grad_(True)
        
        for n in range(max_order + 1):
            if n == 2:
                shapelet_values = laguerre_shapelet(2, r_squared)
                
                if torch.isnan(shapelet_values).any():
                    print(f"NaN detected in S2 shapelet values for image {i}")
                    shapelet_values = torch.zeros_like(r_squared)
                
                moment_value = torch.sum(img * shapelet_values)
                
                if torch.isnan(moment_value):
                    print(f"NaN detected in S2 moment value for image {i}")
                    moment_value = torch.tensor(0.0, device=device, requires_grad=True)
                    
                moments[f'S2'] = moment_value
            
            elif n == 4:
                shapelet_values = laguerre_shapelet(4, r_squared)
                
                if torch.isnan(shapelet_values).any():
                    print(f"NaN detected in S4 shapelet values for image {i}")
                    shapelet_values = torch.zeros_like(r_squared)
                
                moment_value = torch.sum(img * shapelet_values)
                
                if torch.isnan(moment_value):
                    print(f"NaN detected in S4 moment value for image {i}")
                    moment_value = torch.tensor(0.0, device=device, requires_grad=True)
                    
                moments[f'S4'] = moment_value
        
        batch_shapelet_moments.append(moments)
    
    return batch_shapelet_moments

def visualize_shapelet_basis(orders=[2, 4], num_points=100):
    """
    Visualize the Laguerre shapelet basis functions.
    
    Args:
        orders: Orders to display
        num_points: Number of points for visualization
    """
    r_vals = torch.linspace(0, 5, num_points, requires_grad=True)
    
    plt.figure(figsize=(10, 6))
    for n in orders:
        shapelet_vals = []
        for r in r_vals:
            val = laguerre_shapelet(n, torch.tensor([r**2], requires_grad=True))
            shapelet_vals.append(val.item())
        
        plt.plot(r_vals.detach().numpy(), shapelet_vals, label=f'Ψ_{n}(r²)')
    
    plt.xlabel('r² (Squared Radial Distance)')
    plt.ylabel('Shapelet Value')
    plt.title('Laguerre Shapelet Basis Functions')
    plt.legend()
    plt.grid(True)
    plt.show()
