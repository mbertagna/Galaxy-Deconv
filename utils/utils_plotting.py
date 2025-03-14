import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_multiscale_batch(model, batch_obs, batch_psf, batch_alpha, batch_gt, example_batch_idxs):
    # Run model on batch
    with torch.no_grad():
        # output_batch = model(batch_obs, batch_psf, batch_alpha)
        output_batch = batch_obs
    
    # Get the multiscale functions and weights from the loss
    multiscales = [torch.nn.AvgPool2d(2 ** scale, 2 ** scale) for scale in range(3)]
    weights = [1 / (2 ** scale) for scale in range(3)]
    loss_fn = torch.nn.L1Loss(reduction='none')  # Use 'none' to get per-element losses
    
    # Create figure
    fig = plt.figure(figsize=(20, 5 * len(example_batch_idxs)))
    
    for idx, galaxy_idx in enumerate(example_batch_idxs):
        # Original images
        output = output_batch[idx].cpu() * batch_alpha[idx].cpu()
        gt = batch_gt[idx].cpu()
        
        # Row for each galaxy
        for scale in range(3):
            # Get downscaled images
            output_scaled = multiscales[scale](output)
            gt_scaled = multiscales[scale](gt)
            
            # Calculate loss
            loss_map = loss_fn(output_scaled, gt_scaled).squeeze().detach()
            weighted_loss = weights[scale] * loss_map
            avg_loss = loss_map.mean().item()
            weighted_avg = weighted_loss.mean().item()
            
            # Convert to numpy for plotting
            output_scaled = output_scaled.squeeze().detach().numpy()
            gt_scaled = gt_scaled.squeeze().detach().numpy()
            loss_map = loss_map.numpy()
            weighted_loss = weighted_loss.numpy()
            
            # Plot output
            ax = fig.add_subplot(len(example_batch_idxs), 9, idx*9 + scale*3 + 1)
            im = ax.imshow(output_scaled, cmap='magma')
            ax.set_title(f'Output Scale {scale}')
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot ground truth
            ax = fig.add_subplot(len(example_batch_idxs), 9, idx*9 + scale*3 + 2)
            im = ax.imshow(gt_scaled, cmap='magma')
            ax.set_title(f'GT Scale {scale}')
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot loss map
            ax = fig.add_subplot(len(example_batch_idxs), 9, idx*9 + scale*3 + 3)
            im = ax.imshow(weighted_loss, cmap='hot')
            ax.set_title(f'Loss Scale {scale}\nWeight: {weights[scale]:.4f}\nAvg: {avg_loss:.4f}\nWeighted: {weighted_avg:.4f}')
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    # plt.savefig('multiscale_loss_comparison.png', dpi=300)
    plt.show()

def plot_batched_images(images, titles=None, figsize=(15, 5), cmap='magma'):
    """
    Plot a batch of images in a single row.
    
    Args:
        images: Tensor of shape (B, H, W) or list of B tensors each of shape (H, W)
        titles: Optional list of B titles
        figsize: Figure size
        cmap: Colormap to use
    """
    if isinstance(images, torch.Tensor):
        images = [img for img in images]
    
    batch_size = len(images)
    
    fig, axs = plt.subplots(1, batch_size, figsize=figsize)
    if batch_size == 1:
        axs = [axs]
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        axs[i].imshow(img, cmap=cmap)
        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i])
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig, axs

def plot_batched_points(points, weights, threshold=0.5, figsize=(15, 5), 
                        titles=None, color_by_weight=True):
    """
    Plot a batch of points with weights above threshold.
    
    Args:
        points: Tensor of shape (B, N, 2) with point coordinates
        weights: Tensor of shape (B, N) with point weights
        threshold: Minimum weight to plot a point
        figsize: Figure size
        titles: Optional list of B titles
        color_by_weight: Whether to color points by their weights
    """
    batch_size = points.shape[0]
    
    fig, axs = plt.subplots(1, batch_size, figsize=figsize)
    if batch_size == 1:
        axs = [axs]
    
    for i in range(batch_size):
        # Get batch item
        batch_points = points[i].detach().cpu().numpy()
        batch_weights = weights[i].detach().cpu().numpy()
        
        # Filter points by threshold
        mask = batch_weights > threshold
        filtered_points = batch_points[mask]
        filtered_weights = batch_weights[mask]
        
        # Plot points
        if color_by_weight:
            scatter = axs[i].scatter(filtered_points[:, 1], filtered_points[:, 0], 
                                   c=filtered_weights, cmap='Blues', 
                                   s=30, alpha=0.8)
            # plt.colorbar(scatter, ax=axs[i])
        else:
            axs[i].scatter(filtered_points[:, 1], filtered_points[:, 0], 
                         s=30, alpha=0.8)
        
        # Set title and limits
        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i])
        axs[i].set_xlim(0, batch_points.max())
        axs[i].set_ylim(batch_points.max(), 0)  # Inverted y-axis to match image coordinates
        axs[i].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, axs

def plot_batch_with_ellipses(images, ellipses_params, titles=None, figsize=(15, 5), cmap='magma'):
    # Ensure images are (B, H, W)
    if images.dim() == 4:  # (B, C, H, W)
        images = images.mean(dim=1)
    
    batch_size = images.shape[0]
    images_np = images.detach().cpu().numpy()
    
    fig, axs = plt.subplots(1, batch_size, figsize=figsize)
    if batch_size == 1:
        axs = [axs]
    
    # Generate ellipse points
    t = np.linspace(0, 2*np.pi, 100)
    color_codes = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for i in range(batch_size):
        # Get image dimensions
        height, width = images_np[i].shape
        
        # Plot the image with explicit extent
        axs[i].imshow(images_np[i], cmap=cmap, extent=[0, width-1, height-1, 0])
        
        # Plot each ellipse
        for j, ellipse_params in enumerate(ellipses_params):
            params = ellipse_params[i].detach().cpu().numpy()
            cx, cy, theta, a, b = params
            
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
            color = color_codes[j % len(color_codes)]
            axs[i].plot(points[:, 1], points[:, 0], color+'-', linewidth=2)
            axs[i].plot(cy, cx, color+'+', markersize=10)
        
        # Set title
        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i])
        # else:
            # axs[i].set_title(f'Image {i}')
            
        # Set axes limits precisely
        axs[i].set_xlim(-0.5, width-0.5)
        axs[i].set_ylim(height-0.5, -0.5)
    
    plt.tight_layout()
    return fig, axs