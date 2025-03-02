import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from utils.fit_ellipse import transform_tensor_batched, safe_ellipse_params_batched, ellipse_fit_metric


# import utils.cadmos_lib as cl

def get_model_name(method, loss, filter='Laplacian', n_iters=8, llh='Gaussian', PnP=True, remove_SubNet=False):
    if method == 'Unrolled_ADMM':
        model_name = f'{llh}{"_PnP" if PnP else ""}_ADMM_{n_iters}iters{"_No_SubNet" if remove_SubNet else ""}' 
    elif method == 'Tikhonet' or method == 'ShapeNet':
        model_name = f'{method}_{filter}'
    else:
        model_name = method 
        
    if not method == 'ShapeNet':
        model_name = f'{model_name}_{loss}'
    
    return model_name

class BestEllipseLoss(nn.Module):
    def __init__(self, ellipse_levels=[0.3, 0.4, 0.5, 0.6, 0.7], 
                 center_weight=1.0, angle_weight=1.0, axis_weight=1.0):
        super(BestEllipseLoss, self).__init__()
        self.ellipse_levels = ellipse_levels
        self.num_ellipses = len(ellipse_levels)
        self.center_weight = center_weight
        self.angle_weight = angle_weight
        self.axis_weight = axis_weight
    
    def ellipse_loss_symmetric(self, output_params, target_params):
        """
        Compute symmetric loss between output and target ellipse parameters
        """
        # Unpack parameters
        cx_out, cy_out, theta_out, a_out, b_out = output_params.unbind(-1)
        cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params.unbind(-1)
        
        # Center loss - using MSE
        center_coords_out = torch.stack([cx_out, cy_out], dim=-1)
        center_coords_tgt = torch.stack([cx_tgt, cy_tgt], dim=-1)
        
        # Use maximum-based normalization for symmetric behavior
        out_max_axis = torch.max(torch.stack([a_out, b_out], dim=-1), dim=-1)[0]
        tgt_max_axis = torch.max(torch.stack([a_tgt, b_tgt], dim=-1), dim=-1)[0]
        coord_scale = torch.maximum(out_max_axis, tgt_max_axis).unsqueeze(-1) + 1e-8
        
        normalized_center_loss = F.mse_loss(
            center_coords_out / coord_scale.unsqueeze(-1),
            center_coords_tgt / coord_scale.unsqueeze(-1),
            reduction='none'
        ).mean(dim=-1)  # Mean across coordinates but keep batch dimension
        
        # Angle loss using cosine similarity
        angle_vec_out = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=-1)
        angle_vec_tgt = torch.stack([torch.cos(theta_tgt), torch.sin(theta_tgt)], dim=-1)
        
        # Compute cosine similarity (dot product of normalized vectors)
        cosine_sim = torch.sum(angle_vec_out * angle_vec_tgt, dim=-1)
        # Convert to loss (1 - cos_sim ranges from 0 to 2)
        normalized_angle_loss = 1 - cosine_sim
        
        # Axis loss with symmetric normalization
        axis_scale = torch.maximum(out_max_axis, tgt_max_axis).unsqueeze(-1) + 1e-8
        normalized_a_loss = ((a_out / axis_scale) - (a_tgt / axis_scale))**2
        normalized_b_loss = ((b_out / axis_scale) - (b_tgt / axis_scale))**2
        normalized_axis_loss = 0.5 * (normalized_a_loss + normalized_b_loss)
        
        # Combine losses (keeping batch dimension)
        total_loss = (
            self.center_weight * normalized_center_loss +
            self.angle_weight * normalized_angle_loss +
            self.axis_weight * normalized_axis_loss
        )
        
        return total_loss  # Shape: [batch_size]
    
    def forward(self, output, target):
        batch_size = output.shape[0]
        device = output.device
        
        # Transform tensors
        output_transformed = transform_tensor_batched(output)
        target_transformed = transform_tensor_batched(target)
        
        # Arrays to store ellipse parameters and fit metrics for each level
        gt_params_all_levels = []
        gt_fit_metrics = torch.zeros((batch_size, self.num_ellipses), device=device)
        
        # Step 1: Compute ellipse parameters and fit metrics for each peak position on the ground truth
        for i, pp in enumerate(self.ellipse_levels):
            # Extract ground truth ellipse parameters at current peak position level
            gt_params, _ = safe_ellipse_params_batched(target_transformed, peak_pos=pp)
            gt_params_all_levels.append(gt_params)
            
            # Compute how well this ellipse fits the ground truth image
            fit_metric = ellipse_fit_metric(target_transformed, gt_params)
            gt_fit_metrics[:, i] = fit_metric
        
        # Step 2: Find the best ellipse for each image in the batch
        best_ellipse_indices = torch.argmax(gt_fit_metrics, dim=1)  # Shape: [batch_size]
        
        # Step 3: Get the corresponding parameters for the best ellipses
        best_gt_params = torch.zeros((batch_size, 5), device=device)
        
        for b in range(batch_size):
            best_idx = best_ellipse_indices[b].item()
            best_gt_params[b] = gt_params_all_levels[best_idx][b]
        
        # Step 4: Compute output ellipse parameters using the same peak positions as the best ground truth ellipses
        output_params = torch.zeros((batch_size, 5), device=device)
        
        for b in range(batch_size):
            best_idx = best_ellipse_indices[b].item()
            pp = self.ellipse_levels[best_idx]
            # Extract a single image from the batch
            single_output = output_transformed[b:b+1]
            # Compute ellipse params for this single image with the best peak position
            params, _ = safe_ellipse_params_batched(single_output, peak_pos=pp)
            output_params[b] = params[0]  # Extract from batch dimension
        
        # Step 5: Compute loss between output and ground truth using the best ellipse parameters
        losses = self.ellipse_loss_symmetric(output_params, best_gt_params)
        
        # Return the mean loss across the batch
        return losses.mean()
    
class MomentBasedLoss(nn.Module):
    def __init__(self, normalize=True, central_moments_weight=1.0, centroid_weight=1.0):
        super(MomentBasedLoss, self).__init__()
        self.normalize = normalize
        self.central_moments_weight = central_moments_weight
        self.centroid_weight = centroid_weight
    
    def compute_moments(self, image_tensor):
        # Convert to grayscale if needed
        if image_tensor.dim() == 4 and image_tensor.shape[1] > 1:
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device)
            image_tensor = torch.einsum('bchw,c->bhw', image_tensor, rgb_weights)
        elif image_tensor.dim() == 4 and image_tensor.shape[1] == 1:
            image_tensor = image_tensor.squeeze(1)
        
        B, H, W = image_tensor.shape
        device = image_tensor.device
        
        # Normalize image if requested - make sure this creates a new tensor
        if self.normalize:
            # Assuming transform_tensor_batched returns a new tensor and doesn't modify in-place
            images = transform_tensor_batched(image_tensor)
        else:
            # Create a new tensor to avoid modifying the original
            images = image_tensor.clone()
        
        # Prepare coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Preallocate tensors for moments
        m00 = torch.zeros(B, device=device)
        centroids = torch.zeros((B, 2), device=device)  # [cy, cx]
        central_moments = torch.zeros((B, 3), device=device)  # [mu20, mu11, mu02]
        
        for i in range(B):
            img = images[i]
            
            # Zero-order moment (total intensity)
            m00[i] = torch.sum(img) + 1e-8
                
            # First-order moments (for centroid)
            m10 = torch.sum(img * x_coords)
            m01 = torch.sum(img * y_coords)
            
            # Centroid
            cx = m10 / m00[i]
            cy = m01 / m00[i]
            centroids[i, 0] = cy  # Store y-coordinate first to match ellipse params
            centroids[i, 1] = cx
            
            # Central moments - avoid in-place operations
            x_diff = x_coords - cx
            y_diff = y_coords - cy
            mu20 = torch.sum(img * x_diff.pow(2)) / m00[i]
            mu11 = torch.sum(img * x_diff * y_diff) / m00[i]
            mu02 = torch.sum(img * y_diff.pow(2)) / m00[i]
            
            central_moments[i, 0] = mu20
            central_moments[i, 1] = mu11
            central_moments[i, 2] = mu02
        
        return {
            'mass': m00,
            'centroids': centroids,
            'central_moments': central_moments
        }
    
    def forward(self, output, target):
        # Compute moments for both output and target
        output_moments = self.compute_moments(output)
        target_moments = self.compute_moments(target)
        
        # Centroid loss (direct comparison)
        centroid_loss = F.mse_loss(
            output_moments['centroids'],
            target_moments['centroids']
        )
        
        # Central moments loss
        central_moments_loss = F.mse_loss(
            output_moments['central_moments'],
            target_moments['central_moments']
        )
        
        # Total loss
        total_loss = (
            self.centroid_weight * centroid_loss + 
            self.central_moments_weight * central_moments_loss
        )
        
        return total_loss

class MultiScaleLoss(nn.Module):
    def __init__(self, scales=3, norm='L1', aux_loss_fn=None, aux_weight=0.1):
        super(MultiScaleLoss, self).__init__()
        self.scales = scales
        self.aux_loss_fn = aux_loss_fn
        self.aux_weight = aux_weight

        if norm == 'L1':
            self.loss = nn.L1Loss()
        elif norm == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported norm type. Use 'L1' or 'L2'.")

        self.weights = torch.FloatTensor([1 / (2 ** scale) for scale in range(self.scales)])
        self.multiscales = [nn.AvgPool2d(2 ** scale, 2 ** scale) for scale in range(self.scales)]

    def forward(self, output, target):
        loss = 0
        for i in range(self.scales):
            output_i, target_i = self.multiscales[i](output), self.multiscales[i](target)
            
            primary_loss = self.loss(output_i, target_i)
            
            aux_loss = self.aux_loss_fn(output_i, target_i) if self.aux_loss_fn else 0
            
            loss += self.weights[i] * (primary_loss + self.aux_weight * aux_loss)
        
        return loss

class ShapeConstraint(nn.Module):
    def __init__(self, device, fov_pixels=48, gamma=1, n_shearlet=2):
        super(ShapeConstraint, self).__init__()
        self.mse = nn.MSELoss()
        self.gamma = gamma
        U = cl.makeUi(fov_pixels, fov_pixels)
        shearlets, shearlets_adj = cl.get_shearlets(fov_pixels, fov_pixels, n_shearlet)
        # shealret adjoint of U, i.e Psi^{Star}(U)
        self.psu = np.array([cl.convolve_stack(ui, shearlets_adj) for ui in U])
        self.mu = torch.Tensor(cl.comp_mu(self.psu))
        self.mu = torch.Tensor(self.mu).to(device)
        self.psu = torch.Tensor(self.psu).to(device)
        
    def forward(self, output, target):
        loss = self.mse(output, target)
        for i in range(6):
            for j in range(self.psu.shape[1]):
                loss += self.gamma * self.mu[i,j] * (F.l1_loss(output*self.psu[i,j], target*self.psu[i,j]) ** 2) / 2.
        return loss
    
    
if __name__ == "__main__":
    print(get_model_name('ResUNet', 'MSE'))
    
    