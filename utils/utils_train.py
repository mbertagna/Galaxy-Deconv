import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from utils.fit_ellipse import transform_tensor_batched, ellipse_params_batched


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

class MultiEllipseLoss(nn.Module):
    def __init__(self, ellipse_levels=[0.3, 0.4, 0.5, 0.6, 0.7], center_weight=1.0, angle_weight=1.0, axis_weight=1.0, 
                 ellipse_weights=None, loss_aggregation='weighted_sum'):
        """
        Loss function that handles multiple ellipse predictions at different peak positions.
        
        Args:
            ellipse_levels: List of peak position thresholds for ellipse detection
            center_weight: Weight for center position loss
            angle_weight: Weight for angle loss
            axis_weight: Weight for axis length loss
            ellipse_weights: Optional weights for different ellipses (None for equal weighting)
            loss_aggregation: How to aggregate losses across ellipses ('weighted_sum', 'min', 'max')
        """
        super(MultiEllipseLoss, self).__init__()
        self.ellipse_levels = ellipse_levels
        self.num_ellipses = len(ellipse_levels)
        self.center_weight = center_weight
        self.angle_weight = angle_weight
        self.axis_weight = axis_weight
        
        # Default to equal weighting if not specified
        if ellipse_weights is None:
            self.ellipse_weights = torch.ones(self.num_ellipses) / self.num_ellipses
        else:
            self.ellipse_weights = torch.tensor(ellipse_weights)
            self.ellipse_weights = self.ellipse_weights / self.ellipse_weights.sum()  # Normalize
            
        self.loss_aggregation = loss_aggregation
            
    def ellipse_loss_symmetric(self, output_params, target_params):
        """
        Symmetric version of ellipse loss using maximum-based normalization.
        
        Args:
            output_params: Tensor of shape (B, 5) containing predicted ellipse parameters
            target_params: Tensor of shape (B, 5) containing target ellipse parameters
            
        Returns:
            Scalar loss value
        """
        # Unpack parameters
        cx_out, cy_out, theta_out, a_out, b_out = output_params.unbind(-1)
        cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params.unbind(-1)
        
        # Center loss
        center_coords_out = torch.stack([cx_out, cy_out], dim=-1)
        center_coords_tgt = torch.stack([cx_tgt, cy_tgt], dim=-1)
        
        # Use maximum-based normalization for symmetric behavior
        out_max_axis = torch.max(torch.stack([a_out, b_out], dim=-1), dim=-1)[0]
        tgt_max_axis = torch.max(torch.stack([a_tgt, b_tgt], dim=-1), dim=-1)[0]
        coord_scale = torch.maximum(out_max_axis, tgt_max_axis).unsqueeze(-1) + 1e-8
        
        normalized_center_loss = F.mse_loss(
            center_coords_out / coord_scale.unsqueeze(-1),
            center_coords_tgt / coord_scale.unsqueeze(-1)
        )
        
        # Angle loss using vector representation (already symmetric)
        angle_vec_out = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=-1)
        angle_vec_tgt = torch.stack([torch.cos(theta_tgt), torch.sin(theta_tgt)], dim=-1)
        normalized_angle_loss = 0.5 * F.mse_loss(angle_vec_out, angle_vec_tgt)
        
        # Axis loss with symmetric normalization
        axis_scale = torch.maximum(out_max_axis, tgt_max_axis).unsqueeze(-1) + 1e-8
        normalized_axis_loss = 0.5 * (
            F.l1_loss(a_out / axis_scale, a_tgt / axis_scale) +
            F.l1_loss(b_out / axis_scale, b_tgt / axis_scale)
        )
        
        # Combine losses
        total_loss = (
            self.center_weight * normalized_center_loss +
            self.angle_weight * normalized_angle_loss +
            self.axis_weight * normalized_axis_loss
        )
        
        return total_loss
    
    def forward(self, output, target):
        """
        Process output and target images to extract and compare ellipses at different levels.
        
        Args:
            output: Tensor of shape (B, C, H, W) containing predicted images
            target: Tensor of shape (B, C, H, W) containing target images
            
        Returns:
            Scalar loss value aggregating all ellipses
        """
        # Extract ellipse parameters for each level for both output and target
        output_params_list = []
        target_params_list = []
        
        for pp in self.ellipse_levels:
            # Extract ellipse parameters from output and target at current peak position level
            output_params, _ = ellipse_params_batched(transform_tensor_batched(output), peak_pos=pp)
            target_params, _ = ellipse_params_batched(transform_tensor_batched(target), peak_pos=pp)
            
            output_params_list.append(output_params)
            target_params_list.append(target_params)
        
        # Compute individual losses for each ellipse level
        individual_losses = []
        for i in range(self.num_ellipses):
            loss_i = self.ellipse_loss_symmetric(output_params_list[i], target_params_list[i])
            individual_losses.append(loss_i)
            
        # Convert to tensor
        individual_losses = torch.stack(individual_losses)
        
        # Aggregate losses based on selected method
        if self.loss_aggregation == 'weighted_sum':
            # Weighted sum of all ellipse losses
            final_loss = torch.sum(individual_losses * self.ellipse_weights.to(individual_losses.device))
            
        elif self.loss_aggregation == 'min':
            # Take the minimum loss (focus on best matching ellipse)
            final_loss = torch.min(individual_losses)
            
        elif self.loss_aggregation == 'max':
            # Take the maximum loss (focus on worst matching ellipse)
            final_loss = torch.max(individual_losses)
            
        elif self.loss_aggregation == 'median':
            # Take the median loss
            final_loss = torch.median(individual_losses)
            
        elif self.loss_aggregation == 'mean':
            # Simple average
            final_loss = torch.mean(individual_losses)
            
        elif self.loss_aggregation == 'adaptive':
            # Weight inversely proportional to loss value (focus more on well-matched ellipses)
            weights = 1.0 / (individual_losses + 1e-8)
            weights = weights / weights.sum()
            final_loss = torch.sum(individual_losses * weights)
            
        else:
            raise ValueError(f"Unsupported aggregation method: {self.loss_aggregation}")
            
        return final_loss

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


# class MultiScaleLoss(nn.Module):
# 	def __init__(self, scales=3, norm='L1'):
# 		super(MultiScaleLoss, self).__init__()
# 		self.scales = scales
# 		if norm == 'L1':
# 			self.loss = nn.L1Loss()
# 		if norm == 'L2':
# 			self.loss = nn.MSELoss()

# 		self.weights = torch.FloatTensor([1/(2**scale) for scale in range(self.scales)])
# 		self.multiscales = [nn.AvgPool2d(2**scale, 2**scale) for scale in range(self.scales)]

# 	def forward(self, output, target):
# 		loss = 0
# 		for i in range(self.scales):
# 			output_i, target_i = self.multiscales[i](output), self.multiscales[i](target)
# 			loss += self.weights[i]*self.loss(output_i, target_i)
			
# 		return loss


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
    
    