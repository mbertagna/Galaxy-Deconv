import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from utils.fit_ellipse import transform_tensor_batched, safe_ellipse_params_batched, ellipse_fit_metric, compute_moments, compute_shapelet_moments
from utils.utils_fourier import get_bfunc, project_img_onto_bfunc, get_shape_params

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
        cx_out, cy_out, theta_out, a_out, b_out = output_params.unbind(-1)
        cx_tgt, cy_tgt, theta_tgt, a_tgt, b_tgt = target_params.unbind(-1)
        
        center_coords_out = torch.stack([cx_out, cy_out], dim=-1)
        center_coords_tgt = torch.stack([cx_tgt, cy_tgt], dim=-1)
        
        out_max_axis = torch.max(torch.stack([a_out, b_out], dim=-1), dim=-1)[0]
        tgt_max_axis = torch.max(torch.stack([a_tgt, b_tgt], dim=-1), dim=-1)[0]
        coord_scale = torch.maximum(out_max_axis, tgt_max_axis).unsqueeze(-1) + 1e-8
        
        normalized_center_loss = F.mse_loss(
            center_coords_out / coord_scale.unsqueeze(-1),
            center_coords_tgt / coord_scale.unsqueeze(-1),
            reduction='none'
        ).mean(dim=-1) 
        
        angle_vec_out = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=-1)
        angle_vec_tgt = torch.stack([torch.cos(theta_tgt), torch.sin(theta_tgt)], dim=-1)
        
        cosine_sim = torch.sum(angle_vec_out * angle_vec_tgt, dim=-1)
        normalized_angle_loss = 1 - cosine_sim
        
        axis_scale = torch.maximum(out_max_axis, tgt_max_axis).unsqueeze(-1) + 1e-8
        normalized_a_loss = ((a_out / axis_scale) - (a_tgt / axis_scale))**2
        normalized_b_loss = ((b_out / axis_scale) - (b_tgt / axis_scale))**2
        normalized_axis_loss = 0.5 * (normalized_a_loss + normalized_b_loss)
        
        total_loss = (
            self.center_weight * normalized_center_loss +
            self.angle_weight * normalized_angle_loss +
            self.axis_weight * normalized_axis_loss
        )
        
        return total_loss 
    
    def forward(self, output, target):
        batch_size = output.shape[0]
        device = output.device
        
        output_transformed = transform_tensor_batched(output)
        target_transformed = transform_tensor_batched(target)
        
        gt_params_all_levels = []
        gt_fit_metrics = torch.zeros((batch_size, self.num_ellipses), device=device)
        
        for i, pp in enumerate(self.ellipse_levels):
            gt_params, _ = safe_ellipse_params_batched(target_transformed, peak_pos=pp)
            gt_params_all_levels.append(gt_params)
            
            fit_metric = ellipse_fit_metric(target_transformed, gt_params)
            gt_fit_metrics[:, i] = fit_metric
        
        best_ellipse_indices = torch.argmax(gt_fit_metrics, dim=1)
        
        best_gt_params = torch.zeros((batch_size, 5), device=device)
        
        for b in range(batch_size):
            best_idx = best_ellipse_indices[b].item()
            best_gt_params[b] = gt_params_all_levels[best_idx][b]
        
        output_params = torch.zeros((batch_size, 5), device=device)
        
        for b in range(batch_size):
            best_idx = best_ellipse_indices[b].item()
            pp = self.ellipse_levels[best_idx]
            single_output = output_transformed[b:b+1]
            params, _ = safe_ellipse_params_batched(single_output, peak_pos=pp)
            output_params[b] = params[0]  # Extract from batch dimension
        
        losses = self.ellipse_loss_symmetric(output_params, best_gt_params)
        
        return losses.mean()

class MomentBasedLoss(nn.Module):
    def __init__(self, central_moments_weight=1.0, centroid_weight=1.0, third_order_weight=1.0):
        super(MomentBasedLoss, self).__init__()
        self.central_moments_weight = central_moments_weight
        self.centroid_weight = centroid_weight
        self.third_order_weight = third_order_weight
    
    def forward(self, output, target):
        output_batch_moments = compute_moments(output)
        target_batch_moments = compute_moments(target)
        
        device = output.device
        B = len(output_batch_moments)
        
        output_centroids = torch.zeros((B, 2), device=device)
        target_centroids = torch.zeros((B, 2), device=device)
        output_central_moments = torch.zeros((B, 3), device=device)
        target_central_moments = torch.zeros((B, 3), device=device)
        output_third_order = torch.zeros((B, 4), device=device)
        target_third_order = torch.zeros((B, 4), device=device)
        
        for i in range(B):
            output_moments = output_batch_moments[i]
            target_moments = target_batch_moments[i]
            
            output_centroids[i, 0] = output_moments['cy']
            output_centroids[i, 1] = output_moments['cx']
            target_centroids[i, 0] = target_moments['cy']
            target_centroids[i, 1] = target_moments['cx']
            
            output_central_moments[i, 0] = output_moments['mu20']
            output_central_moments[i, 1] = output_moments['mu11']
            output_central_moments[i, 2] = output_moments['mu02']
            target_central_moments[i, 0] = target_moments['mu20']
            target_central_moments[i, 1] = target_moments['mu11']
            target_central_moments[i, 2] = target_moments['mu02']
            
            output_third_order[i, 0] = output_moments['mu30']
            output_third_order[i, 1] = output_moments['mu21']
            output_third_order[i, 2] = output_moments['mu12']
            output_third_order[i, 3] = output_moments['mu03']
            target_third_order[i, 0] = target_moments['mu30']
            target_third_order[i, 1] = target_moments['mu21']
            target_third_order[i, 2] = target_moments['mu12']
            target_third_order[i, 3] = target_moments['mu03']
        
        centroid_loss = F.mse_loss(output_centroids, target_centroids)
        
        central_moments_loss = F.mse_loss(output_central_moments, target_central_moments)
        
        third_order_loss = F.mse_loss(output_third_order, target_third_order)
        
        total_loss = (
            self.centroid_weight * centroid_loss + 
            self.central_moments_weight * central_moments_loss +
            self.third_order_weight * third_order_loss
        )
        
        return total_loss
    
class ShapeletMomentsLoss(nn.Module):
    def __init__(self, s2_weight=1.0, s4_weight=1.0, combined_weight=1.0):
        """
        Initializes the Shapelet Moments Loss function.
        
        Args:
            s2_weight: Weight for the 2nd order shapelet moment loss
            s4_weight: Weight for the 4th order shapelet moment loss
            combined_weight: Overall weight for the shapelet loss term
        """
        super(ShapeletMomentsLoss, self).__init__()
        self.s2_weight = s2_weight
        self.s4_weight = s4_weight
        self.combined_weight = combined_weight
    
    def forward(self, output, target):
        """
        Computes the loss between output and target based on shapelet moments.
        
        Args:
            output: Model output tensor of shape [B, C, H, W]
            target: Ground truth tensor of shape [B, C, H, W]
            
        Returns:
            total_loss: The weighted shapelet moments loss
        """
        if torch.isnan(output).any():
            print("NaN detected in output tensor in ShapeletMomentsLoss")
            for b in range(output.shape[0]):
                if torch.isnan(output[b]).any():
                    print(f"  NaN found in output batch element {b}")
        
        if torch.isnan(target).any():
            print("NaN detected in target tensor in ShapeletMomentsLoss")
            for b in range(target.shape[0]):
                if torch.isnan(target[b]).any():
                    print(f"  NaN found in target batch element {b}")
        
        output_batch_moments = compute_shapelet_moments(output)
        target_batch_moments = compute_shapelet_moments(target)
        
        B = len(output_batch_moments)
        
        s2_losses = []
        s4_losses = []
        
        for i in range(B):
            output_moments = output_batch_moments[i]
            target_moments = target_batch_moments[i]
            
            if 'S2' not in output_moments or 'S2' not in target_moments:
                print(f"Missing S2 moment for batch element {i}")
                continue
                
            if 'S4' not in output_moments or 'S4' not in target_moments:
                print(f"Missing S4 moment for batch element {i}")
                continue
            
            s2_loss_i = (output_moments['S2'] - target_moments['S2']) ** 2
            s4_loss_i = (output_moments['S4'] - target_moments['S4']) ** 2
            
            if torch.isnan(s2_loss_i):
                print(f"NaN detected in S2 loss for batch element {i}")
                print(f"  output S2: {output_moments['S2'].item() if not torch.isnan(output_moments['S2']) else 'NaN'}")
                print(f"  target S2: {target_moments['S2'].item() if not torch.isnan(target_moments['S2']) else 'NaN'}")
                continue
                
            if torch.isnan(s4_loss_i):
                print(f"NaN detected in S4 loss for batch element {i}")
                print(f"  output S4: {output_moments['S4'].item() if not torch.isnan(output_moments['S4']) else 'NaN'}")
                print(f"  target S4: {target_moments['S4'].item() if not torch.isnan(target_moments['S4']) else 'NaN'}")
                continue
            
            s2_losses.append(s2_loss_i)
            s4_losses.append(s4_loss_i)
        
        if s2_losses:
            s2_loss = torch.stack(s2_losses).mean()
            s4_loss = torch.stack(s4_losses).mean()
            
            if torch.isnan(s2_loss):
                print("NaN detected in aggregated S2 loss")
                s2_loss = torch.tensor(0.0, device=output.device, requires_grad=True)
                
            if torch.isnan(s4_loss):
                print("NaN detected in aggregated S4 loss")
                s4_loss = torch.tensor(0.0, device=output.device, requires_grad=True)
            
            shapelet_loss = self.s2_weight * s2_loss + self.s4_weight * s4_loss
            
            if torch.isnan(shapelet_loss):
                print("NaN detected in final shapelet loss")
                print(f"  s2_loss: {s2_loss.item()}, s4_loss: {s4_loss.item()}")
                print(f"  weights: s2={self.s2_weight}, s4={self.s4_weight}, combined={self.combined_weight}")
                shapelet_loss = torch.tensor(0.0, device=output.device, requires_grad=True)
            
            total_loss = self.combined_weight * shapelet_loss
            
            if torch.isnan(total_loss):
                print("NaN detected in weighted total loss")
                total_loss = torch.tensor(0.0, device=output.device, requires_grad=True)
            
            return total_loss
        else:
            print("Warning: No valid moments found in batch, returning zero loss")
            return torch.tensor(0.0, device=output.device, requires_grad=True)

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
            
            # Add aux loss only at full scale
            if not i: 
                aux_loss = self.aux_loss_fn(output_i, target_i) if self.aux_loss_fn else 0
                loss += self.aux_weight * aux_loss
            
            loss += self.weights[i] * primary_loss
        
        return loss

class FPFSLoss(nn.Module):
    def __init__(self, norm='L1'):
        super(FPFSLoss, self).__init__()
        self.bfunc, self.bfunc_key = get_bfunc(npix=48, pixel_scale=0.2, sigma_arcsec=0.52)

        if norm == 'L1':
            self.loss = nn.L1Loss()
        elif norm == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported norm type. Use 'L1' or 'L2'.")
        
    def forward(self, output, target):
        loss = 0.0

        target_coeffs = project_img_onto_bfunc(target, self.bfunc)
        output_coeffs = project_img_onto_bfunc(output, self.bfunc)

        target_shape_params = get_shape_params(target_coeffs, self.bfunc_key)
        output_shape_params = get_shape_params(output_coeffs, self.bfunc_key)

        loss += self.loss(target_shape_params["total_flux"], output_shape_params["total_flux"])
        loss += self.loss(target_shape_params["e1"], output_shape_params["e1"])
        loss += self.loss(target_shape_params["e2"], output_shape_params["e2"])
        
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
    
    