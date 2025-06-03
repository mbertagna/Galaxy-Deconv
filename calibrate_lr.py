import argparse
import logging
import os
import numpy as np

import torch
# from torch.optim import Adam

from models.ResUNet import ResUNet
from models.Tikhonet import Tikhonet
# from models.Unrolled_ADMM import Unrolled_ADMM
from models.unrolled_admm_gaussian import UnrolledADMMGaussian
from utils.utils_data import get_dataloader
from utils.utils_train import MultiScaleLoss, ShapeConstraint, get_model_name, BestEllipseLoss, MomentBasedLoss, ShapeletMomentsLoss, FPFSLoss, FPFSCoeffLoss

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_grad_norm(model):
    """Compute the L2 norm of gradients across all model parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def train(model_name='Unrolled ADMM', n_iters=8, llh='Poisson', PnP=True, remove_SubNet=False, filter='Laplacian',
          loss='MultiScale', flux_norm=False, loss_coeffs=[], 
          data_path='./simulated_datasets/LSST_23.5_deconv/', train_val_split=0.8, batch_size=32,
          model_save_path='./saved_models/', pretrained_epochs=0):
    
    model_name = get_model_name(method=model_name, loss=loss, filter=filter, n_iters=n_iters, llh=llh, PnP=PnP, remove_SubNet=remove_SubNet)

    if flux_norm:
        model_name += '_flux_norm'

    if loss_coeffs:
        model_name += f'_{"_".join(c for c in loss_coeffs)}'

    logger = logging.getLogger('Train')
    logger.info(' Start learning rate calibration %s on %s data.', model_name, data_path)
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    train_loader, val_loader = get_dataloader(data_path=data_path, train=True, train_val_split=train_val_split, batch_size=batch_size, num_workers=0)
    
    if 'ADMM' in model_name:
        model = UnrolledADMMGaussian(n_iters=n_iters)
    elif 'Tikhonet' in model_name:
        model = Tikhonet(filter=filter)
    elif 'ShapeNet' in model_name:
        model = Tikhonet(filter=filter)
    elif model_name == 'ResUNet':
        model = ResUNet()

    model.to(device)
    if pretrained_epochs > 0:
        try:
            pretrained_file = os.path.join(model_save_path, f'{model_name}_{pretrained_epochs}epochs.pth')
            model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
            logger.info(' Successfully loaded in %s.', pretrained_file)
        except:
            raise Exception(' Failed loading in %s!', pretrained_file)

    if 'ShapeNet' in model_name or loss == 'Shape':
        loss_fn = ShapeConstraint(device=device, fov_pixels=48, n_shearlet=2, gamma=1)
    elif loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss == 'MultiScale':
        loss_fn = MultiScaleLoss()
    elif loss == 'BestEllipse':
        step = 0.025
        pps = np.arange(start=0.3, stop=0.7+step, step=step)
        loss_fn = BestEllipseLoss(
            ellipse_levels=pps,
            center_weight=1.0,
            angle_weight=1.0,
            axis_weight=1.0,
        )
    elif loss == 'ShapeletMomentsLoss':
        loss_fn = ShapeletMomentsLoss()
    elif loss == 'FPFSLoss':
        loss_fn = FPFSLoss()
    elif loss == 'FPFSCoeffLoss':
        loss_fn = FPFSCoeffLoss(flux_norm=flux_norm, remove_coeffs=remove_coeffs)
    elif loss == 'L1_FPFSLoss':
        loss_fn = MultiScaleLoss(scales=1, aux_loss_fn=FPFSLoss(), aux_weight=0.1)
    elif loss == 'L1':
        loss_fn = MultiScaleLoss(scales=1)
    
    # Evaluate on train dataset.
    train_loss = 0.0
    train_grad_norm = 0.0
    model.train()
    for _, ((obs, psf, alpha), gt) in enumerate(train_loader):
        obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
        model.zero_grad()
        rec = model(obs, psf, alpha)
        loss = loss_fn(gt, rec)
        loss.backward()
        grad_norm = compute_grad_norm(model)
        train_loss += loss.item()
        train_grad_norm += grad_norm

    # Evaluate on val dataset.
    val_loss = 0.0
    val_grad_norm = 0.0
    model.train()
    for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
        obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
        model.zero_grad()
        rec = model(obs, psf, alpha)
        loss = loss_fn(gt, rec)
        loss.backward()
        grad_norm = compute_grad_norm(model)
        val_loss += loss.item()
        val_grad_norm += grad_norm

    logger.info(" [train_loss={:.4g}  train_grad_norm={:.4g}  val_loss={:.4g}  val_grad_norm={:.4g}]".format(
        train_loss/len(train_loader),
        train_grad_norm/len(train_loader),
        val_loss/len(val_loader),
        val_grad_norm/len(val_loader)))



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--model', type=str, default='Unrolled_ADMM', choices=['Unrolled_ADMM', 'Tikhonet', 'ShapeNet', 'ResUNet'])
    parser.add_argument('--llh', type=str, default='Gaussian', choices=['Gaussian', 'Poisson'])
    parser.add_argument('--remove_SubNet', action="store_true")
    parser.add_argument('--filter', type=str, default='Laplacian', choices=['Identity', 'Laplacian'])
    parser.add_argument('--loss', type=str, default='MultiScale', choices=['MultiScale', 'MSE', 'Shape', 'BestEllipse', 'MomentBasedLoss', 'ShapeletMomentsLoss', 'FPFSLoss', 'L1_FPFSLoss', 'FPFSCoeffLoss', 'L1'])
    parser.add_argument('--train_val_split', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_epochs', type=int, default=0)
    parser.add_argument('--flux_norm', action='store_true', help='Enable flux normalization')
    parser.add_argument('--loss_coeffs', nargs='*', type=str, default=[], help='List of coefficient names to include', choices=["m00", "m20", "m22c", "m22s", "m40", "m42c", "m42s", "m44c", "m44s", "m60", "m64c", "m64s"])
    opt = parser.parse_args()


    train(model_name=opt.model, n_iters=opt.n_iters, llh=opt.llh, PnP=True, remove_SubNet=opt.remove_SubNet, filter=opt.filter,
          loss=opt.loss, flux_norm=opt.flux_norm, loss_coeffs=opt.loss_coeffs, 
          data_path='/Users/michaelbertagna/git/Galaxy-Deconv/simulated_datasets/LSST_23.5_deconv/', train_val_split=opt.train_val_split, batch_size=opt.batch_size,
          model_save_path='./saved_models_shape_loss/', pretrained_epochs=opt.pretrained_epochs)