import argparse
import json
import logging
import os
import re
import time

import torch
from tqdm import tqdm

from models.Richard_Lucy import Richard_Lucy
from models.Tikhonet import Tikhonet
# from models.Unrolled_ADMM import Unrolled_ADMM
from models.unrolled_admm_gaussian import UnrolledADMMGaussian
from models.Wiener import Wiener
from utils.utils_data import get_dataloader
from utils.utils_test import delta_2D, estimate_shear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_shear(method, n_iters, model_file, n_gal, data_path, result_path):
    logger = logging.getLogger('Shear Test')
    logger.info(' Testing method: %s', method)
        
    psf_delta = delta_2D(48, 48)
    
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results.json')
    
    # Load the model.
    model = None
    if method == 'Wiener':
        model = Wiener()
    elif 'Richard-Lucy' in method:
        model = Richard_Lucy(n_iters=n_iters)
    elif method == 'Tikhonet':
        model = Tikhonet(filter='Identity')
    elif 'ShapeNet' in method or 'Laplacian' in method:
        model = Tikhonet(filter='Laplacian')
    elif 'Gaussian' in method:
        model = UnrolledADMMGaussian(n_iters=n_iters)
    else:
        # model = Unrolled_ADMM(n_iters=n_iters, llh='Poisson', PnP=True)
        pass

    if model is not None:
        model.to(device)
        if 'Tikhonet' in method or 'ShapeNet' in method or 'ADMM' in method:
            try: # Load the pretrained wieghts.
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(' Successfully loaded in %s.', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
        model.eval()
    
    # Extract SNR from data_path
    snr_match = re.search(r'snr(\d+)', data_path)
    if snr_match:
        snr = int(snr_match.group(1))
    else:
        logger.warning('Could not extract SNR from data_path: %s. Using default SNR=100 for logging.', data_path)
        snr = 100 # default

    logger.info(' Running shear test with %s SNR=%s galaxies.\n', n_gal, snr)
    test_loader = get_dataloader(data_path=data_path, train=False,
                                 obs_folder='obs/', gt_folder='gt/', num_workers=20)
    
    rec_shear, gt_shear = [], []
    for ((obs, psf, alpha), gt), idx in zip(test_loader, tqdm(range(n_gal))):
        with torch.no_grad():
            if method == 'No_Deconv':
                gt = gt.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                gt_shear.append(estimate_shear(gt, psf_delta))
                rec_shear.append(estimate_shear(obs, psf_delta))
            elif method == 'FPFS':
                psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(obs, psf))
            elif method == 'Wiener':
                obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                rec = model(obs, psf, alpha) 
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
            elif 'Richard-Lucy' in method:
                obs, psf = obs.to(device), psf.to(device)
                rec = model(obs, psf) 
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
            else: # Unrolled ADMM, Wiener, Tikhonet, ShapeNet
                obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                rec = model(obs, psf, alpha)
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
    
    # Save results.
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(" Successfully loaded in %s.", results_file)
    except:
        results = {} 
        logger.critical(" Failed loading in %s.", results_file)
        
    if str(snr) not in results:
        results[str(snr)] = {}
    results[str(snr)]['rec_shear'] = rec_shear
    if method == 'No_Deconv':
        results[str(snr)]['gt_shear'] = gt_shear
    
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logger.info(" Shear test results saved to %s.\n", results_file)
    


def test_time(method, n_iters, model_file, n_gal, data_path, result_path):  
    """Test the time consumption of different methods."""
    logger = logging.getLogger('Time Test')
    logger.info(' Running time test with %s galaxies.', n_gal)
    logger.info(' Testing method: %s', method)
    
    test_loader = get_dataloader(data_path=data_path, train=False, num_workers=20, obs_folder='obs/', gt_folder='gt/')
    
    psf_delta = delta_2D(48, 48)
    
    result_folder = os.path.join(result_path, method)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    results_file = os.path.join(result_folder, 'results.json')

    # Load the model.
    model = None
    if method == 'Wiener':
        model = Wiener()
    elif 'Richard-Lucy' in method:
        model = Richard_Lucy(n_iters=n_iters)
    elif method == 'Tikhonet':
        model = Tikhonet(filter='Identity')
    elif 'ShapeNet' in method or 'Laplacian' in method:
        model = Tikhonet(filter='Laplacian')
    elif 'Gaussian' in method:
        model = UnrolledADMMGaussian(n_iters=n_iters)
    else:
        # model = Unrolled_ADMM(n_iters=n_iters, llh='Poisson', PnP=True)
        pass

    if model is not None:
        model.to(device)
        if 'Tikhonet' in method or 'ShapeNet' in method or 'ADMM' in method:
            try: # Load the pretrained wieghts.
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logger.info(' Successfully loaded in %s.', model_file)
            except:
                raise Exception('Failed loading in %s', model_file)
        model.eval()

    rec_shear = []
    time_start = time.time()
    for ((obs, psf, alpha), gt), idx in zip(test_loader, tqdm(range(n_gal))):
        with torch.no_grad():
            if method == 'No_Deconv':
                obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(obs, psf_delta))
            elif method == 'FPFS':
                psf = psf.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                obs = obs.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(obs, psf))
            elif method == 'Wiener':
                obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                rec = model(obs, psf, alpha)
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
            elif 'Richard-Lucy' in method:
                obs, psf = obs.to(device), psf.to(device)
                rec = model(obs, psf) 
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
            else: # Unrolled ADMM, Wiener, Tikhonet, ShapeNet
                obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                rec = model(obs, psf, alpha)
                rec = rec.cpu().squeeze(dim=0).squeeze(dim=0).detach().numpy()
                rec_shear.append(estimate_shear(rec, psf_delta))
                
    time_end = time.time()
    logger.info(' Tested %s on %s galaxies: Time = {:.4g}s.'.format(time_end-time_start),method, n_gal)

    # Save test results.
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logger.info(" Successfully loaded in %s.", results_file)
    except:
        results = {} 
        logger.critical(" Failed loading in %s.", results_file)
    results['time'] = (time_end-time_start, n_gal)
    
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logger.info(" Time test results saved to %s.\n", results_file)
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for shear test and time test.')
    parser.add_argument('--test', type=str, default='shear', choices=['shear', 'time'])
    parser.add_argument('--n_gal', type=int, default=10000)
    parser.add_argument('--result_path', type=str, default='results_200/')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    # Uncomment the methods to be tested.
    methods = {
        'Unrolled_ADMM_Gaussian(2)_snr20': (
            2, 
            "./saved_models_shape_loss/G_PnP_ADMM_2it_FPFSCoeffLoss_m20_m22c_m22s_m40_m42c_m42s_m44c_m44s_snr20_20epochs.pth",
            "./simulated_datasets/LSST_23.5_deconv_snr020/"
        ),
        'Unrolled_ADMM_Gaussian(2)_snr40': (
            2,
            "./saved_models_shape_loss/G_PnP_ADMM_2it_FPFSCoeffLoss_m20_m22c_m22s_m40_m42c_m42s_m44c_m44s_snr40_20epochs.pth", 
            "./simulated_datasets/LSST_23.5_deconv_snr040/"
        ),
        'Unrolled_ADMM_Gaussian(2)_snr60': (
            2,
            "./saved_models_shape_loss/G_PnP_ADMM_2it_FPFSCoeffLoss_m20_m22c_m22s_m40_m42c_m42s_m44c_m44s_snr60_20epochs.pth",
            "./simulated_datasets/LSST_23.5_deconv_snr060/"
        ),
        'Unrolled_ADMM_Gaussian(2)_snr80': (
            2,
            "./saved_models_shape_loss/G_PnP_ADMM_2it_FPFSCoeffLoss_m20_m22c_m22s_m40_m42c_m42s_m44c_m44s_snr80_20epochs.pth",
            "./simulated_datasets/LSST_23.5_deconv_snr080/"
        ),
        'Unrolled_ADMM_Gaussian(2)_snr100': (
            2,
            "./saved_models_shape_loss/G_PnP_ADMM_2it_FPFSCoeffLoss_m20_m22c_m22s_m40_m42c_m42s_m44c_m44s_snr100_20epochs.pth",
            "./simulated_datasets/LSST_23.5_deconv_snr100/"
        ),
    }
    

    if opt.test == 'shear':
        for method, (n_iters, model_file, data_path) in methods.items():
            test_shear(method=method, n_iters=n_iters, model_file=model_file, n_gal=opt.n_gal,
                       data_path=data_path, result_path=opt.result_path)
    elif opt.test == 'time':
        for method, (n_iters, model_file, data_path) in methods.items():
            for i in range(3): # Run 2 dummy test first to warm up the GPU.
                test_time(method=method, n_iters=n_iters, model_file=model_file, n_gal=opt.n_gal,
                          data_path=data_path, result_path=opt.result_path)
    else:
        raise ValueError("Invalid test type.")