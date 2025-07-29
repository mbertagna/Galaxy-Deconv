import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import re

# Add the parent directory to the path to import utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils_plot import get_label, get_color

def plot_shear_error_vs_snr(results_path='results_snr/', output_file='shear_error_vs_snr.png'):
    """
    Generates a plot of shear error vs. dataset SNR.

    Args:
        results_path (str): The path to the directory containing the result folders.
        output_file (str): The name of the file to save the plot to.
    """
    
    # --- Configuration ---
    
    # SNRs to include in the plot. These should match the SNRs of your datasets.
    snrs = [20, 40, 60, 80, 100]
    
    # Methods to plot. The keys should match the method names used in `test_custom.py`
    # The values are tuples for styling: (linestyle, alpha, linewidth)
    methods_to_plot = {
      'Unrolled_ADMM_Gaussian(2)': ('-', 1, 4.5),
      'FPFS': ('--', 1, 4.5),
    }
    
    # --- Plotting ---
    
    fig, ax1 = plt.subplots(figsize=(14, 9), facecolor='white')
    
    gt_shear_all_snrs = {}

    # Load ground truth shear values
    for snr in snrs:
        gt_method_name = f'No_Deconv_snr{snr}'
        gt_results_file = os.path.join(results_path, gt_method_name, 'results.json')
        
        try:
            with open(gt_results_file, 'r') as f:
                results = json.load(f)
                gt_shear_all_snrs[str(snr)] = np.array(results[str(snr)]['gt_shear'])
        except FileNotFoundError:
            print(f"Warning: Could not find ground truth results file: {gt_results_file}")
            continue
        except KeyError:
            print(f"Warning: 'gt_shear' not found in {gt_results_file} for SNR {snr}")
            continue

    # Plot each method
    for method_base_name, style in methods_to_plot.items():
        g_errs = []
        
        for snr in snrs:
            method_instance_name = f"{method_base_name}_snr{snr}"
            
            # For FPFS, the method name doesn't include the model part
            if "FPFS" in method_base_name:
                method_instance_name = f"FPFS_snr{snr}"

            results_file = os.path.join(results_path, method_instance_name, 'results.json')
            
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Results file not found for method {method_instance_name}, skipping.")
                g_errs.append(np.nan) # Add a placeholder
                continue

            # Get reconstructed shear and calculate error
            rec_shear = np.array(results.get(str(snr), {}).get('rec_shear'))
            gt_shear = gt_shear_all_snrs.get(str(snr))

            if rec_shear.size == 0 or gt_shear is None or rec_shear.shape != gt_shear.shape:
                print(f"Warning: Data missing or mismatched for {method_instance_name}, skipping.")
                g_errs.append(np.nan)
                continue
                
            # The 3rd column (index 2) is the ellipticity
            rec_err = np.abs(rec_shear[:, 2] - gt_shear[:, 2])
            g_errs.append(np.median(rec_err))
            
        # Plotting the data for the current method
        label = get_label(method_base_name)
        color = get_color(method_base_name)
        ax1.plot(snrs, g_errs, style[0], label=label, color=color, linewidth=style[2], alpha=style[1])

    # --- Formatting the plot ---
    
    ax1.set_yscale('log', base=10)
    ax1.set_xlabel('Galaxy Image SNR', fontsize=24)
    ax1.set_ylabel('Median Ellipticity Error\n($\\Delta g=|g_{gt} - g_{rec}|$)', fontsize=24)
    ax1.set_title('Ellipticity Errors of Different Deconvolution Methods', fontsize=24.5)
    
    ax1.set_xticks(snrs)
    ax1.set_xticklabels([str(s) for s in snrs])
    
    ax1.legend(fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    # You can customize the results path if needed
    plot_shear_error_vs_snr(results_path='results_snr/') 