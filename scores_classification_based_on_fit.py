import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths

def exponential_fit(x, a, b, c):
    return a * np.exp(b * x) + c

def linear_fit(x, m, c):
    return m * x + c

def fit_curves(file_path, row_range=(91, 180), PC_indices=range(1, 16), output_folder=None):
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    if row_range[1] > len(data): 
        print(f"Invalid range for file: {file_path}")
        return []

    results = []
    n_PCs = len(PC_indices)
    n_rows = int(np.ceil(n_PCs / 4))  

    fig_fits, axes_fits = plt.subplots(n_rows, 4, figsize=(16, 5 * n_rows))
    fig_residuals, axes_residuals = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

    axes_fits = axes_fits.flatten() 
    axes_residuals = axes_residuals.flatten()

    for i, PC_index in enumerate(PC_indices): 
        if PC_index >= len(data.columns): 
            print(f"PC index {PC_index} out of range for file: {file_path}")
            continue 

        y_data = data.iloc[row_range[0]:row_range[1], PC_index].to_numpy() 
        x_data = np.arange(len(y_data))

        exp_params, _ = curve_fit(exponential_fit, x_data, y_data, p0=(1, -0.01, np.mean(y_data)), maxfev=10000) 
        y_exp_fit = exponential_fit(x_data, *exp_params)
        lin_params, _ = curve_fit(linear_fit, x_data, y_data)
        y_lin_fit = linear_fit(x_data, *lin_params)

        exp_eq = f"y = {exp_params[0]:.2f} * exp({exp_params[1]:.6f} * x) + {exp_params[2]:.2f}"
        lin_eq = f"y = {lin_params[0]:.2f} * x + {lin_params[1]:.2f}"

        # Calculate residuals
        residuals_exp = y_data - y_exp_fit
        residuals_lin = y_data - y_lin_fit

        # Calculate SSR
        ssr_exp = np.sum(residuals_exp**2)
        ssr_lin = np.sum(residuals_lin**2)

        # Determine best fit
        best_fit = "Exponential" if ssr_exp < ssr_lin else "Linear"

        results.append({
            'file': file_path,
            'PC': PC_index,
            'exponential_equation': exp_eq,
            'linear_equation': lin_eq,
            'ssr_exponential': ssr_exp,
            'ssr_linear': ssr_lin,
            'best_fit': best_fit
        })

        axes_fits[i].scatter(x_data, y_data, label='Data', color='black', alpha=0.7)
        axes_fits[i].plot(x_data, y_exp_fit, label=f'Exp Fit', color='red')
        axes_fits[i].plot(x_data, y_lin_fit, label=f'Lin Fit', color='blue')
        axes_fits[i].set_title(f'PC{PC_index}')
        axes_fits[i].set_xlabel('Index')
        axes_fits[i].set_ylabel('Value')
        axes_fits[i].legend()

        axes_residuals[i].scatter(x_data, residuals_exp, label='Exp Residuals', color='red', alpha=0.7)
        axes_residuals[i].scatter(x_data, residuals_lin, label='Lin Residuals', color='blue', alpha=0.7)
        axes_residuals[i].axhline(0, color='black', linestyle='--')
        axes_residuals[i].set_title(f'PC{PC_index}')
        axes_residuals[i].set_xlabel('Index')
        axes_residuals[i].set_ylabel('Residuals')
        axes_residuals[i].legend()

    # Hide unused subplots
    for i in range(len(PC_indices), len(axes_fits)):
        axes_fits[i].axis('off')
        axes_residuals[i].axis('off')

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(file_path), 'Fit Results')
    os.makedirs(output_folder, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    grouped_fits_path = os.path.join(output_folder, f'{file_name}_grouped_fits.png')
    grouped_residuals_path = os.path.join(output_folder, f'{file_name}_grouped_residuals.png')

    grouped_fits_svg_path = grouped_fits_path.replace(".png", ".svg")
    grouped_residuals_svg_path = grouped_residuals_path.replace(".png", ".svg")

    fig_fits.tight_layout()
    fig_residuals.tight_layout()

    fig_fits.savefig(grouped_fits_path)
    fig_fits.savefig(grouped_fits_svg_path, format='svg')

    fig_residuals.savefig(grouped_residuals_path)
    fig_residuals.savefig(grouped_residuals_svg_path, format='svg')

    plt.close(fig_fits)
    plt.close(fig_residuals)
    
    
    return results

if __name__ == "__main__":
    base_dir = "...
    include_folders = {"DS_00132", "DS_00133"}
    output_folder = os.path.join(base_dir, "PC classification based on curve fits")
    os.makedirs(output_folder, exist_ok=True)

    all_results = []

    for folder_name in include_folders:
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name == "PCA_scores.txt":
                    file_path = os.path.join(root, file_name)
                    print(f"Processing file: {file_path}")

                    try:
                        fit_results = fit_curves(file_path, PC_indices=range(1, 17), output_folder=output_folder)
                        all_results.extend(fit_results)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = os.path.join(output_folder, 'fitting_results.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"Results saved to: {results_csv}")
    else:
        print("No results to save.")
