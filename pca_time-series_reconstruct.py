
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import magma
#import matplotlib.colors as colors
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths



file_path = "..."
output_folder_path = "..."
experiment_classification = '_08'

df = pd.read_csv(file_path, header=None, skiprows=0)

df = df.T  # Transposing to have the variables as the first column
df.columns = df.iloc[0]  # Making the wavelengths the header
df = df[1:911]  # Using only the first 910 spectra

mean_values = df.mean()

centered_data = df - mean_values
color_map  = {
    'PC 1': '#f44336',
    'PC 2': '#e81e63', 
    'PC 3': '#9c27b0', 
    'PC 4': '#673ab7',
    'PC 5': '#3f51b5', 
    'PC 6': '#2196f3', 
    'PC 7': '#03a9f4', 
    'PC 8': '#00bcd4', 
    'PC 9': '#009688', 
    'PC 10': '#4caf50', 
    'PC 11': '#8bc34a', 
    'PC 12': '#cddc39', 
    'PC 13': '#ffeb3b', 
    'PC 14': '#ffc107', 
    'PC 15': '#ff9800', 
    'PC 16': '#ff5722', 
    'PC 17': '#795548', 
    'PC 18': '#9e9e9e', 
    'PC 19': '#607d8b', 
    'PC 20': '#1F3B4D',
    'PC 21': '#1F3B4D'
    }



#%%

time_per_spectrum = 1.1 # Time it takes to record one spectrum
total_time = time_per_spectrum * df.shape[0]  # Total time for all spectra
time = np.linspace(0, total_time, df.shape[0])


covariance_matrix = centered_data.cov()

#%% alternatively, from sklearn import decomposition or SVD

eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)


#%%

sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort indices in descending order
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Convert eigenvectors to a DataFrame
eigenvectors_df = pd.DataFrame(sorted_eigenvectors, columns=df.columns)

output_file_path = output_folder_path + "eigenvectors.csv"
eigenvectors_df.to_csv(output_file_path, index=False)

#%% CVE


pcs_to_plot_in_cve = 20 # PCs number must be equal or smaller than the number of rows or columns in the cov matrix


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(1, pcs_to_plot_in_cve + 1), np.cumsum(sorted_eigenvalues)[:pcs_to_plot_in_cve]/np.sum(sorted_eigenvalues)*100, '-o')
plt.xlabel('PC', fontsize=18)
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], fontsize=16)
plt.ylabel('Cumulative Variance Explained (%)', fontsize=18)
plt.title('Cumulative variance explained', fontsize=18)
ax.yaxis.set_tick_params(labelsize=16)

# Inset plot
left, bottom, width, height = [0.4, 0.25, 0.45, 0.45]
inset_ax = fig.add_axes([left, bottom, width, height])

inset_ax.plot(range(4, pcs_to_plot_in_cve + 1), np.cumsum(sorted_eigenvalues)[3:pcs_to_plot_in_cve]/np.sum(sorted_eigenvalues)*100, '-o')
inset_ax.set_xlabel('PC', fontsize=14)
inset_ax.set_ylabel('CVE (%)', fontsize=14)
#inset_ax.set_title('Zoomed-in View', fontsize=14)
inset_ax.set_xticks([4, 8, 12, 16, 20])
inset_ax.xaxis.set_tick_params(labelsize=12)
inset_ax.yaxis.set_tick_params(labelsize=12)

output_plot_path = output_folder_path + "cve"
plt.savefig(output_plot_path + ".png", dpi=300, bbox_inches="tight")
#plt.savefig(output_plot_path + ".eps", format='eps', dpi=300, bbox_inches="tight")
plt.savefig(output_plot_path + ".svg", format='svg', bbox_inches="tight")


plt.show()

#%% Scores, Eigenspectra and Rconstructed data


############# Before adding back the mean value ######################

def pcs_to_indices(pcs):
    return [pc - 1 for pc in pcs]

selected_pcs = [3]  # PCs to use
pc_colors = {f'PC {pc}': color_map[f'PC {pc}'] for pc in selected_pcs if f'PC {pc}' in color_map}


selected_components = pcs_to_indices(selected_pcs) # selected principal component numbers to zero-based indices
selected_evectors = sorted_eigenvectors[:, selected_components]
scores = centered_data.values @ selected_evectors @ np.diag(1/(np.sqrt(sorted_eigenvalues[selected_components]))) # it is necessary to use .values because centered_data is a dataframe and @ doesn´t work with them, for those I would need to use np.dot()
reconstructed_data = scores @ np.diag(sorted_eigenvalues[selected_components]) @ selected_evectors.T

#reconstructed_data += mean_values.values.reshape(1,-1)
reconstructed_data[0, :] = df.columns.astype(float)


reconstructed_df_before_mean_value = pd.DataFrame(reconstructed_data)
pcs_string = '_'.join(map(str, selected_pcs)) # This is to add the PCs used in the name of the csv
output_file_path = output_folder_path + f"reconstructed_data_before_mean_PCs{pcs_string}.csv"  # Set the desired file path
reconstructed_df_before_mean_value.to_csv(output_file_path, index=False)

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(reconstructed_data[1:, :], cmap='magma', aspect='auto',
                extent=[reconstructed_data[0, 0], reconstructed_data[0, -1], 
                        time[-1], time[0]],  # Flip the Y-axis here
                )
plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=18)
plt.xticks([3500, 3000, 2500, 2000, 1500, 1000], fontsize=16)
plt.yticks([100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize=16)
plt.ylabel('Time (s)', fontsize=18)
plt.title(f'Reconstructed Spectra using PC(s) {selected_pcs}', fontsize=18)
ax.yaxis.set_tick_params(labelsize=16)  # Adjust the font size as needed
ax.invert_yaxis()


output_plot_path = output_folder_path + f"reconstructed_2D_spectra_using_PCs{selected_pcs}_before_mean_value"
plt.savefig(output_plot_path + ".png", dpi=300, bbox_inches="tight")
#plt.savefig(output_plot_path + ".eps", format='eps', dpi=300, bbox_inches="tight")
plt.savefig(output_plot_path + ".svg", format='svg', bbox_inches="tight")

plt.show()


experiment_numbers = reconstructed_df_before_mean_value.iloc[1:]
experiment_time = np.arange(len(experiment_numbers)) * 1.10

x = reconstructed_data[0,:] 
time = experiment_time
#depth = np.arange(df.shape[0])
X, Y = np.meshgrid(x, time)
Z = reconstructed_data[1:,:]  # Use the entire dataframe for Z


# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')
#cset = ax.contourf(X, Y, Z, zdir='z', offset=-300, cmap='viridis', levels=100)  # Adjust levels
ax.invert_xaxis()
#ax.set_zlim(-300, 500)
ax.set_xlabel('Wavenumbers (cm$^{-1}$)', fontsize=14)
ax.set_ylabel('Time (s)', fontsize=14)
#ax.set_zlabel('Transmittance (a.u.)', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10 
ax.zaxis.labelpad = 10
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) # Eliminates color of bkg
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) # Eliminates color of bkg
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) # Eliminates color of bkg

ax.set_zticklabels([]) # Eliminates text in vertical axis
ax.set_xticks([4000, 3500, 3000, 2500, 2000, 1500, 1000, 500]) 
ax.set_yticks([0, 200, 400, 600, 800, 1000]) 

#ax.set_zticks([]) # Eliminates the Z ticks
ax.set_box_aspect([1.5, 1.2, 1])  # Stretch wavenumber axis (change 2 to the factor you desire)

ax.grid(True)

ax.view_init(elev=15, azim=-80, roll=0) 

output_plot_path = output_folder_path + f"reconstructed_3D_spectra_using_PCs{selected_pcs}_before_mean_value"
plt.savefig(output_plot_path + ".png", dpi=300, bbox_inches="tight")
#plt.savefig(output_plot_path + ".eps", format='eps', dpi=300, bbox_inches="tight")
plt.savefig(output_plot_path + ".svg", format='svg', bbox_inches="tight")

plt.show()




# Plot scores for the selected components
time_scores = np.linspace(0, total_time, scores.shape[0])

fig, ax = plt.subplots(figsize=(10, 8))
for i, pc in enumerate(selected_pcs):
    color = pc_colors[f'PC {pc}']
    plt.plot(time_scores, scores[:, i], label=f'PC {selected_components[i] + 1}', color=color)
plt.title(f'Scoreplots for PC(s) {selected_pcs}', fontsize=18)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Scores (a.u.)', fontsize=18)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
plt.legend(fontsize=16)
# Save heatmap plot in PNG, EPS, and SVG formats
output_plot_path = output_folder_path + f"scoreplots_using_PCs{selected_pcs}"
plt.savefig(output_plot_path + ".png", dpi=300, bbox_inches="tight")
#plt.savefig(output_plot_path + ".eps", format='eps', dpi=300, bbox_inches="tight")
plt.savefig(output_plot_path + ".svg", format='svg', bbox_inches="tight")


plt.show()

# Plot loadings (eigenvectors)
fig, ax = plt.subplots(figsize=(10, 8))
for i, pc in enumerate(selected_pcs):
    color = pc_colors[f'PC {pc}']
for i in range(len(selected_components)):
    plt.plot(df.columns, selected_evectors[:, i], label=f'PC {selected_components[i] + 1}', color=color)
plt.title(f'Eigenspectra for PC(s) {selected_pcs}', fontsize=18)
plt.xlabel('Wavenumbers ($cm^{-1}$)', fontsize=18)
plt.ylabel('Eigenspectra (a.u.)', fontsize=18)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
plt.legend(fontsize=16)
plt.gca().invert_xaxis()


output_plot_path = output_folder_path + f"eigenspectra_PCs{selected_pcs}"
plt.savefig(output_plot_path + ".png", dpi=300, bbox_inches="tight")
#plt.savefig(output_plot_path + ".eps", format='eps', dpi=300, bbox_inches="tight")
plt.savefig(output_plot_path + ".svg", format='svg', bbox_inches="tight")

plt.show()

#%% Residuals


residuals = centered_data.values[1:, :] - reconstructed_data[1:, :]
residuals_df = pd.DataFrame(residuals)

output_file_path = output_folder_path + "residuals_data_mean_centered.csv"
residuals_df.to_csv(output_file_path, index=False)

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(residuals, cmap='magma', aspect='auto',
                extent=[reconstructed_data[0, 0], reconstructed_data[0, -1], 
                        time[-1], time[0]],
                )
plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=18)
plt.xticks([3500, 3000, 2500, 2000, 1500, 1000], fontsize=16)
plt.yticks([100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize=16)
plt.ylabel('Time (s)', fontsize=18)
plt.title(f'Residuals using PC(s) {selected_pcs}', fontsize=18)
ax.yaxis.set_tick_params(labelsize=16)
ax.invert_yaxis()

for y_tick in ax.get_yticks():
    plt.hlines(y_tick, reconstructed_data[0, 0], reconstructed_data[0, -1], colors='black', alpha=0.75, linestyles='dashed', linewidth=0.5)

output_plot_path = output_folder_path + "residuals_mean_centered_data"
plt.savefig(output_plot_path + ".png", dpi=300, bbox_inches="tight")
plt.savefig(output_plot_path + ".svg", format='svg', bbox_inches="tight")

plt.show()
