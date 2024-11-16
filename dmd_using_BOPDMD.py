import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import magma
import os
import math
from pydmd import DMD
from pydmd.bopdmd import BOPDMD
from pydmd.plotter import plot_summary
#import matplotlib.colors as colors

file_path = "..."

start_reciprocal_cm = 1101
end_reciprocal_cm = 3999
start_reciprocal_cm_bkg = 2500
end_reciprocal_cm_bkg = 3997
spectrum_to_plot_as_example = 500

df = pd.read_csv(file_path, header=None, skiprows=1)
df = df.iloc[:, :911]
df = df.iloc[::-1]

wavenumbers = df.iloc[:, 0].values
start_index = np.where(wavenumbers >= start_reciprocal_cm)[0][0]
end_index = np.where(wavenumbers <= end_reciprocal_cm)[0][-1]
start_index_bkg = np.where(wavenumbers >= start_reciprocal_cm_bkg)[0][0]
end_index_bkg = np.where(wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
trimmed_wavenumbers = wavenumbers[start_index:end_index+1]

index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]

corrected_spectra = pd.DataFrame()
for col in df.columns[1:]:
    spectrum = df.iloc[start_index:end_index+1, col].values
    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
    intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
    background_array = slope * trimmed_wavenumbers + intercept
    spectrum_corrected = spectrum - background_array
    corrected_spectra[col] = spectrum_corrected

plt.figure(figsize=(12, 6))
plt.plot(trimmed_wavenumbers, corrected_spectra[spectrum_to_plot_as_example], label='Background Corrected Data')
plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=12)
plt.ylabel('Intensity (a.u.)', fontsize=12)
plt.xlim(start_reciprocal_cm, end_reciprocal_cm)
plt.gca().invert_xaxis()
plt.legend(fontsize=12)
plt.title('Background Corrected Spectrum', fontsize=12)

folder_name = os.path.basename(os.path.dirname(file_path))
folder = os.path.dirname(file_path)
folder_path = os.path.join(folder, folder_name)
os.makedirs(folder_path, exist_ok=True)
filename = os.path.basename(folder) + "_withoutbackground_correction"
png_path = os.path.join(folder_path, f"{filename}.png")
svg_path = os.path.join(folder_path, f"{filename}.svg")
plt.savefig(png_path)
plt.savefig(svg_path, format='svg', transparent=True)
plt.show()

corrected_spectra_path = os.path.join(folder_path, "corrected_spectra.csv")
corrected_spectra.to_csv(corrected_spectra_path, index=False)


    
    

#%% Mean-centering the corrected data

mean_values = corrected_spectra.mean(axis='columns')
centered_data = corrected_spectra.subtract(mean_values, axis=0)
corrected_spectra_centered_path = os.path.join(folder_path, "corrected_spectra_centered.csv")
centered_data.to_csv(corrected_spectra_centered_path, index=True) 

#%%  Following https://www.youtube.com/watch?v=KAau5TBU0Sc&list=WL&index=39&t=657s, https://www.youtube.com/watch?v=sQvrK8AGCAo and https://www.youtube.com/watch?v=kVbEHH_laNU

X = centered_data.iloc[:, :-1].values
X2 = centered_data.iloc[:, 1:].values

time_per_spectrum = 1.1 # Time it takes to record one spectrum
total_time = time_per_spectrum * df.shape[1]
time = np.linspace(0, total_time, df.shape[1]-2)


optdmd = BOPDMD (svd_rank=20, num_trials=0, varpro_opts_dict={'tol': 1e-4, 'maxiter': 2000})

#%%
optdmd.fit(X, time)
#%%
plot_summary(optdmd, index_modes=[4,5,6], order='F')

#%%

extent = [0, total_time, trimmed_wavenumbers[-1], trimmed_wavenumbers[0]] 

plt.figure(figsize=(12, 6))
plt.imshow(optdmd.reconstructed_data.real, aspect='auto', extent=extent)
plt.colorbar(label='Intensity (a.u.)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Wavenumbers (cm$^{-1}$)', fontsize=12)
plt.title('Reconstructed Data', fontsize=12)
plt.gca().invert_yaxis()
plt.show()

#%%  Reconstructing specific modes

selected_modes = [7]

modes = optdmd.modes.real
dynamics = optdmd.dynamics.real
selected_modes_matrix = modes[:, selected_modes]
selected_dynamics = dynamics[selected_modes, :]

reconstructed_data_manual = np.dot(selected_modes_matrix, selected_dynamics)


plt.figure(figsize=(10, 6))
plt.imshow(reconstructed_data_manual, aspect='auto', cmap='magma', extent = extent)
plt.colorbar(label='Intensity')
plt.xlabel('Time Index')
plt.ylabel('Wavenumbers')
plt.title('Manually Reconstructed Data using Specific Modes')
plt.show()


data_to_save = np.column_stack((trimmed_wavenumbers, reconstructed_data_manual))
reconstructed_data_file_path = os.path.join(folder_path, "reconstructed_data_manual.csv")
np.savetxt(reconstructed_data_file_path, data_to_save, delimiter=",", header="Wavenumbers,Reconstructed Data")


#%%

modes = optdmd.modes.real[:, ::-1]
dynamics = optdmd.dynamics.real[::-1, :]

modes_file_path = os.path.join(folder_path, "dmd_modes.csv")
np.savetxt(modes_file_path, modes, delimiter=",")


dynamics_file_path = os.path.join(folder_path, "dmd_dynamics.csv")
np.savetxt(dynamics_file_path, dynamics, delimiter=",")


