import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['svg.fonttype'] = 'none'  # Do not convert text to paths, retain as font
plt.rcParams['font.family'] = 'Arial'  # Set a standard font family

# Load the CSV file
csv_file_path = '/Users/danielsinausia/Documents/Paper rebuttal/Correlations/current/current.csv'
df = pd.read_csv(csv_file_path)

wavenumbers = df.columns[:]  # Assuming first column is labels, so skip it
wavenumbers = wavenumbers.str.replace("Mean", "").astype(float)

x1 = df.iloc[0].values[:]
y1 = df.iloc[1].values[:]
x2 = df.iloc[2].values[:]
y2 = df.iloc[3].values[:]
x3 = df.iloc[4].values[:]
y3 = df.iloc[5].values[:]
x4 = df.iloc[6].values[:]
y4 = df.iloc[7].values[:]

# Create a color map for the actual wavenumbers
norm = plt.Normalize(vmin=wavenumbers.min(), vmax=wavenumbers.max())
colors = plt.cm.cividis(norm(wavenumbers))

fig, axs = plt.subplots(2, 2, figsize=(18, 12))  # 2 rows, 2 columns for each dataset

def plot_dataset(ax, x, y, dataset_label, marker_style):
    sc = ax.scatter(x, y, c=colors, label=dataset_label, marker=marker_style)
    # Add wavenumber labels next to each point
    for i in range(len(x)):
        ax.text(x[i], y[i], f'{wavenumbers[i]:.0f}', fontsize=9, ha='right', va='bottom')
    ax.set_xlabel('Spearman Correlation Coefficient')
    ax.set_ylabel('DTW Alignment Factor')
    ax.set_title(dataset_label)
    return sc

plot_dataset(axs[0, 0], x1, y1, '0.002 M', 'o')
plot_dataset(axs[0, 1], x2, y2, '0.1 M', 'x')
plot_dataset(axs[1, 0], x3, y3, '0.2 M', '^')
plot_dataset(axs[1, 1], x4, y4, '1 M', 's')

cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap='cividis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label('Wavenumber')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for the colorbar

output_dir = os.path.dirname(csv_file_path)
svg_file_path = os.path.join(output_dir, 'scatter_plots_with_wavenumbers.svg')
png_file_path = os.path.join(output_dir, 'scatter_plots_with_wavenumbers.png')

plt.savefig(svg_file_path, format='svg', bbox_inches='tight')  # Save as SVG with font retention
plt.savefig(png_file_path, format='png', dpi=300, bbox_inches='tight')  # Save as PNG

plt.show()

#%%heatmaps
concentrations = ['0.002 M', '0.1 M', '0.2 M', '1 M']
spearman_data = np.array([x1, x2, x3, x4])
dtw_data = np.array([y1, y2, y3, y4])

plt.figure(figsize=(12, 6))
plt.imshow(spearman_data, aspect='auto', cmap='bone', extent=[wavenumbers.min(), wavenumbers.max(), 0, 4])
plt.colorbar(label='Spearman Correlation Coefficient')
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=concentrations)
plt.xlabel('Wavenumber')
plt.title('Heatmap of Spearman Correlation Coefficient')
plt.savefig(os.path.join(output_dir, 'spearman_heatmap.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'spearman_heatmap.png'), format='png', dpi=300, bbox_inches='tight')
plt.show()

# Create heatmap for DTW alignment factors
plt.figure(figsize=(12, 6))
plt.imshow(dtw_data, aspect='auto', cmap='pink', extent=[wavenumbers.min(), wavenumbers.max(), 0, 4])
plt.colorbar(label='DTW Alignment Factor')
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=concentrations)
plt.xlabel('Wavenumber')
plt.title('Heatmap of DTW Alignment Factors')
plt.savefig(os.path.join(output_dir, 'dtw_heatmap.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'dtw_heatmap.png'), format='png', dpi=300, bbox_inches='tight')
plt.show()
