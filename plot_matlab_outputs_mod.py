'''
to be used following matlab outputs
'''



import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths


def create_variance_plot(df, title, subsubfolder_path, num_colors):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    df.iloc[:num_colors+1].plot(y="% Variance", legend=False, ax=ax, marker='o', linestyle='-', markersize=4)
    ax.set_xlabel("PCs")
    ax.set_ylabel("% Variance")
    plt.xticks(range(0, num_colors + 5, 2))
    plt.xlim(0, num_colors)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    # Add an inset
    inset_ax = inset_axes(ax, width="70%", height="60%", loc='center right', borderpad=0.5)
    inset_ax.plot(df.iloc[2:num_colors+1].index, df.iloc[2:num_colors+1]["% Variance"], marker='o', linestyle='-', markersize=4)
    inset_ax.set_xlim(2, num_colors)
    save_plots(title, subsubfolder_path)


def create_score_plot(df, title, subsubfolder_path, num_colors, start_index, experiment_classification):
    range_start, range_end = 1, num_colors + 1
    columns_to_plot = df.columns[range_start:range_end]
    legend_labels = [f"PC {i}" for i in range(start_index, start_index + len(columns_to_plot))]

    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    y_axis_separation = 0
    text_labels = []

    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]

    for i, (column, label) in enumerate(zip(columns_to_plot, legend_labels)):
        plt.plot(df.index * 1.1, df[column] - (i * y_axis_separation), color=color_map[label], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Score (a.u.)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xlim(0, 1000)

    intersections = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)

    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2] if text_labels else None

        if text_label:
            text_x = (x_start + x_end) / 2
            plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16, color='black', alpha=0.15)

    save_plots(title, subsubfolder_path)


def create_stacked_score_plot(df, title, subsubfolder_path, num_colors, start_index, experiment_classification):
    range_start, range_end = 1, num_colors + 1
    columns_to_plot = df.columns[range_start:range_end]
    legend_labels = [f"PC {i}" for i in range(start_index, start_index + len(columns_to_plot))]

    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    y_axis_separation = 10
    text_labels = []

    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]

    for i, (column, label) in enumerate(zip(columns_to_plot, legend_labels)):
        plt.plot(df.index * 1.1, df[column] - (i * y_axis_separation), color=color_map[label], label=label)
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.ylabel("Score (a.u.)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xlim(0, 1000)

    intersections = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)

    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2] if text_labels else None

        if text_label:
            text_x = (x_start + x_end) / 2
            plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16, color='black', alpha=0.15)

    save_plots(title, subsubfolder_path)


def create_eigenspectra_plot(df, title, subsubfolder_path, num_colors, start_index):
    range_start, range_end = 1, num_colors + 1
    columns_to_plot = df.columns[range_start:range_end]
    legend_labels = [f"PC {i}" for i in range(start_index, start_index + len(columns_to_plot))]

    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    for i, (column, label) in enumerate(zip(columns_to_plot, legend_labels)):
        plt.plot(df.iloc[:, 0], df[column] - (i * 0.1), color=color_map[label], label=label)
    plt.yticks([])
    plt.xlabel("Wavenumbers (cm$^{-1}$)")
    plt.ylabel("Loading (a.u.)")
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.xlim(4000, 1100)
    save_plots(title, subsubfolder_path)


def save_plots(title, subsubfolder_path):
    figure_path = os.path.join(subsubfolder_path, title + ".png")
    svg_path = os.path.join(subsubfolder_path, title + ".svg")

    plt.savefig(figure_path, dpi=300)
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.close()


def process_pca(folder_path, num_colors):
    base_plot_directory = folder_path
    pca_plots_directory = os.path.join(base_plot_directory, "PCA plots")

    if not os.path.exists(pca_plots_directory):
        os.mkdir(pca_plots_directory)

    files_in_folder = os.listdir(folder_path)
    csv_files = [file for file in files_in_folder if file.startswith("DS_") and file.endswith(".csv")]

    if csv_files:
        csv_file = csv_files[0]
        experiment_classification = os.path.splitext(csv_file)[0][-3:]
        print("CSV File:", csv_file)
        print("Experiment Classification:", experiment_classification)
    else:
        print("No CSV file starting with 'DS_' found in the specified folder.")
        return

    cmap = viridis
    colors = cmap(np.linspace(0, 1, num_colors))
    global color_map
    color_map = {f'PC {i}': colors[i - 1] for i in range(1, num_colors + 1)}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "PCA_CVE.txt":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, header=None, names=["% Variance"])
                zero_row = pd.DataFrame([[0]], columns=["% Variance"])
                df = pd.concat([zero_row, df], ignore_index=True)
                title = os.path.splitext(file)[0]
                subsubfolder = os.path.basename(root)
                subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
                if not os.path.exists(subsubfolder_path):
                    os.mkdir(subsubfolder_path)
                create_variance_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, num_colors)
        
        for file in files:
            if file == "PCA_scores.txt":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, delimiter="\t")
                title = os.path.splitext(file)[0]
                subsubfolder = os.path.basename(root)
                subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
                if not os.path.exists(subsubfolder_path):
                    os.mkdir(subsubfolder_path)
                create_score_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, num_colors, 1, experiment_classification)
                
                pc_columns = df.columns[1:num_colors+1].tolist()
                scores_corr_folder = os.path.join(subsubfolder_path, "Scores correlations")
                if not os.path.exists(scores_corr_folder):
                    os.mkdir(scores_corr_folder)

                combinations = list(itertools.product(pc_columns, repeat=2))[:num_colors**2]
                fig, axes = plt.subplots(nrows=num_colors, ncols=num_colors, figsize=(100, 100))
                for i, (pc1, pc2) in enumerate(combinations):
                    ax = axes[i // num_colors, i % num_colors]
                    ax.scatter(df[pc1], df[pc2], c='blue', alpha=0.7)
                    ax.set_xlabel(pc1)
                    ax.set_ylabel(pc2)
                    ax.set_title(f"{pc2} vs {pc1}")
                    corr_coef, p_value = pearsonr(df[pc1], df[pc2])
                    r_squared = corr_coef ** 2
                    spearman_corr, p_value = spearmanr(df[pc1], df[pc2])
                    ax.annotate(fr"$R^2$: {r_squared:.6f}, Spearman's $\rho$: {spearman_corr:.6f}", xy=(0.5, 0.9), xycoords='axes fraction',
                                ha='center', va='center', fontsize=10)

                plt.tight_layout()
                plt.savefig(os.path.join(scores_corr_folder, "PC_Scatter_Matrix.png"))
                plt.close()

                correlation_matrix = np.zeros((len(pc_columns), len(pc_columns)))
                for i, pc1 in enumerate(pc_columns):
                    for j, pc2 in enumerate(pc_columns):
                        spearman_corr, _ = spearmanr(df[pc1], df[pc2])
                        correlation_matrix[i, j] = spearman_corr

                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='magma', xticklabels=pc_columns, yticklabels=pc_columns, vmin=-0.25, vmax=0.25)
                plt.xlabel("PCs")
                plt.ylabel("PCs")
                plt.title("Spearman Correlation Coefficients Matrix")
                plt.savefig(os.path.join(scores_corr_folder, "Spearman_Correlation_Matrix.png"))
                plt.close()
        
        for file in files:
            if file == "PCA_scores.txt":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, delimiter="\t")
                title = os.path.splitext(file)[0] + '_stacked'
                subsubfolder = os.path.basename(root)
                subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
                if not os.path.exists(subsubfolder_path):
                    os.mkdir(subsubfolder_path)
                create_stacked_score_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, num_colors, 1, experiment_classification)
        
        for file in files:
            if file == "PCA_eigenspectra.txt":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, delimiter="\t")
                title = os.path.splitext(file)[0]
                subsubfolder = os.path.basename(root)
                subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
                if not os.path.exists(subsubfolder_path):
                    os.mkdir(subsubfolder_path)
                create_eigenspectra_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, num_colors, 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PCA plots")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the PCA files")
    parser.add_argument("num_colors", type=int, help="Number of principal components to plot")
    args = parser.parse_args()

    process_pca(args.folder_path, args.num_colors)
