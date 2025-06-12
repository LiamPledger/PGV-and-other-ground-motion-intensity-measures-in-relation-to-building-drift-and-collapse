# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:34:05 2025

@author: ljp70
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
cm = 1 / 2.54

# Labels for plot
im_labels = {
    'PGA': 'PGA', 'PGV': 'PGV', 'PGD': 'PGD', 'AI': 'AI', 'CAV': 'CAV',
    'ASI': 'ASI', 'VSI': 'VSI', 'DSI': 'DSI',
    'SaT1_T1': r'$S_a(T_1)$', 'SaT1_0.5': r'$S_a(0.5s)$', 'SaT1_1.0': r'$S_a(1.0s)$', 'SaT1_2.0': r'$S_a(2.0s)$',
    'SaAvg_T1': r'$S_{a,avg}(T_1)$', 'SaAvg_0.5': r'$S_{a,avg}(0.5s)$',
    'SaAvg_1.0': r'$S_{a,avg}(1.0s)$', 'SaAvg_2.0': r'$S_{a,avg}(2.0s)$',
    'FIV3_T1': r'FIV3$(T_1)$', 'FIV3_0.5': r'FIV3$(0.5s)$', 'FIV3_1.0': r'FIV3$(1.0s)$', 'FIV3_2.0': r'FIV3$(2.0s)$',
    'PFA_T1': r'PFA$(T_1)$', 'PFA_0.5': r'PFA$(0.5s)$', 'PFA_1.0': r'PFA$(1.0s)$', 'PFA_2.0': r'PFA$(2.0s)$',
    'FIV3_1Hz_T1': r'FIV3$(T_1)$', 'FIV3_1Hz_0.5': r'FIV3$(0.5s)$', 'FIV3_1Hz_1.0': r'FIV3$(1.0s)$', 'FIV3_1Hz_2.0': r'FIV3$(2.0s)$',
}

# Load data
IM_results = os.path.join("log_std_and_cov_scaled_IMs.xlsx")
df_log_std = pd.read_excel(IM_results, sheet_name='log_std', index_col=0)


# Remove metadata columns
df_log_std.drop(columns=["T1", "N", "Vb", "N/T"], inplace=True)


def plot_log_std_boxplot(width, height, fig_name, exclude_IMs=None):
    if exclude_IMs is None:
        exclude_IMs = []


    # Define all possible IMs
    IM_cols = ['PGA', 'PGV', 'PGD', 'AI', 'CAV', 'ASI', 'VSI', 'DSI',
               'SaT1_T1', 'SaT1_0.5', 'SaT1_1.0', 'SaT1_2.0',
               'SaAvg_T1', 'SaAvg_0.5', 'SaAvg_1.0', 'SaAvg_2.0',
               'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0',
               'PFA_T1', 'PFA_0.5', 'PFA_1.0', 'PFA_2.0',
               'FIV3_1Hz_T1', 'FIV3_1Hz_0.5', 'FIV3_1Hz_1.0', 'FIV3_1Hz_2.0',]

    # Filter out excluded IMs
    IM_cols = [col for col in IM_cols if col not in exclude_IMs]

    # Create plot
    plt.figure(figsize=(width*cm, height*cm), dpi=400)

    df_log_std[IM_cols].boxplot(grid=False, showfliers=False, widths=0.3,
        boxprops=dict(color='black', linewidth=0.75),
        whiskerprops=dict(color='k', linewidth=0.75),
        medianprops=dict(color='red', linewidth=0.75),
        capprops=dict(color='k')
    )

    plt.xticks(ticks=range(1, len(IM_cols) + 1),
               labels=[im_labels[col] for col in IM_cols],
               rotation=45)

    plt.tick_params(axis='x', which='minor', bottom=False, top=False)
    plt.grid(True, linestyle='-', alpha=0.1)
    plt.ylabel(r"Variation in $\sigma_{ln}$ for different IMs")
    plt.xlabel("Intensity Measure")
    plt.ylim(0, 1.2)
    plt.tight_layout()
    filename = f"Plots/{fig_name}_1.pdf"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
        
       
    
    # =============================================================================
    # Plot 2: Scatter Plot by Model Type
    # =============================================================================
    
    # Define model types based on index patterns
    model_types = {'RC': [], 'RCW': [], 'BRB': []}
    label_map = {'RC': 'RC Frame', 'RCW': 'RC Wall', 'BRB': 'BRBF'}
    
    for model in df_log_std.index:
        model_str = str(model)  # Ensure it's a string
        if model_str.startswith("RCW"):
            model_types['RCW'].append(model)
        elif model_str.startswith("BRB"):
            model_types['BRB'].append(model)
        else:
            model_types['RC'].append(model)
    
    markers = {'RC': 'o', 'RCW': 's', 'BRB': 'D'}
    colors = {'RC': 'none', 'RCW': 'dimgrey', 'BRB': 'lightgrey'}

    # Plot
    plt.figure(figsize=(width*cm, height*cm), dpi=400)
    for model_type, model_list in model_types.items():
        mean_vals = df_log_std.loc[model_list][IM_cols].mean(axis=0)
        plt.scatter(IM_cols, mean_vals, label=label_map[model_type],
                    s=25,                   
                    edgecolor='black', facecolor=colors[model_type],
                    alpha = 0.7,
                    marker=markers[model_type], linewidth=0.6)
    plt.xticks(ticks=range(len(IM_cols)), labels=[im_labels[col] for col in IM_cols], rotation=45)
    
    plt.ylabel(r"Mean $\sigma_{ln}$")
    plt.xlabel("Intensity Measure")
    
    plt.ylim(0, 1.2)
    # Turn off minor ticks on x-axis
    plt.tick_params(axis='x', which='minor', bottom=False, top=False)
    
    plt.grid(True, linestyle='-', alpha=0.1)
    plt.legend(frameon=False)
    plt.tight_layout()
    filename = f"Plots/{fig_name}_2.pdf"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    
    
# Exclude selected IMs
plot_log_std_boxplot(exclude_IMs=[
    'SaT1_0.5', 'SaT1_1.0', 'SaT1_2.0',
    'SaAvg_0.5', 'SaAvg_1.0', 'SaAvg_2.0',
    'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0',
    'PFA_0.5', 'PFA_1.0', 'PFA_2.0',
    'FIV3_1Hz_0.5', 'FIV3_1Hz_1.0', 'FIV3_1Hz_2.0',],
    width=12,
    height=9,
    fig_name = "All IMs",
)


# Exclude selected IMs
plot_log_std_boxplot(
    exclude_IMs=['PGA', 'PGV', 'PGD', 'AI', 'CAV', 'ASI', 'VSI', 'DSI',
                 'SaT1_T1', 'SaT1_0.5', 'SaT1_1.0', 'SaT1_2.0',
                 'SaAvg_T1', 'SaAvg_0.5', 'SaAvg_1.0', 'SaAvg_2.0',
                 'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0',
                 'FIV3_1Hz_T1', 'FIV3_1Hz_0.5', 'FIV3_1Hz_1.0', 'FIV3_1Hz_2.0'],
    width=8,
    height=8,
    fig_name = "PFA IMs",
)


# Exclude selected IMs
plot_log_std_boxplot(
    exclude_IMs=['PGA', 'PGV', 'PGD', 'AI', 'CAV', 'ASI', 'VSI', 'DSI',
                 'SaT1_T1', 'SaT1_0.5', 'SaT1_1.0', 'SaT1_2.0',
                 'SaAvg_T1', 'SaAvg_0.5', 'SaAvg_1.0', 'SaAvg_2.0',
                 'PFA_T1', 'PFA_0.5', 'PFA_1.0', 'PFA_2.0',
                 'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0'],
    width=8,
    height=8,
    fig_name = "FIV3 IMs",
)

# Exclude selected IMs
plot_log_std_boxplot(
    exclude_IMs=['PGA', 'PGV', 'PGD', 'AI', 'CAV', 'ASI', 'VSI', 'DSI',
                 'SaT1_T1', 'SaT1_0.5', 'SaT1_1.0', 'SaT1_2.0',
                 'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0',
                 'PFA_T1', 'PFA_0.5', 'PFA_1.0', 'PFA_2.0',
                 'FIV3_1Hz_T1', 'FIV3_1Hz_0.5', 'FIV3_1Hz_1.0', 'FIV3_1Hz_2.0'],
    width=8,
    height=8,
    fig_name = "SaAvg IMs",

)


# Exclude selected IMs
plot_log_std_boxplot(
    exclude_IMs=['PGA', 'PGV', 'PGD', 'AI', 'CAV', 'ASI', 'VSI', 'DSI',
                 'SaAvg_T1', 'SaAvg_0.5', 'SaAvg_1.0', 'SaAvg_2.0',
                 'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0',
                 'PFA_T1', 'PFA_0.5', 'PFA_1.0', 'PFA_2.0',
                 'FIV3_1Hz_T1', 'FIV3_1Hz_0.5', 'FIV3_1Hz_1.0', 'FIV3_1Hz_2.0'],
    width=8,
    height=8,
    fig_name = "SaT1 IMs",

)



# Exclude selected IMs
plot_log_std_boxplot(
    exclude_IMs=['PGA', 'PGD', 'AI', 'CAV', 'ASI', 'DSI',
               'SaT1_T1', 'SaT1_0.5', 'SaT1_1.0', 'SaT1_2.0',
               'SaAvg_0.5', 'SaAvg_1.0', 'SaAvg_2.0',
               'FIV3_T1', 'FIV3_0.5', 'FIV3_1.0', 'FIV3_2.0',
               'PFA_0.5', 'PFA_1.0', 'PFA_2.0',
               'FIV3_1Hz_0.5', 'FIV3_1Hz_1.0', 'FIV3_1Hz_2.0'],
    width=8,
    height=8,
    fig_name = "5 IMs",
)



# Load data
IM_results = os.path.join("log_std_and_cov_scaled_IMs.xlsx")
df_log_std = pd.read_excel(IM_results, sheet_name='log_std', index_col=0)

# Extract metadata
metadata = {
    'T1': df_log_std.pop("T1"),
    'Ti': df_log_std.pop("Ti"),
    'N': df_log_std.pop("N"),
    'Vb': df_log_std.pop("Vb"),
    'N/T': df_log_std.pop("N/T")
}

# Function to generate scatter Plots_Ti
def plot_log_std_vs_metadata(df_log_std, metadata, im_list, x_options, xlims_dict=None):
    
    for im in im_list:
        for x_key in x_options:
            x_vals = metadata[x_key]
            y_vals = df_log_std[im]

            plt.figure(figsize=(9 * cm, 8 * cm), dpi=400)
            plt.scatter(x_vals, y_vals, edgecolor='black', facecolor = 'none', s=20, lw = 0.5)

            # Labels
            x_label = {
                "T1": r"$T_1$ (sec)",
                "Ti": r"$T_i$ (sec)",
                "N": r"$N$ (storeys)",
                "Vb": r"$V_b/W$",
                "N/T": r"$N/T_1 $"
            }.get(x_key, x_key)

            plt.xlabel(x_label)
            im_label = im_labels.get(im, im).replace('$', '')
            plt.ylabel(fr"$\sigma_{{ln,{im_label}}}$")

            # Y and X limits
            plt.ylim(0, 1)
            if xlims_dict and x_key in xlims_dict:
                plt.xlim(xlims_dict[x_key])

            plt.grid(True, linestyle='-', alpha=0.1)
            plt.tight_layout()
            # Save the figure
            if x_key == 'N/T':
                filename = f"Plots/{im}_NT.pdf"
            else:
                filename = f"Plots/{im}_{x_key}.pdf"
            plt.savefig(filename, dpi=400, bbox_inches='tight')
            plt.show()

# --- Define IMs and x-axis options ---
im_list = ['PGV', 'VSI', 'SaT1_T1', 'SaAvg_T1', 'FIV3_1Hz_T1', 'PFA_T1']
x_options = ['Ti']

# Optional x-axis limits (in dictionary form)
xlims_dict = {
    'Ti': (0, 2.5),
}
    
# --- Call the function ---
plot_log_std_vs_metadata(df_log_std, metadata, im_list, x_options, xlims_dict)





# Define model types based on index patterns
model_types = {'RC': [], 'RCW': [], 'BRB': []}
label_map = {'RC': 'RC Frame', 'RCW': 'RC Wall', 'BRB': 'BRBF'}
colors = {'RC': 'none', 'RCW': 'dimgrey', 'BRB': 'lightgrey'}
markers = {'RC': 'o', 'RCW': 's', 'BRB': 'D'}

def plot_log_std_vs_metadata(df_log_std, metadata, im_list, x_options, xlims_dict=None):
    # Assign models to their types
    for model in df_log_std.index:
        model_str = str(model)
        if model_str.startswith("RCW"):
            model_types['RCW'].append(model)
        elif model_str.startswith("BRB"):
            model_types['BRB'].append(model)
        else:
            model_types['RC'].append(model)

    for im in im_list:
        
        # Compute and print the mean beta for RC + RCW only
        selected_models = model_types['RC'] + model_types['RCW']
        mean_beta = df_log_std.loc[selected_models, im].mean()
        print(f"Mean β for {im}: {mean_beta:.3f}")
        
        mean_beta = df_log_std.loc[model_types['RC'], im].mean()
        print(f"Mean β for RC frames - {im}: {mean_beta:.3f}")
        
        mean_beta = df_log_std.loc[model_types['RCW'], im].mean()
        print(f"Mean β for RCW - {im}: {mean_beta:.3f}")
        
        mean_beta = df_log_std.loc[model_types['BRB'], im].mean()
        print(f"Mean β for BRBF - {im}: {mean_beta:.3f}")
        
        for x_key in x_options:
            x_vals = metadata[x_key]
            y_vals = df_log_std[im]

            plt.figure(figsize=(9 * cm, 8 * cm), dpi=400)

            # Plot each type with its marker and color
            for model_type, models in model_types.items():
                mask = df_log_std.index.isin(models)
                if any(mask):
                    plt.scatter(
                        x_vals[mask], y_vals[mask],
                        edgecolor='black', facecolor=colors[model_type],
                        alpha = 0.7, s=20, lw=0.5,
                        label=label_map[model_type],
                        marker=markers[model_type]
                    )
                    
            # Labels
            x_label = {
                "T1": r"$T_1$ (sec)",
                "Ti": r"$T_1$ (sec)",
                "N": r"$N$ (storeys)",
                "Vb": r"$V_b/W$",
                "N/T": r"$N/T_1 $"
            }.get(x_key, x_key)

            plt.xlabel(x_label)
            im_label = im_labels.get(im, im).replace('$', '')
            plt.ylabel(fr"$\sigma_{{ln,{im_label}}}$")

            # Y and X limits
            plt.ylim(0, 1)
            if xlims_dict and x_key in xlims_dict:
                plt.xlim(xlims_dict[x_key])

            plt.grid(True, linestyle='-', alpha=0.1)
            plt.legend(frameon=True)
            plt.tight_layout()

            # Save the figure
            if x_key == 'N/T':
                filename = f"Plots/{im}_NT.pdf"
            else:
                filename = f"Plots/{im}_{x_key}.pdf"
            plt.savefig(filename, dpi=400, bbox_inches='tight')
            plt.show()

# --- Define IMs and x-axis options ---
im_list = ['PGV', 'VSI', 'SaT1_T1', 'SaAvg_T1', 'FIV3_1Hz_T1', 'PFA_T1']

x_options = ['Ti']

# Optional x-axis limits (in dictionary form)
xlims_dict = {
    'T1': (0, 4),
    'Ti':(0, 2.5),
    'N': (0, 25),
    'Vb': (0, 1),
    'N/T': (0, 15)
}

# --- Call the function ---
plot_log_std_vs_metadata(df_log_std, metadata, im_list, x_options, xlims_dict)



