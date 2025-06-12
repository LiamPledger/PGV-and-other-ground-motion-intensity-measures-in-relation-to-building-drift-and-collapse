# -*- coding: utf-8 -*-
"""
Created on Fri May  9 21:04:39 2025

@author: ljp70
"""

# Sufficiency plots
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import re
import statsmodels.api as sm

plt.style.use('science')
cm = 1 / 2.54
tot_model = 52 #
tot_gms   = 100


IMs_raw = pd.read_excel('Intensity_Measures.xlsx') # read pandas data file
IMs = IMs_raw.dropna() # Drop rows where any cell has NaN
SF_raw = pd.read_excel('amplification_factors_All.xlsx') # read pandas data file 
# SF_raw = pd.read_excel('amplification_factors_All (1).xlsx') # read pandas data file 

SF = SF_raw.dropna() # Drop rows where any cell has NaN

SaC     = np.zeros([SF.shape[0],SF.shape[1]-2])
PGVC    = np.zeros([SF.shape[0],SF.shape[1]-2])
VSIC    = np.zeros([SF.shape[0],SF.shape[1]-2])
SAavgC  = np.zeros([SF.shape[0],SF.shape[1]-2])
PFAC    = np.zeros([SF.shape[0],SF.shape[1]-2])
FIV3C   = np.zeros([SF.shape[0],SF.shape[1]-2])

SaRatio = np.zeros([SF.shape[0],SF.shape[1]-2])
for i in range(SF.shape[0]): #range(SF.shape[0])
    model_period = SF.iloc[0,1]
    # model_period = f"{model_period:.2f}"

    IM_vals   = IMs[IMs.iloc[:, 0] == f"SaT1_{model_period:.2f}"]
    if IM_vals.size != 0:
        SaAvg_vals = IMs[IMs.iloc[:, 0] == f'SaAvg_{model_period:.2f}']
        PFA_vals = IMs[IMs.iloc[:, 0] == f'PFA_{model_period:.2f}']
        FIV3_vals = IMs[IMs.iloc[:, 0] == f'FIV3_1Hz_{model_period:.2f}']
    if IM_vals.size == 0:
        IM_vals = IMs[IMs.iloc[:, 0] == f'SaT1_{0.01+model_period:.2f}']
        PFA_vals = IMs[IMs.iloc[:, 0] == f'PFA_{0.01+model_period:.2f}']
        FIV3_vals = IMs[IMs.iloc[:, 0] == f'FIV3_1Hz_{0.01+model_period:.2f}']
        SaAvg_vals = IMs[IMs.iloc[:, 0] == f'SaAvg_{0.01+model_period:.2f}']
        if IM_vals.size == 0:
            IM_vals = IMs[IMs.iloc[:, 0] == f'SaT1_{-0.01+model_period:.2f}']
            PFA_vals = IMs[IMs.iloc[:, 0] == f'PFA_{-0.01+model_period:.2f}']
            FIV3_vals = IMs[IMs.iloc[:, 0] == f'FIV3_1Hz_{-0.01+model_period:.2f}']
            SaAvg_vals = IMs[IMs.iloc[:, 0] == f'SaAvg_{-0.01+model_period:.2f}']
            

    SaC[i,:]  = SF.iloc[i,2:]*IM_vals.iloc[0,1:] # SaT1 at collapse 
    PGVC[i,:] = SF.iloc[i,2:]*IMs.iloc[1,1:] # PGV at collapse 
    VSIC[i,:] = SF.iloc[i,2:]*IMs.iloc[6,1:] # VSI at collapse 
    PFAC[i,:] = SF.iloc[i,2:]*PFA_vals.iloc[0,1:] # VSI at collapse
    FIV3C[i,:] = SF.iloc[i,2:]*FIV3_vals.iloc[0,1:] # VSI at collapse 
    SAavgC[i,:]= SF.iloc[i,2:]*SaAvg_vals.iloc[0,1:] # SAavg at collapse 
    
    SaRatio[i,:]  = IM_vals.iloc[0,1:]/SaAvg_vals.iloc[0,1:]
    
    
mstart = np.arange(0,5200,100)
mend   = np.arange(100,5201,100)
IM_list     = [SaC,PGVC,VSIC,PFAC,FIV3C,SAavgC]
IM_names    = ['Sa','PGV','VSI','PFA','FIV3','SAavg']
plot_yaxis  = ['$\epsilon_{S_a(T_1)}$','$\epsilon_{PGV}$','$\epsilon_{VSI}$','$\epsilon_{PFA(T_1)}$','$\epsilon_{FIV3(T_1)}$','$\epsilon_{S_{a,avg}}$']



# Load Ds data
Ds_x = np.genfromtxt('Ds_x.txt')
Ds_y = np.genfromtxt('Ds_y.txt')

# Prepare Ds array similar to SaC etc.
Ds = np.zeros_like(SaC)
Ds[0, np.arange(0, tot_gms, 2)] = Ds_x
Ds[0, np.arange(1, tot_gms, 2)] = Ds_y
Ds[1:, :] = Ds[0, :]


# Initialize storage
sufficiency = {}
SaR_std_list = []
Ds_std_list = []
SaR_R2_list = []
Ds_R2_list = []

# Loop through IMs
for idx, IM_data in enumerate(IM_list):
    IM_name = IM_names[idx]
    sufficiency[IM_name + '_SaR'] = []
    sufficiency[IM_name + '_Ds'] = []


    for model_no in range(tot_model):
        # Normalize IM values for this model
        IM_vals_model = IM_data[model_no, :]
        IM_norm = IM_vals_model / np.median(IM_vals_model)
        
        # --- SaRatio regression (filtered) ---
        SaRatio_vals_full = SaRatio[model_no, :]
        mask_sar = SaRatio_vals_full <= 100
        SaRatio_vals = SaRatio_vals_full[mask_sar]
        IM_norm_sar = IM_norm[mask_sar]
        
        SaRatio_vals_norm = SaRatio_vals / np.mean(SaRatio_vals)  # normalize by mean
        X_sar = sm.add_constant(SaRatio_vals_norm)
        model_sar = sm.OLS(IM_norm_sar, X_sar).fit()
        slope_sar = model_sar.params[1]
        sufficiency[IM_name + '_SaR'].append(slope_sar)
        
        if idx == 0:
            SaR_std_list.append(np.std(SaRatio_vals))  # filtered std

        # --- Ds regression ---
        Ds_col = Ds.reshape(-1, 1)[mstart[model_no]:mend[model_no]].flatten()
        Ds_col_norm = Ds_col / np.mean(Ds_col)  # normalize by mean
        
        X_ds = sm.add_constant(Ds_col_norm)
        model_ds = sm.OLS(IM_norm, X_ds).fit()
        slope_ds = model_ds.params[1]
        sufficiency[IM_name + '_Ds'].append(slope_ds)

        if idx == 0:
            Ds_std_list.append(np.std(Ds_col))

# Convert to arrays for processing
SaR_std_array = np.array(SaR_std_list)
Ds_std_array = np.array(Ds_std_list)

# Normalize and scale by standard deviation
norm_scaled_slopes_SaR = []
norm_scaled_slopes_Ds = []

# Compute standard deviation of the slopes for error bars
slope_std_SaR = []
slope_std_Ds = []

for name in IM_names:
    slopes_SaR = np.array(sufficiency[name + '_SaR'])
    slopes_Ds = np.array(sufficiency[name + '_Ds'])

    # Standard deviation-based scaling
    norm_SaR = np.abs(slopes_SaR) #* SaR_std_array
    norm_Ds = np.abs(slopes_Ds)   #* Ds_std_array

    norm_scaled_slopes_SaR.append(np.mean(norm_SaR))
    norm_scaled_slopes_Ds.append(np.mean(norm_Ds))
    
    slope_std_SaR.append(np.std(norm_SaR))  # Standard deviation across models
    slope_std_Ds.append(np.std(norm_Ds))  # Standard deviation across models

slope_std_SaR = np.array(slope_std_SaR)  # Convert to array
slope_std_Ds = np.array(slope_std_Ds)  # Convert to array



# Get model periods from SF (column 1)
periods = SF.iloc[:, 1].values

# Identify indices where period < 1s
indices_under_1s = np.where(periods < 1.0)[0]
# Extract PGV sufficiency slopes for those models
PGV_vals_flat = [
    abs(sufficiency['PGV_SaR'][i]) for i in indices_under_1s
    if i < len(sufficiency['PGV_SaR']) and not np.isnan(sufficiency['PGV_SaR'][i])
]

VSI_vals_flat = [
    abs(sufficiency['VSI_SaR'][i]) for i in indices_under_1s
    if i < len(sufficiency['VSI_SaR']) and not np.isnan(sufficiency['VSI_SaR'][i])
]

# Convert to array for statistics
PGV_vals_flat = np.array(PGV_vals_flat)
VSI_vals_flat = np.array(VSI_vals_flat)

# Compute mean and standard deviation
PGV_mean_under_1s = np.mean(PGV_vals_flat)
PGV_std_under_1s = np.std(PGV_vals_flat)

VSI_mean_under_1s = np.mean(VSI_vals_flat)
VSI_std_under_1s = np.std(VSI_vals_flat)

# Add to bar chart lists
# Ensure lists before appending
slope_std_SaR = list(slope_std_SaR)
norm_scaled_slopes_SaR = list(norm_scaled_slopes_SaR)

# norm_scaled_slopes_SaR.extend([PGV_mean_under_1s, VSI_mean_under_1s])
# slope_std_SaR.extend([PGV_std_under_1s, VSI_std_under_1s])

x_axis_labels = ['$S_a(T_1)$', '$PGV$', '$VSI$', '$PFA$', '$FIV3$', '$S_{a,avg}$']

# x_axis_labels.extend(['$PGV1$', '$VSI1$'])



# --- Plot 1: Normalized Slopes vs SaRatio ---
plt.figure(figsize=(10*cm, 8*cm), dpi=300)
plt.bar(x_axis_labels, norm_scaled_slopes_SaR, color='salmon')
plt.ylabel(r'Mean $\left| \text{Slope} \right| \text{for } S_a\mathrm{Ratio}$', fontsize=11)
plt.xlabel('Intensity Measures', fontsize=11)
plt.grid(axis='y', alpha=0.1)
plt.ylim(0, 1.4)
plt.tight_layout()
plt.savefig("Sufficiency_SaRatio_std.pdf", dpi=400, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10*cm, 8*cm), dpi=300)
plt.bar(
    x_axis_labels,
    norm_scaled_slopes_SaR,
    yerr=slope_std_SaR,
    capsize=5,
    color='salmon',
    edgecolor='none',
    error_kw={
        'elinewidth': 0.7,   # Thickness of the error bar line
        'ecolor': 'black',   # Color of the error bar
        'capthick': 0.7      # Thickness of the cap lines
    },
    label='Mean |Slope|'
)

plt.ylabel(r'Mean $\left| \text{Slope} \right| \text{for } S_a\mathrm{Ratio}$', fontsize=11)
plt.xlabel('Intensity Measures', fontsize=11)
plt.grid(axis='y', alpha=0.1)
plt.ylim(0, 1.4)
plt.tight_layout()
plt.savefig("Sufficiency_SaRatio_std_with_errorbars.pdf", dpi=400, bbox_inches='tight')
plt.show()



# --- Plot 2: Normalized Slopes vs Ds ---
plt.figure(figsize=(10*cm, 8*cm), dpi=300)
plt.bar(x_axis_labels, norm_scaled_slopes_Ds, color='indianred')
plt.ylabel(r'Mean $\left| \text{Slope} \right| \text{for } D_{s, 5-75}$', fontsize=11)
plt.xlabel('Intensity Measures', fontsize=11)
plt.grid(axis='y', alpha=0.1)
plt.ylim(0, 1.4)
plt.tight_layout()
plt.savefig("Sufficiency_Ds_std.pdf", dpi=400, bbox_inches='tight')
plt.show()
            


plt.figure(figsize=(10*cm, 8*cm), dpi=300)
plt.bar(
    x_axis_labels,
    norm_scaled_slopes_Ds,
    yerr=slope_std_Ds,
    capsize=5,
    color='indianred',
    edgecolor='none',
    error_kw={
        'elinewidth': 0.7,   # Thickness of the error bar line
        'ecolor': 'black',   # Color of the error bar
        'capthick': 0.7      # Thickness of the cap lines
    },
    label='Mean |Slope|'
)

plt.ylabel(r'Mean $\left| \text{Slope} \right| \text{for } D_{s, 5-75}$', fontsize=11)
plt.xlabel('Intensity Measures', fontsize=11)
plt.grid(axis='y', alpha=0.1)
plt.ylim(0, 1.4)
plt.tight_layout()
plt.savefig("Sufficiency_Ds_std_with_errorbars.pdf", dpi=400, bbox_inches='tight')
plt.show()







""" Plotting the trendlines between IMs at collpase and SaRatio """

def plot_SaT1_Sufficiency(IM_array, IM_name, units,  model_index):
    """
    Plot a scatter plot of a selected IM (e.g. PGV) vs SaRatio for a specific model,
    and overlay the linear regression trend line.

    Parameters:
    - IM_array: numpy array of the IM (e.g., PGVC, VSIC, etc.)
    - IM_name: name string for labeling
    - model_index: index of the structural model (0 to tot_model-1)
    """
    
    x_vals = SaRatio[model_index, :]
    y_vals = IM_array[model_index, :]

    # Fit linear regression
    X = sm.add_constant(x_vals)
    model = sm.OLS(y_vals, X).fit()
    trendline = model.predict(X)
    r_squared = model.rsquared

    # Plot
    plt.figure(figsize=(9 * cm, 7 * cm), dpi=400)
    plt.scatter(x_vals, y_vals, alpha=1, s = 25, edgecolor='k', facecolor = 'none', lw = 0.5)
    plt.plot(x_vals, trendline, color='red', label=f'$S_a(T_1)_c$ = {model.params[1]:.2f}$S_aRatio + C$\n$R^2$ = {r_squared:.2f}')
    plt.xlabel('$S_aRatio$', fontsize=11)
    plt.ylabel(f'{IM_name} at collapse ({units})', fontsize=11)
    # plt.title(f'{IM_name} vs SaRatio (Model {model_index})')
    plt.grid(True, alpha=0.1)
    plt.xlim(0, 5)
    plt.ylim(0, 4)
    plt.legend()
    plt.tight_layout()
    filename = f"SaT1_vs_SaRatio.pdf"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    
    
        
    x_vals = Ds[model_index, :]
    y_vals = IM_array[model_index, :]

    # Fit linear regression
    X = sm.add_constant(x_vals)
    model = sm.OLS(y_vals, X).fit()
    trendline = model.predict(X)
    r_squared = model.rsquared
    
    # Plot
    plt.figure(figsize=(9 * cm, 7 * cm), dpi=400)
    plt.scatter(x_vals, y_vals, alpha=1, s = 20, edgecolor='k', facecolor = 'none', lw = 0.5)
    plt.plot(x_vals, trendline, color='red', label=f'$S_a(T_1)_c$ = {model.params[1]:.3f}$D_{{s, 5-75}} + C$\n$\mathrm{{R}}^2$ = {r_squared:.2f}')
    plt.xlabel('$D_{s, 5-75}$ (sec)', fontsize=11)
    plt.ylabel(f'{IM_name} at collapse ({units})', fontsize=11)
    # plt.title(f'{IM_name} vs $D_s$ (Model {model_index})')
    plt.grid(True, alpha=0.1)
    plt.xlim(0, 50)
    plt.ylim(0, 4)
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = "SaT1_vs_Ds.pdf"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()



# for i in range(0, tot_model):
#     plot_SaAvg_vs_SaRatio(SaC, IM_name='Sa(T1)', units = 'm/s',  model_index=i)


i = 6
plot_SaT1_Sufficiency(SaC, IM_name='$S_a(T_1)$', units = 'g',  model_index=i)








def plot_scatter_vs_SaRatio(IM_array, IM_name, units,  model_index, y_max):
    """
    Plot a scatter plot of a selected IM (e.g. PGV) vs SaRatio for a specific model,
    and overlay the linear regression trend line.

    Parameters:
    - IM_array: numpy array of the IM (e.g., PGVC, VSIC, etc.)
    - IM_name: name string for labeling
    - model_index: index of the structural model (0 to tot_model-1)
    """
    
    x_vals = SaRatio[model_index, :]
    y_vals = IM_array[model_index, :]

    # Fit linear regression
    X = sm.add_constant(x_vals)
    model = sm.OLS(y_vals, X).fit()
    trendline = model.predict(X)
    r_squared = model.rsquared

    # Plot
    plt.figure(figsize=(9 * cm, 7 * cm), dpi=400)
    plt.scatter(x_vals, y_vals, alpha=1, s = 20, edgecolor='k', facecolor = 'none', lw = 0.5)
    plt.plot(x_vals, trendline, color='red', label=f'$PGV_c$ = {model.params[1]:.2f}$S_a\mathrm{{Ratio}} + C$\n$\mathrm{{R}}^2$ = {r_squared:.2f}')
    plt.xlabel('$S_a\mathrm{Ratio}$', fontsize=11)
    plt.ylabel(f'{IM_name} at collapse ({units})', fontsize=11)
    # plt.title(f'{IM_name} vs SaRatio (Model {model_index})')
    plt.grid(True, alpha=0.1)
    plt.xlim(0, 5)
    plt.ylim(0, y_max)
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = "PGV_vs_SaRatio.pdf"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()


plot_scatter_vs_SaRatio(PGVC, IM_name='PGV', units = 'm/s',  model_index=1, y_max = 2.5)



def plot_scatter_vs_Ds(IM_array, IM_name, units,  model_index, y_max):
    """
    Plot a scatter plot of a selected IM (e.g. PGV) vs SaRatio for a specific model,
    and overlay the linear regression trend line.

    Parameters:
    - IM_array: numpy array of the IM (e.g., PGVC, VSIC, etc.)
    - IM_name: name string for labeling
    - model_index: index of the structural model (0 to tot_model-1)
    """
    
    x_vals = Ds[model_index, :]
    y_vals = IM_array[model_index, :]

    # Fit linear regression
    X = sm.add_constant(x_vals)
    model = sm.OLS(y_vals, X).fit()
    trendline = model.predict(X)
    r_squared = model.rsquared
    
    # Plot
    plt.figure(figsize=(9 * cm, 7 * cm), dpi=400)
    plt.scatter(x_vals, y_vals, alpha=1, s = 20, edgecolor='k', facecolor = 'none', lw = 0.5)
    plt.plot(x_vals, trendline, color='red', label=f'$PGV_c$ = {model.params[1]:.3f}$D_{{s, 5-75}} + C$\n$\mathrm{{R}}^2$ = {r_squared:.2f}')
    plt.xlabel('$D_{s, 5-75}$ (sec)', fontsize=11)
    plt.ylabel(f'{IM_name} at collapse ({units})', fontsize=11)
    # plt.title(f'{IM_name} vs $D_s$ (Model {model_index})')
    plt.grid(True, alpha=0.1)
    plt.xlim(0, 50)
    plt.ylim(0, y_max)
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = "PGV_vs_Ds.pdf"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()


# plot_scatter_vs_Ds(PGVC, IM_name='PGV', units = 'm/s',  model_index=1, y_max = 2.5)

