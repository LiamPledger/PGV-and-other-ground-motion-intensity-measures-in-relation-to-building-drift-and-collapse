# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:26:57 2025

@author: ljp70
"""

import os
import pandas as pd
import numpy as np


base_dir    = r"D:\HC_IDA_GMs"
IM_file     = os.path.join(base_dir, "Intensity_Measures.xlsx")
SF_file     = os.path.join(base_dir, "amplification_factors_All.xlsx")

df_IMs      = pd.read_excel(IM_file, index_col=0)
df_SF       = pd.read_excel(SF_file, index_col=0)

T1_dict = df_SF["T1 (sec)"].to_dict()
df_SF = df_SF.drop(columns=["T1 (sec)", "Ti (sec)"], errors='ignore')


# --- Function to select closest-period IM -----------------------------------

def find_closest_row(im_prefix, period):
    matching = [row for row in df_IMs.index if row.startswith(im_prefix)]
    if not matching:
        return None
    periods = [float(row.split('_')[-1]) for row in matching]
    closest_idx = np.argmin(np.abs(np.array(periods) - period))
    return matching[closest_idx]


# --- Config -----------------------------------------------------------------

period_dependent_prefixes = ['SaT1_', 'SaAvg_', 'PFA_', 'FIV3_', 'FIV3_1Hz_']
fixed_periods = [0.5, 1.0, 2.0]

# --- Main computation -------------------------------------------------------

log_std_results = {}
cov_results = {}
scaled_vals_dict = {}

for model_id in df_SF.index:
    T1 = T1_dict[model_id]
    sf_vals = df_SF.loc[model_id].values

    log_std_row = {}
    cov_row = {}

    for im in df_IMs.index:
        is_periodic = any(im.startswith(prefix) for prefix in period_dependent_prefixes)

        if is_periodic:
            im_prefix = '_'.join(im.split('_')[:-1]) + '_'
            # --- T1 Period ---
            for label, period in [('T1', T1), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]:
                closest_row = find_closest_row(im_prefix, period)
                if closest_row is None:
                    continue
                im_vals = df_IMs.loc[closest_row].values
                scaled_vals = sf_vals * im_vals
                scaled_vals = scaled_vals[~np.isnan(scaled_vals)]
                key = f"{im_prefix.rstrip('_')}_{label}"
                if len(scaled_vals) > 0:
                    log_std_row[key] = np.std(np.log(scaled_vals))
                    cov_row[key] = np.std(scaled_vals) / np.mean(scaled_vals)
                    scaled_vals_dict[(model_id, key)] = scaled_vals
                else:
                    log_std_row[key] = np.nan
                    cov_row[key] = np.nan
        else:
            # Non-period-dependent IMs
            im_vals = df_IMs.loc[im].values
            scaled_vals = sf_vals * im_vals
            scaled_vals = scaled_vals[~np.isnan(scaled_vals)]
            key = im
            if len(scaled_vals) > 0:
                log_std_row[key] = np.std(np.log(scaled_vals))
                cov_row[key] = np.std(scaled_vals) / np.mean(scaled_vals)
                scaled_vals_dict[(model_id, key)] = scaled_vals
            else:
                log_std_row[key] = np.nan
                cov_row[key] = np.nan

    log_std_results[model_id] = log_std_row
    cov_results[model_id] = cov_row


# --- Save results -----------------------------------------------------------
output_file = os.path.join(base_dir, "log_std_and_cov_scaled_IMs.xlsx")

df_log_std = pd.DataFrame(log_std_results).T.sort_index()
df_cov = pd.DataFrame(cov_results).T.sort_index()

with pd.ExcelWriter(output_file) as writer:
    df_log_std.to_excel(writer, sheet_name="log_std")
    df_cov.to_excel(writer, sheet_name="cov")

print(f"Results saved to:\n{output_file}")