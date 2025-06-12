# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:08:17 2025

@author: ljp70

Compute a suite of earthquake Intensity Measures for 50 GMs and
append them — now including Sa(T1) and SaAvg — to 'Intensity_Measures.xlsx'
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt, detrend
import eqsig


# --- 1) set up ----------------------------------------------------------------
base_dir   = r"D:\HC_IDA_GMs"
sa_file    = os.path.join(base_dir, "response_spectra_sa.xlsx")

# load SA & compute SV, SD as before
df_sa   = pd.read_excel(sa_file, index_col=0)            # SA [g]
periods = df_sa.index.to_numpy()                        # T [s]
omega   = np.where(periods != 0, 2 * np.pi / periods, np.nan)
df_sv   = df_sa.div(omega, axis=0).fillna(0) * 9.81      # SV [m/s]
df_sd   = df_sa.div(omega**2, axis=0).fillna(0) * 9.81   # SD [m]

# --- 2) define all your IM labels --------------------------------------------
# (a) the “classic” ones:
IMs   = ['PGA','PGV','PGD','AI','CAV','ASI','VSI','DSI']
comps = ['x','y']
gm_cols = [f"GM{gm}_{c}" for gm in range(1,51) for c in comps]

# (b) the T1‐based ones:
SaT1_vals    = [
    0.5, 1.0, 2.0,
    1.3, 1.02, 0.72, 0.48, 2.10, 1.68, 1.26, 0.80,
    3.04, 2.38, 1.76, 1.26, 3.92, 3.30, 2.40, 1.62,
    1.38, 1.04, 0.78, 
    
    0.48, 2.30, 2.02, 1.08, 0.62,
    3.20, 2.44, 1.50, 0.86, 
    
    0.5, 0.66, 0.38,  
    0.56, 0.34, 0.82, 1.70, 1.68, 
    1.36, 0.84, 1.42, 0.78, 
    0.94, 0.68, 1.56, 1.46,
    
    0.76,  0.40, 0.30,
    1.22,  0.96, 0.46,
    1.70,  1.02, 
    2.24,  1.88,  1.36, 0.92,
    
    0.28, 0.46, 0.60, 0.80,
    0.36, 0.64, 1.20, 
    0.50, 0.90, 1.44, 1.88,
]


SaT1_labels  = [f"SaT1_{T1:.2f}"   for T1 in SaT1_vals]
SaAvg_labels = [f"SaAvg_{T1:.2f}"  for T1 in SaT1_vals]
Sa1p5_labels = [f"PFA_{T1:.2f}" for T1 in SaT1_vals]
FIV3_labels =  [f"FIV3_{T1:.2f}" for T1 in SaT1_vals]
FIV3_1Hz_labels =  [f"FIV3_1Hz_{T1:.2f}" for T1 in SaT1_vals]


all_IMs = IMs + SaT1_labels + SaAvg_labels + Sa1p5_labels + FIV3_labels + FIV3_1Hz_labels
df_im = pd.DataFrame(index=all_IMs, columns=gm_cols, dtype=float)

# --- 3) loop & fill ----------------------------------------------------------
for gm in range(1, 51):
    for c in comps:
        col = f"GM{gm}_{c}"

        # load dt & acc
        dt_arr   = np.loadtxt(os.path.join(base_dir, f"DT_{c}.txt"))
        dt       = dt_arr[gm-1]
        acc_file = os.path.join(base_dir, f"gacc_{gm}_{c}.txt")
        acc      = np.loadtxt(acc_file)*9.81  # [m/s²]

        # classic IMs
        # 1) detrend the acceleration
        acc_dt = detrend(acc)
        
        # 2) design a 4th‐order Butterworth bandpass (0.25–25 Hz)
        fs   = 1.0 / dt
        nyq  = 0.5 * fs
        low  = 0.25 / nyq
        
        if dt >= 0.02:
            high = 15 / nyq
        else:
            high = 25.0 / nyq
        
        b, a = butter(4, [low, high], btype='bandpass')
        
        # 3) apply zero‐phase filtering
        acc_bp = filtfilt(b, a, acc_dt)
        
        # 4) now compute PGA, velocity, displacement, PGV and PGD on the filtered trace
        PGA  = np.max(np.abs(acc_bp)) / 9.81
        vel  = cumtrapz(acc_bp, dx=dt, initial=0.0)
        disp = cumtrapz(vel,   dx=dt, initial=0.0)
        PGV  = np.max(np.abs(vel))
        PGD  = np.max(np.abs(disp))
        
        AI   = (np.pi / (2*9.81)) * np.sum(acc**2 * dt)
        CAV  = np.sum(np.abs(acc) * dt)
        
        mask = (periods >= 0.1) & (periods <= 0.5)
        ASI = np.trapz(df_sa[col].values[mask], periods[mask])
        
        mask = (periods >= 0.1) & (periods <= 2.5)
        VSI  = np.trapz(df_sv[col].values[mask], periods[mask])
        
        mask = (periods >= 2) & (periods <= 5)
        DSI  = np.trapz(df_sd[col].values[mask], periods[mask])
        
        periods_1p5 = [1.5 * T1 for T1 in SaT1_vals]
        
        rec = eqsig.AccSignal(acc_bp, dt)
        rec.generate_response_spectrum(response_times=periods_1p5, xi=0.707)
        sa_vals = rec.s_a / 9.81

        for T1, sa70 in zip(SaT1_vals, sa_vals):
            df_im.loc[f"PFA_{T1:.2f}", col] = sa70

        df_im.loc['PGA', col] = PGA
        df_im.loc['PGV', col] = PGV
        df_im.loc['PGD', col] = PGD
        df_im.loc['AI',  col] = AI
        df_im.loc['CAV', col] = CAV
        df_im.loc['ASI', col] = ASI
        df_im.loc['VSI', col] = VSI
        df_im.loc['DSI', col] = DSI

        # T1-based IMs
        for T1 in SaT1_vals:
            # 1) Sa(T1) via interpolation
            sa_t1 = np.interp(T1, periods, df_sa[col].values)
            df_im.loc[f"SaT1_{T1:.2f}", col] = sa_t1

            # 2) geometric mean of Sa over [0.2 T1, 3.0 T1]
            mask = (periods >= 0.2*T1) & (periods <= 3.0*T1)
            sa_band = df_sa[col].values[mask]
            # drop any zeros or negatives before log:
            sa_band = sa_band[sa_band > 0]
            sa_avg  = np.exp(np.mean(np.log(sa_band)))
            df_im.loc[f"SaAvg_{T1:.2f}", col] = sa_avg
            
        
        alpha = 0.7
        for T1 in SaT1_vals:
            fc = 0.85 / T1                      # cutoff frequency
            nyq = 0.5 / dt
            wn = fc / nyq                      # normalized cutoff for Butterworth

            # 2nd-order low-pass filter
            b_fiv3, a_fiv3 = butter(N=2, Wn=wn, btype='low')
            acc_filt = filtfilt(b_fiv3, a_fiv3, acc_dt)
        
            # window size in samples
            window_pts = int(round(alpha * T1 / dt))

            # Compute Vs(t)
            Vs_list = [
                np.trapz(acc_filt[i:i + window_pts], dx=dt)
                for i in range(len(acc_filt) - window_pts)
            ]
            Vs_arr = np.array(Vs_list)

            Vs_sorted = np.sort(Vs_arr)
            max3 = Vs_sorted[-3:]
            min3 = Vs_sorted[:3]
        
            FIV3_val = max(np.sum(max3), abs(np.sum(min3)))
            df_im.loc[f"FIV3_{T1:.2f}", col] = FIV3_val
            
        for T1 in SaT1_vals:
            fc = 1                      # cutoff frequency
            nyq = 0.5 / dt
            wn = fc / nyq                      # normalized cutoff for Butterworth

            # 2nd-order low-pass filter
            b_fiv3, a_fiv3 = butter(N=2, Wn=wn, btype='low')
            acc_filt = filtfilt(b_fiv3, a_fiv3, acc_dt)
        
            # window size in samples
            window_pts = int(round(alpha * T1 / dt))

            # Compute Vs(t)
            Vs_list = [
                np.trapz(acc_filt[i:i + window_pts], dx=dt)
                for i in range(len(acc_filt) - window_pts)
            ]
            Vs_arr = np.array(Vs_list)

            Vs_sorted = np.sort(Vs_arr)
            max3 = Vs_sorted[-3:]
            min3 = Vs_sorted[:3]
        
            FIV3_val = max(np.sum(max3), abs(np.sum(min3)))
            df_im.loc[f"FIV3_1Hz_{T1:.2f}", col] = FIV3_val

# --- 4) write out to Excel --------------------------------------------------
output_file = os.path.join(base_dir, "Intensity_Measures.xlsx")
df_im.to_excel(output_file)


print(f"✅ Wrote {len(all_IMs)} IMs × {len(gm_cols)} GMs to\n   {output_file}")
