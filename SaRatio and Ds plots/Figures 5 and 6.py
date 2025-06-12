import numpy as np
import matplotlib.pyplot as plt
import scienceplots



cm = 1/2.54

RotD50_all                     = np.genfromtxt('RotD50.txt') # RotD50 of GMs used for IDA
PSA_x_all                      = np.genfromtxt('PSA_x.txt') # RotD50 of GMs used for IDA
PSA_y_all                      = np.genfromtxt('PSA_y.txt') # RotD50 of GMs used for IDA
SaR_x_all                      = np.genfromtxt('SaR_x.txt') # RotD50 of GMs used for IDA
SaR_y_all                      = np.genfromtxt('SaR_y.txt') # RotD50 of GMs used for IDA
 
 
GMno1 = 1 # Gm no 1 in the plot
GMno2 = 6 # Gm no 2 in the plot
GMno3 = 41 # Gm no 3 in the plot

period = 1 # period of interest 

fig, ax = plt.subplots(figsize=(10 * cm,7 * cm), dpi=400) 
plt.xscale('log') # log x scale
plt.yscale('log') # log y scale          
PSA_1 = PSA_x_all[:,GMno1]*1/PSA_x_all[int(period*100-1),GMno1] # scaling GM1 at period to have a val of 1g
PSA_2 = PSA_x_all[:,GMno2]*1/PSA_x_all[int(period*100-1),GMno2] # scaling GM2 at period to have a val of 1g
PSA_3 = PSA_x_all[:,GMno3]*1/PSA_x_all[int(period*100-1),GMno3] # scaling GM3 at period to have a val of 1g

'plotting the GMs'

plt.plot(PSA_x_all[:,0],PSA_1,color='k',label='$S_aRatio$ = '+str(round(np.array(SaR_x_all[np.argmax(SaR_x_all[:,0] >= period),:])[GMno1],1)))
plt.plot(PSA_y_all[:,0],PSA_2,color='k',alpha=0.55,linestyle=':',label='$S_aRatio$ = '+str(round(np.array(SaR_x_all[np.argmax(SaR_x_all[:,0] >= period),:])[GMno2],1)))
# plt.plot(PSA_y_all[:,0],PSA_3,color='k',alpha=0.3,linestyle='--',label='$S_aRatio$ = '+str(round(np.array(SaR_x_all[np.argmax(SaR_x_all[:,0] >= period),:])[GMno3],2)))

'plot formatting'         

ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'])
ax.set_yticks([0.01, 0.02, 0.05, 0.2, 0.5, 1, 2, 5])
ax.set_yticklabels(['0.01', '0.02', '0.05', '0.2', '0.5', '1', '2', '5']) 
plt.xticks()
plt.yticks()    
plt.grid(False)  
plt.legend()                   
plt.tight_layout()
ax.set_xlabel('T (s)')
ax.set_ylabel('$S_{a}$ (T, $\zeta$=5\%)  (g)')
ax.axvline(x=period, color='black', linewidth=1.0)
plt.xlim(PSA_x_all[:,0][4],5)
plt.tight_layout()
plt.grid(True, linestyle='-', alpha=0.1)
plt.savefig('SaRatio eg.pdf', dpi=1000, bbox_inches='tight')
plt.show()
# plt.ylim(0.005,np.max(RotD50_all[:,1:])*3)






fig, ax = plt.subplots(figsize=(10 * cm,7 * cm), dpi=400) 
    
PSA_1 = PSA_x_all[:,GMno1]*1/PSA_x_all[int(period*100-1),GMno1] # scaling GM1 at period to have a val of 1g
# PSA_2 = PSA_x_all[:,GMno2]*1/PSA_x_all[int(period*100-1),GMno2] # scaling GM2 at period to have a val of 1g
PSA_3 = PSA_x_all[:,GMno3]*1/PSA_x_all[int(period*100-1),GMno3] # scaling GM3 at period to have a val of 1g

'plotting the GMs'

plt.plot(PSA_x_all[:,0],PSA_1,color='k',label='$S_aRatio$ = '+str(round(np.array(SaR_x_all[np.argmax(SaR_x_all[:,0] >= period),:])[GMno1],1)))
# plt.plot(PSA_y_all[:,0],PSA_2,color='k',alpha=0.55,linestyle=':',label='$S_aRatio$ = '+str(round(np.array(SaR_x_all[np.argmax(SaR_x_all[:,0] >= period),:])[GMno2],1)))
plt.plot(PSA_y_all[:,0],PSA_3,color='k',alpha=0.3,linestyle='--',label='$S_aRatio$ = '+str(round(np.array(SaR_x_all[np.argmax(SaR_x_all[:,0] >= period),:])[GMno3],2)))

'plot formatting'         

plt.xlim(0, 3)
plt.ylim(0, 2)
plt.yticks()    
plt.grid(False)  
plt.legend()                   
plt.tight_layout()
ax.set_xlabel('T (s)')
ax.set_ylabel('$S_{a}$ (T, $\zeta$=5\%)  (g)')
ax.axvline(x=period, color='black', linewidth=1.0)
# plt.xlim(PSA_x_all[:,0][4],5)
plt.tight_layout()
plt.grid(True, linestyle='-', alpha=0.1)
plt.savefig('SaRatio eg Sa.pdf', dpi=1000, bbox_inches='tight')
plt.show()



fig, ax = plt.subplots(figsize=(10 * cm, 7 * cm), dpi=400)

# Convert spectral acceleration (g) to spectral displacement (m) using Sd = Sa * T^2 / (4π^2)
T_vals = PSA_x_all[:, 0]  # Assuming this is the period vector

# Spectral acceleration scaling to 1g at target period
PSA_1 = PSA_x_all[:, GMno1] * 1 / PSA_x_all[int(period * 100 - 1), GMno1]
PSA_3 = PSA_x_all[:, GMno3] * 1 / PSA_x_all[int(period * 100 - 1), GMno3]

# Convert to spectral displacement
Sd_1 = PSA_1 * 9.81 * T_vals**2 / (4 * np.pi**2)
Sd_3 = PSA_3 * 9.81 * T_vals**2 / (4 * np.pi**2)

# Plotting the GMs
plt.plot(T_vals, Sd_1, color='k',
         label='$S_aRatio$ = ' + str(round(SaR_x_all[np.argmax(SaR_x_all[:, 0] >= period), GMno1], 1)))
plt.plot(PSA_y_all[:, 0], Sd_3, color='k', alpha=0.3, linestyle='--',
         label='$S_aRatio$ = ' + str(round(SaR_x_all[np.argmax(SaR_x_all[:, 0] >= period), GMno3], 2)))

# Plot formatting         
plt.xlim(0, 3)
plt.ylim(0, 2.5)  # You may adjust upper limit depending on your Sd values
plt.yticks()
plt.grid(True, linestyle='-', alpha=0.1)
plt.legend()
plt.tight_layout()
ax.set_xlabel('T (s)')
ax.set_ylabel('$S_d$ (T, $\zeta$=5\%)  (m)')
ax.axvline(x=period, color='black', linewidth=1.0)
plt.savefig('SaRatio eg Sd.pdf', dpi=1000, bbox_inches='tight')
plt.show()



# Find indices where T is between 0.2 and 3.0 s
mask = (T_vals >= 0.2) & (T_vals <= 3.0)

# Extract corresponding Sd values
Sd_1_selected = Sd_1[mask]
Sd_3_selected = Sd_3[mask]

# Calculate geometric mean: exp(mean(log(Sd)))
# Avoid log(0) by filtering out any zero or negative values (if present)
Sd_1_selected = Sd_1_selected[Sd_1_selected > 0]
Sd_3_selected = Sd_3_selected[Sd_3_selected > 0]

geo_mean_Sd_1 = np.exp(np.mean(np.log(Sd_1_selected)))
geo_mean_Sd_3 = np.exp(np.mean(np.log(Sd_3_selected)))

print(f"Geometric mean of Sd (GM1) from 0.2s to 3s: {geo_mean_Sd_1:.4f} m")
print(f"Geometric mean of Sd (GM3) from 0.2s to 3s: {geo_mean_Sd_3:.4f} m")






# # Load acceleration data and define time step
# acc_data = np.loadtxt('gacc_6_x.txt')  # Replace with your actual file name
# dt = 0.005  # time step in seconds
# time = np.arange(len(acc_data)) * dt

# # Plot acceleration time history
# fig, ax = plt.subplots(figsize=(10 * cm, 5 * cm), dpi=400)
# ax.plot(time, acc_data, color='k', lw = 0.5)
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Acceleration (g)')
# ax.set_ylim(-0.4, 0.4)
# ax.set_xlim(0, 40)
# plt.grid(True, linestyle='-', alpha=0.1)
# # ax.set_title('Acceleration Time History - GMno2')
# plt.tight_layout()
# # plt.savefig('GMno2_acc_history.pdf', dpi=1000, bbox_inches='tight')
# plt.show()


# # Compute Arias Intensity
# g = 9.81  # gravity acceleration in m/s^2
# arias_intensity = np.cumsum(acc_data**2) * (np.pi / (2 * g)) * dt
# arias_total = arias_intensity[-1]

# # Find time when AI reaches 5% and 75%
# i5 = np.where(arias_intensity >= 0.05 * arias_total)[0][0]
# i75 = np.where(arias_intensity >= 0.75 * arias_total)[0][0]
# t5 = time[i5]
# t75 = time[i75]
# Ds_5_75 = t75 - t5

# # Print significant duration
# print(f"Significant Duration (Ds,5–75) for GMno2: {Ds_5_75:.2f} s")

# # Plot cumulative AI with dots and annotations
# fig, ax = plt.subplots(figsize=(10 * cm, 5 * cm), dpi=400)
# ax.plot(time, arias_intensity / arias_total, color='black')

# # Add points for 5% and 75%
# ax.plot(t5, arias_intensity[i5] / arias_total, 'ko', ms = 3)  # red dot
# ax.plot(t75, arias_intensity[i75] / arias_total, 'ko', ms = 3)  # blue dot

# # Add text next to the points
# ax.text(t5 + 1, arias_intensity[i5] / arias_total + 0.01, f'$t_5$ = {t5:.1f}s', color='k', fontsize=9, ha='left')
# ax.text(t75 + 1, arias_intensity[i75] / arias_total + 0.01, f'$t_{{75}}$ = {t75:.1f}s', color='k', fontsize=9, ha='left')

# # Add textbox with Ds,5-75
# props = dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none')
# ax.text(0.68, 0.9, f'$D_{{s,5-75}}$ = {Ds_5_75:.1f} s',
#         transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', bbox=props)

# # Labels and limits
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Cumulative $I_A$')
# ax.set_ylim(0, 1)
# ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
# ax.set_xlim(0, 40)
# plt.grid(True, linestyle='-', alpha=0.1)

# # ax.legend()
# plt.tight_layout()
# # plt.savefig('GMno2_arias_duration.pdf', dpi=1000, bbox_inches='tight')
# plt.show()




# Load acceleration data and define time step
acc_data = np.loadtxt('gacc_6_x.txt')  # Replace with your actual file name
dt = 0.005  # time step in seconds
time = np.arange(len(acc_data)) * dt

# Compute Arias Intensity
g = 9.81  # gravity acceleration in m/s^2
arias_intensity = np.cumsum(acc_data**2) * (np.pi / (2 * g)) * dt
arias_total = arias_intensity[-1]

# Find time when AI reaches 5% and 75%
i5 = np.where(arias_intensity >= 0.05 * arias_total)[0][0]
i75 = np.where(arias_intensity >= 0.75 * arias_total)[0][0]
t5 = time[i5]
t75 = time[i75]
Ds_5_75 = t75 - t5

# Print significant duration
print(f"Significant Duration (Ds,5–75) for GMno2: {Ds_5_75:.2f} s")

# Create stacked subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10 * cm, 8 * cm), dpi=400, sharex=True, constrained_layout=True)

# Plot 1: Acceleration time history
ax1.plot(time, acc_data, color='k', lw=0.5)
ax1.set_ylabel('Acceleration (g)')
ax1.set_ylim(-0.4, 0.4)
ax1.set_xlim(0, 40)
ax1.grid(True, linestyle='-', alpha=0.1)
ax1.tick_params(axis='y', pad=3)

# Plot 2: Normalized Arias Intensity with markers and annotations
ax2.plot(time, arias_intensity / arias_total, color='black')
ax2.plot(t5, arias_intensity[i5] / arias_total, 'ko', ms=3)
ax2.plot(t75, arias_intensity[i75] / arias_total, 'ko', ms=3)

# Add time annotations
ax2.text(t5 + 1, arias_intensity[i5] / arias_total + 0.01, f'$t_5$ = {t5:.1f}s', fontsize=9, ha='left')
ax2.text(t75 + 1, arias_intensity[i75] / arias_total + 0.01, f'$t_{{75}}$ = {t75:.1f}s', fontsize=9, ha='left')

# Add textbox with Ds,5-75
props = dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none')
ax2.text(0.68, 0.9, f'$D_{{s,5-75}}$ = {Ds_5_75:.1f} s',
          transform=ax2.transAxes, fontsize=10,
          verticalalignment='top', bbox=props)

# Labels and limits
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Cumulative $I_A$')
ax2.set_ylim(0, 1)
ax1.set_xlim(0, 40)
ax2.set_yticks(np.linspace(0, 1, 6))
ax2.grid(True, linestyle='-', alpha=0.1)

# Export (optional)
plt.savefig('Ds5_75 eg.pdf', dpi=1000)
plt.show()