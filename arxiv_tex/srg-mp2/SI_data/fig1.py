#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import PathCollection
import pandas as pd
import os

cwd = os.getcwd()
# Reduce horizontal space between boxplots within each subplot by adjusting width and dodge
boxplot_kwargs = dict(width=0.3, dodge=False, fliersize=3)

# df = pd.read_csv('values.csv')
# df = pd.read_csv('values.csv')
# df2= pd.read_csv('SRG-QP-GF2-0.525.csv')
# df3= pd.read_csv('SRG-QP-GF2-1.4-0.0-1.0.csv')
# df4= pd.read_csv('SRG-QP-GF2-0.7-0.6-1.0.csv')

# # Prepare data for IP and EA
# ip_data = [
#     df['mol'],
#     -df2['err_homo'],
#     -df3['err_homo'],
#     -df4['err_homo'],
#     -df['SRG_qsGW_IP'] + df['ccsd(t)_IP']
# ]
# # Create a DataFrame with explicit column names for IP data
# ip_df = pd.DataFrame(
#     np.array(ip_data).T,
#     columns=["mol",'SRG-qsGF2', 'SRG-SOS-qsGF2', 'SRG-SCS-qsGF2', 'SRG-qsGW']
# )
# ip_data = [ip_df[col] for col in ip_df.drop(columns=["mol"]).columns]  # keep ip_data as list for plotting

# ip_labels = ['SRG-qsGF2', 'SRG-SOS-qsGF2','SRG-SCS-qsGF2', 'SRG-qs$GW$']

# ea_data = [
#     df['mol'],
#     -df2['err_lumo'],
#     -df3['err_lumo'],
#     -df4['err_lumo'],
#     -df['SRG_qsGW_EA'] + df['ccsd(t)_EA']
# ]
# # Create a DataFrame with explicit column names for EA data
# ea_df = pd.DataFrame(
#     np.array(ea_data).T,
#     columns=['mol','SRG-qsGF2', 'SRG-SOS-qsGF2', 'SRG-SCS-qsGF2', 'SRG-qsGW']
# )
# ea_data = [ea_df[col] for col in ea_df.drop(columns=["mol"]).columns]  # keep ea_data as list for plotting
# ea_labels=ip_labels

# Read the CSV, skipping comment lines
df = pd.read_csv("Fig_I.csv", comment='#', header=[0, 1], index_col=0)

# Now df has a MultiIndex for columns: (method, property)
# For example: ('SRG-qsGF2', 'IP'), ('SRG-qsGF2', 'EA'), ...

# Prepare IP and EA data
ip_df = df.xs('IP', axis=1, level=1)
ea_df = df.xs('EA', axis=1, level=1)

# If you want lists for plotting (like before):
ip_data = [ip_df[col]-ip_df['ccsd(t)'] for col in ip_df.drop(columns=['ccsd(t)']).columns]
ea_data = [ea_df[col]-ea_df['ccsd(t)'] for col in ea_df.drop(columns=['ccsd(t)']).columns]

# If you want labels:
ip_labels = list(ip_df.columns)
ea_labels = list(ea_df.columns)

# Example: print or use for plotting
print(ip_data)
print(ea_data)

#%% Violin plots

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

# Add horizontal line at y=0 for both subplots
for ax in axs:
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
# Plot IP violinplots
print("IP Data:", ip_data)
bp_ip = sns.violinplot(data=pd.DataFrame(ip_data).T, ax=axs[0], palette=["skyblue", "green", "purple","gold"], width=0.5, saturation=0.4)
for patch in bp_ip.artists:
    patch.set_alpha(0.6)
for artist in axs[0].lines:
    artist.set_zorder(12)
for artist in axs[0].findobj(PathCollection):
    artist.set_zorder(10)
sns.stripplot(data=pd.DataFrame(ip_data).T, ax=axs[0], color='k', size=3.5, palette=["#001AA1", "#B3990B", "#ff8800", "#222222"], jitter=True, alpha=0.8)
axs[0].set_title("a)")
axs[0].set_xticklabels(ip_labels, rotation=40)
axs[0].set_ylabel('Error in eV', labelpad=2)
axs[0].set_ylim(-1.3, 1.5)
axs[0].yaxis.set_major_locator(MaxNLocator(integer=False, steps=[1, 5]))

# Plot EA violinplots
bp_ea = sns.violinplot(data=pd.DataFrame(ea_data).T, ax=axs[1], palette=["skyblue", "green", "purple","gold"], width=0.5, saturation=0.6, cut=1)
for patch in bp_ea.artists:
    patch.set_alpha(0.6)
sns.stripplot(data=pd.DataFrame(ea_data).T, ax=axs[1], color='k', size=3, palette=["#001AA1", "#B3990B", "#ff8800", "#222222"], jitter=True, alpha=0.8)
axs[1].set_title("b)")
axs[1].set_xticklabels(ea_labels, rotation=40)
axs[1].set_ylabel('Error in eV', labelpad=2)
axs[1].set_ylim(-1.3, 1.5)
axs[1].yaxis.set_major_locator(MaxNLocator(integer=False,steps=[1, 5]))

fig.subplots_adjust(wspace=0.2)
plt.tight_layout()
plt.savefig(f"{cwd}/srgGW50_violinplots.png", transparent=True, dpi=600, bbox_inches='tight')
plt.show()
