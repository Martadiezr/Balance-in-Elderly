import pandas as pd
import os, re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_sto(path):
    # primero averiguamos cuántas líneas tiene el header
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == 'endheader':
                header_end = i
                break
    # cargamos a partir de la línea siguiente al "endheader"
    df = pd.read_csv(path,
                     sep='\t',
                     skiprows=header_end+1,
                     engine='python')
    return df

paths = [
    '/Users/marti/Documents/SCONE/results/250416.184928.f0916m.R32.BW.D30.R3/0225_2.219_1.951.par.sto',
    '/Users/marti/Documents/SCONE/results/250416.194628.modelo2.R32.BW.D30.R2/0388_94.535_2.214.par.sto',
    '/Users/marti/Documents/SCONE/results/250417.114710.modelo3.R32.BW.D30.R3/0162_47.247_1.430.par.sto'
]

# cargar todos en un dict de DataFrames
dfs = {Path(p).name: load_sto(p) for p in paths}


variables_balance = [
    # CoM
    'com_x','com_y','com_z','com_x_dot','com_y_dot',
    # CoP
    'cop_x_r','cop_y_r','cop_x_l','cop_y_l',
    # GRF
    'grf_vz_r','grf_vx_r','grf_vy_r',
    'grf_vz_l','grf_vx_l','grf_vy_l',
    # Ángulos
    'pelvis_tilt','pelvis_list','pelvis_rotation',
    'hip_flexion_r','knee_angle_r','ankle_angle_r',
    # Momentos
    'moment_hip_flexion_r','moment_knee_flexion_r','moment_ankle_flexion_r',
    # Ejemplo muscular (si existen)
    'force_tib_ant_r','force_gas_med_r'
]

# Extraer para cada modelo
balance_dfs = {
    name: df[ [c for c in variables_balance if c in df.columns] ]
    for name, df in dfs.items()
}

summary_list = []
for name, df in balance_dfs.items():
    stats = df.describe().T[['mean', 'std', 'min', 'max']].reset_index().rename(columns={'index':'variable'})
    stats['model'] = name
    summary_list.append(stats)

summary_df = pd.concat(summary_list, ignore_index=True)


for model in summary_df['model'].unique():
    print(f"\n=== Modelo: {model} ===")
    sub = summary_df[summary_df['model'] == model]
    print(sub[['variable','mean','std','min','max']].to_string(index=False))





# Variables de CoM
com_vars = ['com_x','com_y','com_z']

# Variables de fuerzas de contacto (GRF) – izquierda y derecha
grf_vars_l = ['leg0_l.grf_x','leg0_l.grf_y','leg0_l.grf_z']
grf_vars_r = ['leg1_r.grf_x','leg1_r.grf_y','leg1_r.grf_z']

# Para acumular resultados
com_summary   = []
force_summary = []

for model_name, df in dfs.items():
    # 1) Centro de Masa
    available_com = [c for c in com_vars if c in df.columns]
    stats_com = df[available_com].agg(['mean','std','min','max']).T
    stats_com['model'] = model_name
    stats_com = stats_com.reset_index().rename(columns={'index':'variable'})
    com_summary.append(stats_com)
    
    # 2) Fuerzas de contacto
    #   a) GRF izquierda
    avail_l = [c for c in grf_vars_l if c in df.columns]
    fl = df[avail_l].copy()
    fl['res_l'] = np.sqrt((fl**2).sum(axis=1))
    #   b) GRF derecha
    avail_r = [c for c in grf_vars_r if c in df.columns]
    fr = df[avail_r].copy()
    fr['res_r'] = np.sqrt((fr**2).sum(axis=1))
    
    #   c) Estadísticos
    if 'res_l' in fl:
        force_summary.append({
            'model': model_name,
            'side': 'left',
            'mean': fl['res_l'].mean(),
            'std':  fl['res_l'].std(),
            'min':  fl['res_l'].min(),
            'max':  fl['res_l'].max()
        })
    if 'res_r' in fr:
        force_summary.append({
            'model': model_name,
            'side': 'right',
            'mean': fr['res_r'].mean(),
            'std':  fr['res_r'].std(),
            'min':  fr['res_r'].min(),
            'max':  fr['res_r'].max()
        })

# Construir DataFrames de salida
com_df   = pd.concat(com_summary,   ignore_index=True)
force_df = pd.DataFrame(force_summary)

# Mostrar resultados
print("\n=== Resumen Centro de Masa ===")
print(com_df.to_string(index=False))

print("\n=== Resumen Fuerzas de Contacto (GRF) ===")
print(force_df.to_string(index=False))

#Así podrás comparar directamente qué modelo mantiene una carga de contacto más estable 
# (menor variabilidad y picos suaves) y evaluar cómo el envejecimiento simulado afecta la GRF total

# 4. Cálculo de fuerzas resultantes para cada pie
for name, df in dfs.items():
    df['res_l'] = np.sqrt(df['leg0_l.grf_x']**2 + df['leg0_l.grf_y']**2 + df['leg0_l.grf_z']**2)
    df['res_r'] = np.sqrt(df['leg1_r.grf_x']**2 + df['leg1_r.grf_y']**2 + df['leg1_r.grf_z']**2)

# 5. Graficar resultado comparativo
# 5a. Pie izquierdo
plt.figure()
for name, df in dfs.items():
    plt.plot(df['time'], df['res_l'], label=name)
plt.xlabel('Time (s)')
plt.ylabel('Resultant GRF Left (N)')
plt.title('Resultant Ground Reaction Force - Left Foot')
plt.legend()
plt.tight_layout()
plt.show()

# 5b. Pie derecho
plt.figure()
for name, df in dfs.items():
    plt.plot(df['time'], df['res_r'], label=name)
plt.xlabel('Time (s)')
plt.ylabel('Resultant GRF Right (N)')
plt.title('Resultant Ground Reaction Force - Right Foot')
plt.legend()
plt.tight_layout()
plt.show()

#plot centros de masa
for axis in ['com_x', 'com_y', 'com_z']:
    plt.figure()
    for name, df in dfs.items():
        plt.plot(df['time'], df[axis], label=name)
    plt.xlabel('Time (s)')
    plt.ylabel(f'{axis} (m)')
    plt.title(f'Center of Mass - {axis}')
    plt.legend()
    plt.tight_layout()
    plt.show()