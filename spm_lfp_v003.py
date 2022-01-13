### Sources: 
# https://honors.libraries.psu.edu/files/final_submissions/3057
# https://github.com/davidhowey/Spectral_li-ion_SPM/blob/master/source/derivs_spm.m 
# M. Safari and C. Delacourt 2011 J. Electrochem. Soc. 158 A562
# M. Safari and C. Delacourt 2011 J. Electrochem. Soc. 158 A1123
###

import math
import pandas as pd
import numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import figure

t = 7000  #simulation time. s

F= 9.64853399e4  # Faraday number C/mol 
R_gas = 8.314 # [J / mol K] - Gas constant 
T = 293 # [K] - ambient temperature 
alpha = 0.5 # Symmetrie factor 
n = 1 # [# of electrons] - number of electrons transfered
Cmax_cath = 22800 # mol/m^3
Cmax_anode = 31370 # mol/m^3
R_cath= 5e-8 # 50 nm 
R_anode = 3.5e-6 # 3 micro meter
D_cath = 1e-17 # Solid diffusivity of the cathode, m^2*s^-1
D_anode = 2e-14
C_rate_inv = 1.05
I=2.5/C_rate_inv # current for 2.5 Ah cell 
R_int = 0.020 # mOhm


N = 8 # number of radial segments
dr_cath = R_cath/N # radial segment length, m
dr_anode = R_anode/N

dt = dr_cath**2/(2*D_cath)*0.9
dt_anode = dr_anode**2/(2*D_anode)*0.9
numofsteps = int(t/dt)
numofsteps_anode = int(t/dt_anode)
timehis = np.linspace(0,t,num=numofsteps)
timehis_anode = np.linspace(0,t, num = numofsteps_anode)

I_dyn = np.linspace(I , I , numofsteps)
I_dyn[int(numofsteps/2):int(3*numofsteps/4)] = (-I/2)
I_dyn[int(3*numofsteps/4):(numofsteps)] = 0

# Cathode ---------------------------------------------------------
coul_cath = 2.5*3600 #A123 2.5Ah
mol_cath = coul_cath/F
vol_cath = mol_cath/Cmax_cath
partvol_cath = (4/3)*math.pi*(R_cath**3)
numpart_cath = vol_cath/partvol_cath
A_cath = numpart_cath*4*math.pi*(R_cath**2)
k_cath = -4.3e-16*I**2 + 2e-14*np.abs(I) + 1.1e-14 # Safari et al. / reaction rate cathode

# Anode ---------------------------------------------------------
vol_anode = mol_cath/Cmax_anode
partvol_anode = (4/3)*math.pi*(R_anode**3)
numpart_anode = vol_anode/partvol_anode
A_anode = numpart_anode*4*math.pi*(R_anode**2)
k_anode = 8.2e-12 # reaction rate anode 

r_cath = np.zeros(N)
r_anode = np.zeros(N)

# Initial conditions
c_init_cath = np.linspace(0 , Cmax_cath , 50)
c_init_anode = np.linspace(0 , Cmax_anode , 50)

# Cathode ---------------------------------------------------------
deltaC_cath = np.zeros([N,numofsteps])
c2r2 = np.zeros([N,numofsteps])
c1r1 = np.zeros([N,numofsteps])

# Anode ---------------------------------------------------------
deltaC_anode = np.zeros([N,numofsteps])
a2r2 = np.zeros([N,numofsteps])
a1r1 = np.zeros([N,numofsteps])

i_area_c = I / A_cath # A / m2
i_area_a = I / A_anode  # A / m2
i_eta_cath = i_area_c /(A_cath * vol_cath) #i_area_c / 2*R_cath / (4*math.pi*(R_cath**2))
i_eta_anode = i_area_a /(A_anode * vol_anode)

for k in range(N): 
    m=k+1
    r_cath[k] = m*dr_cath 
    r_anode[k] = m*dr_anode      


# Cathode ---------------------------------------------------------
C_cath = np.zeros([N+1,numofsteps])
U_cath = np.zeros([numofsteps])
i_0_cath = np.zeros([numofsteps])
eta_cath = np.zeros([numofsteps])

# Anode ---------------------------------------------------------
C_anode = np.zeros([N+1, numofsteps])
U_anode = np.zeros([numofsteps])
i_0_anode = np.zeros([numofsteps])
eta_anode = np.zeros([numofsteps])

U_cell = np.zeros([numofsteps])


for i in range(N+1):
    for k in range(numofsteps):
        C_cath[i,0]=c_init_cath[45] 
        C_anode[i,0] = c_init_anode[0]

dr2_cath = dr_cath*dr_cath
dr2_anode = dr_anode*dr_anode

# Cathode ---------------------------------------------------------
for j in range(numofsteps-1): 
    for i in range(1,N):
        r2 = r_cath[i]**2
               
        c2r2[i,j] = (C_cath[i+1,j] - 2*C_cath[i,j] + C_cath[i-1,j]) / dr2_cath
        c1r1[i,j] = (C_cath[i+1,j] - C_cath[i-1,j]) / dr_cath
        deltaC_cath[i,j] = (D_cath/r2)*(r2 * c2r2[i,j] + r_cath[i]*c1r1[i,j])
        C_cath[i,j+1] = C_cath[i,j] + (deltaC_cath[i,j]*dt)
        
    C_cath[0,j+1] = C_cath[1,j+1]
    C_cath[N,j+1] = (dr_cath*(-I_dyn[j])/(D_cath*A_cath*F)) + C_cath[N-1,j+1]
    
    U_cath[j] = 3.4323 - 0.8428*np.exp(-80.2493*(1-(C_cath[i,j]/Cmax_cath))**1.3198)-3.247e-6*np.exp(20.2645*(1-(C_cath[i,j]/Cmax_cath))**3.8003)+3.2482e-6*np.exp(20.2646*(1-(C_cath[i,j]/Cmax_cath))**3.7995)
    
    # Exchange current density from Butler-Volmer 
    i_0_cath[j] = k_cath*F*C_cath[N,j]**alpha*(Cmax_cath - C_cath[N,j])**alpha
    
    # Overpotential at cathode
    eta_cath[j] = ((2*R_gas*T)/(n*F))*np.arcsinh(((-alpha*i_eta_cath))/i_0_cath[j])
    
    print("Cathode ", round(j/numofsteps,2))
cathode_plot = figure(1)
concentration_cath = C_cath[N,0:-1]
#pyplot.plot(concentration_cath[0:-1] / Cmax_cath, U_cath[0:-2])
#pyplot.plot(timehis[0:-2], U_cath[0:-2])

# Anode ---------------------------------------------------------
for j in range(numofsteps-1): 
    for i in range(1,N):
        r2 = r_anode[i]**2
               
        a2r2[i,j] = (C_anode[i+1,j] - 2*C_anode[i,j] + C_anode[i-1,j]) / dr2_anode
        a1r1[i,j] = (C_anode[i+1,j] - C_anode[i-1,j]) / dr_anode
        deltaC_anode[i,j] = (D_anode/r2)*(r2 * a2r2[i,j] + r_anode[i]*a1r1[i,j])
        C_anode[i,j+1] = C_anode[i,j] + (deltaC_anode[i,j]*dt)
        
    C_anode[0,j+1] = C_anode[1,j+1]
    C_anode[N,j+1] = (dr_anode*(I_dyn[j])/(D_anode*A_anode*F)) + C_anode[N-1,j+1]
    
    U_anode[j] = 0.6379 + 0.5416*np.exp(-305.5309*(C_anode[i,j]/Cmax_anode))+0.044*np.tanh((-(C_anode[i,j]/Cmax_anode)-0.1958)/0.1088)-0.1978*np.tanh(((C_anode[i,j]/Cmax_anode)-1.0571)/0.0854)-0.6875*np.tanh(((C_anode[i,j]/Cmax_anode)+0.0117)/0.0529)-0.0175*np.tanh(((C_anode[i,j]/Cmax_anode)-0.5692)/0.0875)
    
    i_0_anode[j] = k_anode * F * C_anode[N,j]**alpha * (Cmax_anode - C_anode[N,j])**alpha
    
    # Overpotential at anode
    eta_anode[j] = ((2*R_gas*T)/(n*F))*np.arcsinh(((-alpha*i_eta_anode))/i_0_anode[j])
    
    print("Anode ", round(j/numofsteps,2))    
anode_plot = figure(2)
concentration_anode = C_anode[N,0:-1]
#pyplot.plot(concentration_anode[0:-1] / Cmax_anode, U_anode[0:-2])
#pyplot.plot(timehis[0:-2], U_anode[0:-2])


# Full Cell ---------------------------------------------------------
U_cell = U_cath - U_anode + R_int*I_dyn
cell_plot = figure(3)
pyplot.plot(timehis[0:-2] , U_cell[0:-2])
pyplot.ylim(3.2, 3.5)


