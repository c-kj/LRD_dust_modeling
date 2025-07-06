import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os


import matplotlib.colors as mcolors
from LRD_IR_spectrum import *
u.set_enabled_equivalencies(u.spectral())

import matplotlib.patches as patches

ranges = [
    (0.5, 2, 'HST', 'lightblue', 0.2, 0.3),
    (0.623, 4.981, 'JWST/NIRCam', 'lightgreen', 0.57,0.8),
    (5.054, 26.733, 'JWST/MIRI', 'gold', 0.57,0.8),
    (4, 200, 'Spitzer', 'sandybrown', 0.2,0.3),
    (100, 700, 'Herschel', 'coral', 0.57,0.8),
    (400,1200, 'SCUBA2', 'tomato', 0.2,0.3),
    (800, 4000, 'ALMA', 'indianred', 0.57,0.8),
]



Orion_Opacity = OpacityData('/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/model/LRDChangeISMGrain/Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc')
Temple_SED = IncidentSED('/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/analysis/data/raw/Temple/quasar5_ly_EUV_new.template')


# restframe_wavelength = np.geomspace(1e2,3e4,40000) # AA
# L_bol = 1e46 # erg/s
# NormalizedFactorAt1450 = L_bol/4.4/2.0675e+15
# TempleWavelength_5_AA, TempleLlambda_5 = np.loadtxt('/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/analysis/data/raw/Temple/quasar5_ly_EUV.template', comments='#', delimiter=' ', usecols=(0, 1), unpack=True)
# TempleLlambda_5_new = LogLogInterpolator(TempleWavelength_5_AA, TempleLlambda_5)(restframe_wavelength)
# np.savetxt('/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/analysis/data/raw/Temple/quasar5_ly_EUV_new.template', np.column_stack((restframe_wavelength, TempleLlambda_5_new)), fmt='%1.18e', delimiter=' ')
# TempleWavelength_5_AA_new = restframe_wavelength
# TempleFrequency_5 = 3e10/(TempleWavelength_5_AA_new*1e-8)
# TempleLnu_5 = TempleLlambda_5_new*TempleWavelength_5_AA_new**2/3e18
# TempleLnu_5_normalized = TempleLnu_5/LogLogInterpolator(TempleWavelength_5_AA_new, TempleLnu_5)(1450)*NormalizedFactorAt1450
# L_temple = -1*np.trapz(TempleLnu_5_normalized, TempleFrequency_5)
# print(L_temple)
# plt.loglog(TempleWavelength_5_AA_new, TempleLnu_5_normalized*TempleFrequency_5)
# plt.show()


# model_B87 = B87Model(n_0=10, gamma=0,L_UV=1e46,T_sub=1500,NH_target=7.5e22,opacity=Orion_Opacity)
# model_Orion = OrionLRDModel(n_0=10, gamma=0,L_UV=1e46,T_sub=1500,NH_target=7.5e22,opacity=Orion_Opacity,incident_SED=Temple_SED)
# RestframeWavelength = np.geomspace(1,1e3,10000)* u.micron # micron
# Lnu_B87 = model_B87.calc_L_nu(RestframeWavelength)
# Lnu_Orion = model_Orion.calc_L_nu(RestframeWavelength)
# L_B87 = -1*np.trapz(Lnu_B87, RestframeWavelength.to(u.Hz))
# L_Orion = -1*np.trapz(Lnu_Orion, RestframeWavelength.to(u.Hz))
# plt.loglog(RestframeWavelength.to(u.micron), Lnu_B87/L_B87*1e46, label='B87, n0=10, gamma=0')
# plt.loglog(RestframeWavelength.to(u.micron), Lnu_Orion/L_Orion*1e46, label='Orion, n0=10, gamma=0')
# plt.legend()
# plt.show()

# import Orion opacity data
CLOUDY_Path = '/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/model/LRDChangeISMGrain'

OrionFile = 'Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc'

OrionOpacityFile = os.path.join(CLOUDY_Path,OrionFile)
OrionWavelength_Rydberg , Orion_Opacity_SigmaPerH , Orion_Opacity_SigmaPerH_abs = np.loadtxt(OrionOpacityFile,comments='#', delimiter='\t', usecols=(0, 1 ,2),unpack=True)

OrionWavelength_micro = 911.27/OrionWavelength_Rydberg*1e-4 # micron
OrionWavelength_x = 1/OrionWavelength_micro # rising order


TempleWavelength_5_AA, TempleLlambda_5 = np.loadtxt('analysis/data/raw/Temple/quasar5_ly_EUV_new.template',comments='#',delimiter=' ',usecols=(0,1),unpack=True)
TempleFrequency_5 = 3e10/(TempleWavelength_5_AA*1e-8)
TempleLnu_5 = TempleLlambda_5*TempleWavelength_5_AA**2/3e18
TempleLnu_5_attenuated = TempleLnu_5*np.exp(np.array([-np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH) for nu in TempleFrequency_5]) * 7.5e22)
L_temple = -1* np.trapz(TempleLnu_5,TempleFrequency_5)
L_temple_attenuated = -1*np.trapz(TempleLnu_5_attenuated,TempleFrequency_5)

arg_4000AA_Temple5 = np.argmin(np.abs(TempleWavelength_5_AA-4000))
L_temple_4000AA = TempleLnu_5_attenuated[arg_4000AA_Temple5]

Dust_rest_frame_wavelength = np.geomspace(0.01,1e6,10000)*u.micron
Addup_wavelength = np.geomspace(0.0912,1e6,10000)*u.micron
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
# for cmap_line in [plt.cm.viridis,plt.cm.plasma,plt.cm.inferno,plt.cm.magma,plt.cm.cividis]:

size = 5
fig = plt.figure(figsize=(size*1.5,size))


plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[plt.cm.viridis_r(i) for i in np.linspace(0,1,40)])
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue'])
# for n0 in [1,10,100,1000]:
# for gamma,n0 in [(0,100),(1,4e3),(2,4e4)]:

# array_pairs = []

# # for gamma,n0 in [(0.7,1000),(0,10)]:
# # para = [(0,10),(0,30),(0,100),(0,300),(0.2,30),(0.2,100),(0.2,300),(0.2,1000),(0.5,300),(0.5,1000),(0,8,1000)]
# # print(len(para))
# # for gamma,n0 in [(0,10),(0,30),(0,100),(0,300),(0.2,30),(0.2,100),(0.2,300),(0.2,1000),(0.5,300),(0.5,1000),(0.8,1000)]:
# for gamma in np.linspace(0,0.9,10):
#     for n0 in np.geomspace(10,1e4,100):
#         if n0 < 10**(0.55*gamma+2.9) and n0 > 10**(2.9*gamma+0.83):
#             # gamma = 0
#             # model_B87 = B87Model(n_0=n0, gamma=gamma,L_UV=1e46,T_sub=1500,NH_target=7.5e22,opacity=Orion_Opacity)
#             model_Orion = OrionLRDModel(n_0=n0, gamma=gamma,L_UV=1e46,T_sub=1500,NH_target=7.5e22,opacity=Orion_Opacity,incident_SED=Temple_SED)
#             # Lnu_B87 = model_B87.calc_L_nu(Dust_rest_frame_wavelength.to(u.Hz))
#             # Lnu_Orion = model_Orion.calc_L_nu(Dust_rest_frame_wavelength.to(u.Hz))
#             Lnu_Orion_photon = model_Orion.calc_L_nu_photon(Dust_rest_frame_wavelength.to(u.Hz))
#             # L_B87 = -1*np.trapz(Lnu_B87, Dust_rest_frame_wavelength.to(u.Hz))
#             # L_Orion = -1*np.trapz(Lnu_Orion, Dust_rest_frame_wavelength.to(u.Hz))
#             L_Orion_photon = -1*np.trapz(Lnu_Orion_photon, Dust_rest_frame_wavelength.to(u.Hz))
#             # Lnu_B87 = Lnu_B87/L_B87*(L_temple-L_temple_attenuated)
#             # Lnu_Orion = Lnu_Orion/L_Orion*(L_temple-L_temple_attenuated)
#             Lnu_Orion_photon = Lnu_Orion_photon/L_Orion_photon*(L_temple-L_temple_attenuated)

#             # y_out_Orion = model_Orion.r_out/model_Orion.r_in
#             # y_photosphere = ((1/4.8176)*(y_out_Orion**(1-gamma)-1)+1)**(1/(1-gamma))
#             # r_photosphere = model_Orion.r_in*y_photosphere
#             # T_photosphere = model_Orion.T_dust_profile(r_photosphere)


#             Addup1 = np.array([np.interp(wave,TempleWavelength_5_AA,TempleLnu_5_attenuated,left=0,right=0) for wave in Addup_wavelength.to(u.angstrom).value])
#             # Addup2Orion = np.array([np.interp(wave,Dust_rest_frame_wavelength.value, Lnu_Orion.value,left=0,right=0) for wave in Addup_wavelength.value])
#             # Addup2B87 = np.array([np.interp(wave,Dust_rest_frame_wavelength.value, Lnu_B87.value,left=0,right=0) for wave in Addup_wavelength.value])
#             Addup2Orion_photon = np.array([np.interp(wave,Dust_rest_frame_wavelength.value, Lnu_Orion_photon.value,left=0,right=0) for wave in Addup_wavelength.value])
#             # AddupOrion = Addup1 + Addup2Orion
#             # AddupB87 = Addup1 + Addup2B87
#             AddupOrion_photon = Addup1 + Addup2Orion_photon
#             # AddupOrion = AddupOrion/L_temple_4000AA
#             # AddupB87 = AddupB87/L_temple_4000AA
#             AddupOrion_photon = AddupOrion_photon/L_temple_4000AA
#             r_outOrion = model_Orion.r_out
#             # r_outB87 = model_B87.r_out
#             array_pairs.append((Addup_wavelength.to(u.micron).value,AddupOrion_photon))

#             # if r_outB87 > 1e1:
#             #     plt.loglog(Addup_wavelength.to(u.micron),AddupB87,label=f"$n_0$={n0},$\gamma$={gamma},$r_{{out}}$={r_outB87/1e3:.2f} kpc")
#             # else:
#             #     plt.loglog(Addup_wavelength.to(u.micron),AddupB87,label=f"$n_0$={n0},$\gamma$={gamma},$r_{{out}}$={r_outB87:.2f} pc")

#             # code for plotting
#         #     if r_outOrion > 1e1:
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0:.0e},$\gamma$={gamma},$r_{{out}}$={r_outOrion/1e3:.2f} kpc")
#         #         plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0:.0e}, $\gamma$={gamma}, $T_{{out}}$={model_Orion.T_out:.1f} K",alpha=0.5,lw=0.5)
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion,ls='--')
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0},$\gamma$={gamma},r_photo={r_photosphere/1e3:.2f} kpc,Photosphere")
#         #     else:
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0:.0e},$\gamma$={gamma},$r_{{out}}$={r_outOrion:.2f} pc, $T_{{out}}$={model_Orion.T_out:.1f} K")
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0:.0e},$\gamma$={gamma},$r_{{out}}$={r_outOrion:.2f} pc")
#         #         plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0:.0e}, $\gamma$={gamma}, $T_{{out}}$={model_Orion.T_out:.1f} K",alpha=0.5,lw=0.5)
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion,ls='--')
#         #         # plt.plot(Addup_wavelength.to(u.micron),AddupOrion_photon,label=f"$n_0$={n0},$\gamma$={gamma},r_photo={r_photosphere:.2f} pc,Photosphere")

# array_pairs = np.array(array_pairs)
# np.savez('analysis/data/processed/LRD_DustIR/Orion_photon.npz',array_pairs=array_pairs)

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmos = FlatLambdaCDM(67.74,0.3089)

L_bol = 5e45 # erg/s
NormalizedFactorAt1450 = L_bol/4.4/2.0675e+15 # erg/s/Hz
Av =3
A_1450 = 1.8*Av
f_1450 = np.e**(-A_1450/1.086)
loaded_data = np.load('analysis/data/processed/LRD_DustIR/Orion_photon.npz')
array_pairs_np_loaded = loaded_data['array_pairs']
for i, (x,y) in enumerate(array_pairs_np_loaded):
    # if i is even, plot the line
    if i< 40:
        y_1450 = y[np.argmin(np.abs(x-0.1450))]
        y = y/y_1450*NormalizedFactorAt1450*f_1450
        for redshift in [6]:
            D_L = cosmos.luminosity_distance(redshift)
            # change D_L to cm
            D_L = D_L.to(u.cm).value
            Flux_nu_o = y*(1+redshift)/(4*np.pi*(D_L)**2)
            Flux_nu_o_nJy = Flux_nu_o*1e23*1e9
            # print(Flux_nu_o_nJy[np.argmin(np.abs(x-0.1450))])
            ObservedWavelength = x*(1+redshift)

        plt.plot(ObservedWavelength,Flux_nu_o_nJy,alpha=0.8,lw=1.0,zorder=2)
        # plt.plot(ObservedWavelength,Flux_nu_o_nJy,label=f"Orion, $n_0$={10**3:.0e}, $\gamma$={0.7}",alpha=0.8,lw=0.5)
        # plt.plot(array_pairs_np_loaded[i][0],array_pairs_np_loaded[i][1],label=f"Orion, $n_0$={10**3:.0e}, $\gamma$={0.7}",alpha=0.1,lw=0.1)
curve_ax = plt.gca()
norm_curve = mcolors.Normalize(vmin=10,vmax=150)
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,norm=norm_curve),ax=curve_ax,ticks=np.linspace(10,150,5),\
    label='$n_0$ [cm$^{-3}$]',fraction=0.035, pad=0.01)
# 调整子图以减少边缘空白
plt.subplots_adjust(left=0.08, right=0.925, top=0.92, bottom=0.09)

# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# plt.plot(TempleWavelength_5_AA*1e-4,TempleLnu_5/L_temple_4000AA,label='De-reddened, No Dust',ls='--',color='#9467bd')

PerazGonzalez_wavelength, PerazGonzalez_flux = np.loadtxt('analysis/data/raw/Perez-Gonzalez.csv',comments='#',delimiter=',',usecols=(0,1),unpack=True)
ComsmosWeb_wavelength, ComsmosWeb_flux = np.loadtxt('analysis/data/raw/Cosmosweb.dat',comments='#',delimiter=' ',usecols=(0,1),unpack=True)
williams_wavelength, williams_flux = np.loadtxt('analysis/data/raw/williams.csv',comments='#',delimiter=',',usecols=(0,1),unpack=True)
williams_upper_bound_wavelength, williams_upper_bound_flux = np.loadtxt('analysis/data/raw/williams_upperbound.csv',comments='#',delimiter=',',usecols=(0,1),unpack=True)

# Swire_wavelength_AA, Swire_flux_Llambda = np.loadtxt('analysis/data/raw/swire_library/Mrk231_template_norm.sed',comments='#',delimiter=',',usecols=(0,1),unpack=True)
# swire_flux_lnu = Swire_flux_Llambda*Swire_wavelength_AA**2/3e18
# swire_freq_Hz = 3e10/(Swire_wavelength_AA*1e-8)
# arg_4000A_swire = np.argmin(np.abs(Swire_wavelength_AA-4000))
# L_swire_4000A = swire_flux_lnu[arg_4000A_swire]
# swire_flux_lnu = swire_flux_lnu/L_swire_4000A*25
# plt.plot(Swire_wavelength_AA*1e-4,swire_flux_lnu,label='Mrk231')

# Swire_wavelength_AA_torus, Swire_flux_Llambda_torus = np.loadtxt('analysis/data/raw/swire_library/Torus_template_norm.sed',comments='#',delimiter=',',usecols=(0,1),unpack=True)
# swire_flux_lnu_torus = Swire_flux_Llambda_torus*Swire_wavelength_AA_torus**2/3e18
# swire_freq_Hz_torus = 3e10/(Swire_wavelength_AA_torus*1e-8)
# arg_4000A_swire_torus = np.argmin(np.abs(Swire_wavelength_AA_torus-4000))
# L_swire_4000A_torus = swire_flux_lnu_torus[arg_4000A_swire_torus]
# swire_flux_lnu_torus = swire_flux_lnu_torus/L_swire_4000A_torus*25
# Swire_wavelength_AA_torus, Swire_flux_Llambda_torus = np.loadtxt('analysis/data/raw/swire_library/QSO2_template_norm.sed',comments='#',delimiter=',',usecols=(0,1),unpack=True)
Swire_wavelength_AA_Arp220, Swire_flux_Llambda_Arp220 = np.genfromtxt('analysis/data/raw/swire_library/Arp220_template_norm.sed',comments='#',delimiter=None,usecols=(0,1),unpack=True)


swire_flux_lnu_arp220 = Swire_flux_Llambda_Arp220*Swire_wavelength_AA_Arp220**2/3e18



swire_freq_Hz_arp220 = 3e10/(Swire_wavelength_AA_Arp220*1e-8)


arg_4000A_swire_arp220 = np.argmin(np.abs(Swire_wavelength_AA_Arp220-4000))


L_swire_4000A_arp220 = swire_flux_lnu_arp220[arg_4000A_swire_arp220]


swire_flux_lnu_arp220 = swire_flux_lnu_arp220/L_swire_4000A_arp220*25

# plt.plot(Swire_wavelength_AA_torus*1e-4*(1+6),swire_flux_lnu_torus,label='Torus',color='#cde7f0',alpha = 0.8,zorder=1)#d62728
# plt.plot(Swire_wavelength_AA_Arp220*1e-4*(1+6),swire_flux_lnu_arp220,label='Arp220',color='#cde7f0',alpha = 0.8,zorder=1)

def plot_dust(dust_name,L_bol):
    wavelength_AA, flux_Llambda = np.genfromtxt(f'analysis/data/raw/swire_library/{dust_name}_template_norm.sed',comments='#',delimiter=None,usecols=(0,1),unpack=True)
    flux_lnu = flux_Llambda*wavelength_AA**2/3e18
    freq_Hz = 3e10/(wavelength_AA*1e-8)
    L_trapz = -1*np.trapz(flux_lnu, freq_Hz)
    flux_lnu_normalized = flux_lnu/L_trapz*L_bol
    flux_lnu_nJy = flux_lnu_normalized*(1+redshift)/(4*np.pi*(D_L)**2)*1e23*1e9
    return wavelength_AA, flux_lnu_nJy

wave_torus, flux_torus = plot_dust('Arp220',5e45)
plt.plot(wave_torus*1e-4*(1+6),flux_torus,label='Tpye-2 QSO/Starburst/ULIRG',color='#cde7f0',alpha = 0.8,zorder=1)#d62728

wave_2, flux_2 = plot_dust('Torus',5e45)
plt.plot(wave_2*1e-4*(1+6),flux_2,color='#cde7f0',alpha = 0.8,zorder=1)#d62728

# wave_3, flux_3 = plot_dust('QSO2',5e45)
# plt.plot(wave_3*1e-4*(1+6),flux_3,label='QSO2',color='purple',alpha = 0.8,zorder=1)#d62728

wave_4, flux_4 = plot_dust('Mrk231',5e45)
plt.plot(wave_4*1e-4*(1+6),flux_4,color='#cde7f0',alpha = 0.8,zorder=1)#d62728

# wave_5, flux_5 = plot_dust('N6240',5e45)
# plt.plot(wave_5*1e-4*(1+6),flux_5,label='NGC6240',color='orange',alpha = 0.8,zorder=1)#d62728

# wave_6, flux_6 = plot_dust('Sey2',5e45)
# plt.plot(wave_6*1e-4*(1+6),flux_6,label='Sey2',color='black',alpha = 0.8,zorder=1)#d62728

# wave_7, flux_7 = plot_dust('Sey18',5e45)
# plt.plot(wave_7*1e-4*(1+6),flux_7,label='Sey18',color='brown',alpha = 0.8,zorder=1)#d62728

arg_1 = (ComsmosWeb_wavelength<10) & (ComsmosWeb_wavelength>0.7)
arg_2 = (ComsmosWeb_wavelength>10) | (ComsmosWeb_wavelength<0.7)
ComsmosWeb_wavelength_1 = ComsmosWeb_wavelength[arg_1]
ComsmosWeb_wavelength_2 = ComsmosWeb_wavelength[arg_2]

# williams_wavelength = williams_wavelength/(1+6)
# williams_upper_bound_wavelength = williams_upper_bound_wavelength/(1+6)

# williams_flux = williams_flux/williams_flux[np.argmin(np.abs(williams_wavelength-0.4))]
# williams_upper_bound_flux = williams_upper_bound_flux/williams_flux[np.argmin(np.abs(williams_wavelength-0.4))]

ComsmosWeb_wavelength = ComsmosWeb_wavelength
ComsmosWeb_flux_1 = ComsmosWeb_flux[arg_1]
ComsmosWeb_flux_2 = ComsmosWeb_flux[arg_2]

# plt.plot(PerazGonzalez_wavelength, PerazGonzalez_flux,label='Stacked LRDs (Pérez-González+24)',zorder=2,color='#ff7f0e',alpha=0.4,lw=4)
# plt.plot(PerazGonzalez_wavelength, PerazGonzalez_flux,label='Stacked LRDs (Pérez-González+24)',zorder=2,color='#0E8DE6',alpha=0.9)
# plt.scatter(ComsmosWeb_wavelength_1,ComsmosWeb_flux_1,label='Stacked LRDs (Akins+24)',zorder=3,color='#0E8DE6')
plt.scatter(ComsmosWeb_wavelength_1,ComsmosWeb_flux_1,label='Akins+24, stacked LRDs',zorder=3,edgecolors='black',facecolors='#ff0033',alpha=1)
plt.scatter(ComsmosWeb_wavelength_2,ComsmosWeb_flux_2,label='Akins+24, upper bound',marker='v',facecolors='none',lw=1.3,edgecolors='#ff0033',zorder=3)
# plt.scatter(ComsmosWeb_wavelength_2,ComsmosWeb_flux_2,label='Stacked LRDs (Akins+24), upper bound',marker='v',zorder=3,c='#0E8DE6')
# plt.scatter(williams_wavelength,williams_flux,label='Stacked LRDs (Williams+14)',zorder=3,edgecolors='black',facecolors='none',alpha=1)
# plt.scatter(williams_upper_bound_wavelength,williams_upper_bound_flux,label='Stacked LRDs (Williams+14), upper bound',marker='v',facecolors='#ff0033',lw=1.3,edgecolors='#ff0033',zorder=3)

# flux_at06 = LogLogInterpolator(Addup_wavelength.value, AddupOrion_photon)(0.6)
# flux_at07 = LogLogInterpolator(Addup_wavelength.value, AddupOrion_photon)(0.7)
# flux_at065 = LogLogInterpolator([0.6,0.7], [flux_at06,flux_at07])(0.65)
# flux_at10 = LogLogInterpolator(Addup_wavelength.value, AddupOrion_photon)(1.03675)

# plt.scatter(0.6,flux_at06,label='Orion, 0.6$\mu$m, flux = {:.2e}'.format(flux_at06),marker='x',zorder=3,c='#d62728')
# plt.scatter(0.7,flux_at07,label='Orion, 0.7$\mu$m,flux = {:.2e}'.format(flux_at07),marker='x',zorder=3,c='#d62728')
# plt.scatter(0.65,flux_at065,label='fitting, 0.65$\mu$m, flux = {:.2e}'.format(flux_at065),marker='x',zorder=3,c='#d62728')
# plt.scatter(1.2,LogLogInterpolator(Addup_wavelength.value, AddupOrion_photon)(1.2),label='Orion, 1.2$\mu$m, flux={:.2e}'.format(LogLogInterpolator(Addup_wavelength.value, AddupOrion_photon)(1.2)),marker='x',zorder=3,c='#d62728')

# plt.plot(TempleWavelength_5_AA,TempleLnu_5_4000AA,label='tmpe5')
# plt.plot(T_1e3_wavelength_micron*(1e4), Weighted_Addup_Jnu/TempleLnu_5[arg_4000AA_Temple5], label='CLOUDY Dust')
# plt.plot(addupwave,Addup,label='Addup')
# plt.plot(Temple5freq_Hz,TempleLnu_5,label='CLOUDY Dust')
# plt.plot(Weighted_freq_Hz[arg_cutoff:],Weighted_Addup_Jnu[arg_cutoff:],label='Weighted Dust')
import matplotlib.lines as mlines
import matplotlib.legend_handler as mlegend_handler
# Custom handler to display both markers in one legend entry
# Custom handler to display both markers in one legend entry
class HandlerMultipleMarkers(mlegend_handler.HandlerTuple):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        x1 = x0 + width / 2.0 - 5  # Adjust position of first marker
        y1 = y0 + height / 2.0
        x2 = x0 + width / 2.0 + 5  # Adjust position of second marker
        y2 = y0 + height / 2.0
        legline1 = mlines.Line2D([x1], [y1], linestyle='None', marker='o', markersize=7,
                                 markeredgecolor='black', markerfacecolor='#ff0033', transform=trans)
        legline2 = mlines.Line2D([x2], [y2], linestyle='None', marker='v', markersize=7,
                                 markeredgecolor='#ff0033', markerfacecolor='none', lw=1.3, transform=trans)
        return [legline1, legline2]

# Create custom legend handles
handle1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=7,
                        markeredgecolor='black', markerfacecolor='#ff0033')
handle2 = mlines.Line2D([], [], color='#ff0033', marker='v', linestyle='None', markersize=7,
                        markeredgecolor='#ff0033', markerfacecolor='none', lw=1.3)

# Combine handles into a tuple
combined_handles = (handle1, handle2)

# Create a custom legend
plt.legend([combined_handles, mlines.Line2D([], [], color='#cde7f0')],
           ['Akins+24', 'AGN torus/Starburst/ULIRG'],loc='upper left', handler_map={tuple: HandlerMultipleMarkers()})

plt.xlim(5e-1,1e4)
plt.ylim(1e-1,1e8)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Observed-frame Wavelength [$\mu$m]')
plt.ylabel('Observed Flux Density [nJy]')

# plt.legend(fontsize=8,loc='upper left')
font = {'family': 'serif', 'weight': 'bold', 'size': 10}
plt.text(0.7,1e3,'Attenuated AGN SED by\nDust Grain $\gtrsim$ 0.1 $\mu$m',fontdict=font,color='#d62728')
plt.text(120,5e6,'Re-emission from Dust\nwith an Extended Distribution',fontdict=font,color='#d62728')
plt.text(0.7,6e4,'$L_\mathrm{bol}=5\\times10^{45}$ erg s$^{-1}$\nRedshift$=$6')
# Get the current axes
ax = plt.gca()

# ax.legend(labels=['QSO','Akins+24','Akins+24'],loc='upper left',fontsize=8)


for start, end, label, color, y_pos , height in ranges:
    rect = patches.Rectangle((start, y_pos), end-start, height=height, color=color, alpha=0.6, label=label)
    ax.add_patch(rect)
    ax.text(np.sqrt(start * end), np.sqrt(y_pos * (y_pos + height)), label, ha='center', va='center', color='white', fontsize=8, weight='bold')


# Show ticks on all four axes
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)


ax2 = plt.twiny()
ax2.set_xlabel('Rest-frame Wavelength [$\mu$m]')
ax2.set_xscale('log')
ax2.set_xlim(5e-1/7,1e4/7)
ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# plt.title('$L_\mathrm{bol}=5\\times10^{45}$ erg s$^{-1}$, Redshift=6')
# plt.ylabel('f_nu/f_nu(4000AA)')

# for n,i in enumerate(np.geomspace(10,1e4,100)):
#     if n < 40:
#         print(i)
# ax3 = plt.gca().twinx()
# ax3.set_ylabel('AB magnitude')
# ax3.set_ylim(33.9000656223,11.4000656223)



# plt.tight_layout()
plt.show()
# plt.savefig('analysis/figure/LRD/Figure6_Colorbar.png',dpi=600)
# plt.savefig('analysis/figure/LRD/VIP_Lyman_off.png',dpi=600)
# plt.savefig('analysis/figure/LRD/Precise_IRSED_gamma.pdf',dpi=600)
# plt.savefig('analysis/figure/LRD/Precise_IRSED_n0.pdf',dpi=600)