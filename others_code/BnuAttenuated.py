import os
import numpy as np

RestframeWavelength_micron = np.logspace(-1,2,10000) # micron
RestframeFrequency = 3e10/(RestframeWavelength_micron*1e-4) # Hz

# import Orion opacity data


OrionOpacityFile = 'Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc'

OrionWavelength_Rydberg , Orion_Opacity_SigmaPerH = np.loadtxt(OrionOpacityFile,comments='#', delimiter='\t', usecols=(0, 1),unpack=True)

OrionWavelength_micro = 911.27/OrionWavelength_Rydberg*1e-4 # micron
OrionWavelength_x = 1/OrionWavelength_micro # rising order

def Bnu(nu,T):
    return 2*6.63e-27*nu**3/(3e10)**2/(np.exp(6.63e-27*nu/(1.38e-16*T))-1)*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH)

T_1e3_Bnu = np.array([Bnu(nu,T=1e3) for nu in RestframeFrequency])
T_2e2_Bnu = np.array([Bnu(nu,T=2e2) for nu in RestframeFrequency])
T_4e2_Bnu = np.array([Bnu(nu,T=4e2) for nu in RestframeFrequency])
T_4e2_Bnu = np.array([Bnu(nu,T=6e2) for nu in RestframeFrequency])
T_6e2_Bnu = np.array([Bnu(nu,T=8e2) for nu in RestframeFrequency])
T_8e2_Bnu = np.array([Bnu(nu,T=1e3) for nu in RestframeFrequency])

# import CLOUDY
T_1e3_wavelength_Ryd, T_1e3_flux_nuJnu = np.loadtxt('analysis/data/processed/LRD_DustIR/LRD_Td1e3.cont', unpack=True, usecols=(0, 4))
T_2e2_wavelength_Ryd, T_2e2_flux_nuJnu = np.loadtxt('analysis/data/processed/LRD_DustIR/LRD_Td2e2.cont', unpack=True, usecols=(0, 4))
T_4e2_wavelength_Ryd, T_4e2_flux_nuJnu = np.loadtxt('analysis/data/processed/LRD_DustIR/LRD_Td4e2.cont', unpack=True, usecols=(0, 4))
T_6e2_wavelength_Ryd, T_6e2_flux_nuJnu = np.loadtxt('analysis/data/processed/LRD_DustIR/LRD_Td6e2.cont', unpack=True, usecols=(0, 4))
T_8e2_wavelength_Ryd, T_8e2_flux_nuJnu = np.loadtxt('analysis/data/processed/LRD_DustIR/LRD_Td8e2.cont', unpack=True, usecols=(0, 4))

T_1e3_wavelength_micron = 1e-4*911.3/T_1e3_wavelength_Ryd
T_2e2_wavelength_micron = 1e-4*911.3/T_2e2_wavelength_Ryd
T_4e2_wavelength_micron = 1e-4*911.3/T_4e2_wavelength_Ryd
T_6e2_wavelength_micron = 1e-4*911.3/T_6e2_wavelength_Ryd
T_8e2_wavelength_micron = 1e-4*911.3/T_8e2_wavelength_Ryd

T_1e3_freq_Hz = 3e10/(T_1e3_wavelength_micron*1e-4)
T_2e2_freq_Hz = 3e10/(T_2e2_wavelength_micron*1e-4)
T_4e2_freq_Hz = 3e10/(T_4e2_wavelength_micron*1e-4)
T_6e2_freq_Hz = 3e10/(T_6e2_wavelength_micron*1e-4)
T_8e2_freq_Hz = 3e10/(T_8e2_wavelength_micron*1e-4)

T_2e2_cut = 3
arg_T_2e2_cut = np.argmin(np.abs(T_2e2_wavelength_micron - T_2e2_cut))
T_4e2_cut = 1.5
arg_T_4e2_cut = np.argmin(np.abs(T_4e2_wavelength_micron - T_4e2_cut))
T_6e2_cut = 0.75
arg_T_6e2_cut = np.argmin(np.abs(T_6e2_wavelength_micron - T_6e2_cut))
T_8e2_cut = 0.375
arg_T_8e2_cut = np.argmin(np.abs(T_8e2_wavelength_micron - T_8e2_cut))
T_1e3_cut = 0.1875
arg_T_1e3_cut = np.argmin(np.abs(T_1e3_wavelength_micron - T_1e3_cut))

T_2e2_flux_nuJnu[arg_T_2e2_cut:] = 0
T_4e2_flux_nuJnu[arg_T_4e2_cut:] = 0
T_6e2_flux_nuJnu[arg_T_6e2_cut:] = 0
T_8e2_flux_nuJnu[arg_T_8e2_cut:] = 0
T_1e3_flux_nuJnu[arg_T_1e3_cut:] = 0

T_1e3_flux_Jnu = T_1e3_flux_nuJnu/T_1e3_freq_Hz
T_2e2_flux_Jnu = T_2e2_flux_nuJnu/T_2e2_freq_Hz
T_4e2_flux_Jnu = T_4e2_flux_nuJnu/T_4e2_freq_Hz
T_6e2_flux_Jnu = T_6e2_flux_nuJnu/T_6e2_freq_Hz
T_8e2_flux_Jnu = T_8e2_flux_nuJnu/T_8e2_freq_Hz



from matplotlib import pyplot as plt
plt.plot(RestframeWavelength_micron,T_1e3_Bnu*5e17,label='T=1e3, Bnu')
plt.plot(T_1e3_wavelength_micron,T_1e3_flux_Jnu,label='T=1e3, CLOUDY')

plt.plot(RestframeWavelength_micron,T_2e2_Bnu*5e17,label='T=2e2, Bnu')
plt.plot(T_2e2_wavelength_micron,T_2e2_flux_Jnu,label='T=2e2, CLOUDY')

# plt.plot(RestframeWavelength_micron,T_4e2_Bnu*1e24,label='T=4e2, Bnu')
# plt.plot(T_4e2_wavelength_micron,T_4e2_flux_Jnu,label='T=4e2, CLOUDY')

# plt.plot(RestframeWavelength_micron,T_6e2_Bnu*1e24,label='T=6e2, Bnu')
# plt.plot(T_6e2_wavelength_micron,T_6e2_flux_Jnu,label='T=6e2, CLOUDY')

# plt.plot(RestframeWavelength_micron,T_8e2_Bnu*1e24,label='T=8e2, Bnu')
# plt.plot(T_8e2_wavelength_micron,T_8e2_flux_Jnu,label='T=8e2, CLOUDY')

plt.ylim(1e-22,1e-7)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()