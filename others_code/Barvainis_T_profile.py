import numpy as np
from typing import override
from scipy import integrate, optimize
# from scipy import integrate
import matplotlib.pyplot as plt
import astropy.units as u
import os
from astropy import constants as const
from LRD_IR_spectrum import Planck_B_nu,LogLogInterpolator

import cProfile


class B87Model:
    sigma_H_UV = 6.82e-23  # TEMP, from Orion's maximum value

    def __init__(self, n_0, gamma, L_UV, T_sub, NH_target=None):
        """
        Initialize the model with these parameters. The units are in cgs unless specified otherwise.
        r_in is in [pc]
        """
        self.n_0 = n_0
        self.gamma = gamma
        self.L_UV = L_UV
        self.T_sub = T_sub
        if T_sub == 1000:
            self.r_in = 3.243 #self.calc_r_in()  # in [pc]
        elif T_sub == 1500:
            self.r_in = 0.974
        self.NH_target = NH_target
        self.r_out = self.find_r_out()
        # self.r_arr = np.logspace(np.log10(self.r_in),np.log10(self.r_out),1000)
        # self.len_r = int(self.r_out/self.r_in*2)
        # self.len_r = max(self.len_r, 1000)
        self.len_r = 1000
        self.r_arr = np.logspace(np.log10(self.r_in), np.log10(self.r_out), self.len_r)
        # if self.r_out/len_r < self.r_in:
        #     self.r_arr = np.linspace(self.r_in, self.r_out, len_r)
        # else:
        #     self.r_arr = np.logspace(np.log10(self.r_in), np.log10(self.r_out), len_r)
            # self.r_arr = np.linspace(self.r_in, self.r_out, len_r)
        self.T_arr = self.T_dust_B87(self.r_arr)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_0={self.n_0}, gamma={self.gamma}, L_UV={self.L_UV}, T_sub={self.T_sub}, r_in={self.r_in})"

    @staticmethod
    def r_in_B87(L_UV_cgs, T_sub):
        L_46 = L_UV_cgs / 1e46
        T_1500 = T_sub / 1500
        return 1.3 * L_46**(1/2) * T_1500**(-2.8)

    @staticmethod
    def r_in_Kohei(L_UV_cgs, T_sub):
        sigma_SB = 5.67e-5
        return np.sqrt(L_UV_cgs/(16.0e0*np.pi*sigma_SB)/(T_sub**4.0e0)) * (u.cm.to(u.pc))

    def calc_r_in(self):  # in [pc]
        return self.r_in_B87(self.L_UV, self.T_sub)

    def n_profile(self, r):
        r_ratio = r / self.r_in
        return self.n_0 * r_ratio**(-self.gamma)

    def NH_profile(self, r):
        """Column density profile in cgs"""
        gamma = self.gamma  # for brevity
        r_ratio = r / self.r_in
        r_in_cgs = self.r_in * (u.pc.to(u.cm))
        if gamma == 1:  # special case: gamma == 1
            factor = np.log(r_ratio)
        else:
            factor = (r_ratio**(1 - gamma) - 1) / (1 - gamma)
        return self.n_0 * r_in_cgs * factor

    def tau_profile(self, r):
        """Optical depth profile"""
        return self.sigma_H_UV * self.NH_profile(r)

    def find_r_out(self):
        """find the r_out that could give the specified NH"""
        NH_target = self.NH_target
        gamma = self.gamma
        r_in_cgs = self.r_in * (u.pc.to(u.cm))
        factor = NH_target / (self.n_0 * r_in_cgs)  #* if NH_target is None, this will raise an error
        if gamma > 1 and factor > 1/(gamma-1):
            #! when gamma > 1, the maximum of NH is n_0*r_in_cgs / (gamma-1). If NH_target > this value, r_out cannot be found and will give negative or complex value.
            raise ValueError(f"The {NH_target = } is larger than possible in this NH_profile with {gamma = }, cannot find r_out")

        if gamma == 1:
            r_out_ratio = np.exp(factor)
        else:
            r_out_ratio = (factor * (1-gamma) + 1) ** (1/(1-gamma))
        return self.r_in * r_out_ratio

    def T_dust_B87(self, r):
        beta = 1.6
        L_46 = self.L_UV / 1e46
        tau = self.tau_profile(r)
        tau = np.where(tau > 500, 500, tau)
        T_dust = 1650 * (L_46/r**2)**(1/(4+beta)) * np.exp(-tau / (4+beta))  # in [K]
        T_dust = np.where(T_dust < 1, 1, T_dust)  # set a floor of 1 K
        return T_dust

    def T_dust_power_law(self, p, r):
        return self.T_sub * (r / self.r_in) ** (-p)


# import Orion opacity data
# CLOUDY_Path = '/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/model/LRDChangeISMGrain'
CLOUDY_Path = '/Users/chenkejian/Library/Mobile Documents/com~apple~CloudDocs/北大/科研/JWST_LRD_SED/IR_spectrum/data'

OrionFile = 'Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc'

# OrionOpacityFile = os.path.join(CLOUDY_Path,OrionFile)
OrionOpacityFile = os.path.join('data',OrionFile)  #CHANGEME
OrionWavelength_Rydberg , Orion_Opacity_SigmaPerH, Orion_Opacity_SigmaPerH_abs = np.loadtxt(OrionOpacityFile,comments='#', delimiter='\t', usecols=(0, 1 ,2),unpack=True)


OrionWavelength_micro = 911.27/OrionWavelength_Rydberg*1e-4 # micron
OrionWavelength_x = 1/OrionWavelength_micro # rising order
# print(np.all(np.diff(OrionWavelength_x)> 0))

# Temple data
TempleWavelength_5_AA, TempleLlambda_5 = np.loadtxt('data/quasar5.template',comments='#',delimiter=' ',usecols=(0,1),unpack=True)
TempleFrequency_5 = 3e10/(TempleWavelength_5_AA*1e-8) # Hz
TempleLnu5 = TempleLlambda_5*TempleWavelength_5_AA**2/3e18
L_bol = 2.4e46 # erg/s
NormalizedFactorAt1450 = L_bol/4.4/2.0675e+15 # erg/s/Hz
Temple_NormalizedLnu_5 = TempleLnu5/TempleLnu5[np.argmin(np.abs(TempleWavelength_5_AA-1450))]*NormalizedFactorAt1450
# print(-1*np.trapz(Temple_NormalizedLnu_5,3e10/(TempleWavelength_5_AA*1e-8)))


RestframeWavelength_micron = np.logspace(-0.5,3.5,200) # micron
RestframeWavelength_micron_forOrion = np.logspace(-2,5,1000)
RestframeFrequency = 3e10/(RestframeWavelength_micron*1e-4) # Hz
RestframeFrequency_forOrion = 3e10/(RestframeWavelength_micron_forOrion*1e-4) # Hz
Lnu_3micro_arg = np.argmin(np.abs(RestframeWavelength_micron-3))
Lnu_10micro_arg = np.argmin(np.abs(RestframeWavelength_micron-10))
Lnu_20micro_arg = np.argmin(np.abs(RestframeWavelength_micron-20))
Lnu_300micro_arg = np.argmin(np.abs(RestframeWavelength_micron-300))
T_sub = 1500

class OrionLRD(B87Model):
    def __init__(self, n_0, gamma, L_UV, T_sub, NH_target=None):
        super().__init__(n_0, gamma, L_UV, T_sub, NH_target)
        self.T_array = np.linspace(0, 1.5e3, 1500) #TEMP
        self.interp = LogLogInterpolator([self.IR_Flux(T) for T in self.T_array], self.T_array)
        self.T_arr = self.T_arr_Orion()

    def UV_Flux(self, r):
        _y0 = r/self.r_in # r in pc
        _integral_factor = (_y0**(-self.gamma+1)-1)/(-self.gamma+1)
        def tau_nu(nu):
            return self.n_0 * _integral_factor * self.r_in * (u.pc.to(u.cm)) * np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH)
        # Lnu e^(-tau)/ 4 /pi/r^2 * sigma_H_abs
        def Lnu(nu):
            return np.interp(3e10/nu*1e8,TempleWavelength_5_AA,Temple_NormalizedLnu_5) * np.exp(-tau_nu(nu)) / (4 * np.pi * (r * u.pc.to(u.cm))**2) * np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs)
        Lnus = np.array([Lnu(nu) for nu in RestframeFrequency_forOrion])
        return integrate.trapezoid(Lnus, RestframeFrequency_forOrion)*1e18*(-1)

    def IR_Flux(self, T):
        def BnuSigma_abs_nu(nu):
            return Planck_B_nu(nu, T) * np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs)*4*np.pi
        Bnus = np.array([BnuSigma_abs_nu(nu) for nu in RestframeFrequency_forOrion])
        return integrate.trapezoid(Bnus, RestframeFrequency_forOrion)*1e18*(-1)

    def T_dust_profile(self, r):
        return self.interp(self.UV_Flux(r))
        # return self.interp(self.UV_Flux(r))

    def T_arr_Orion(self):
        return np.array([self.T_dust_profile(r) for r in self.r_arr])



# Define functions
def Lnu_kappanuBnu(nu,T):
    if 4.8043478261E-11*nu/(T) > 100:
        return 0
    if 4.8043478261E-11*nu/(T) < 1e-3:
        return 2*nu**2*1.38e-16*T/(3e10)**2*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH)
    else:
        return 2*6.63e-27*nu**3/(3e10)**2/(np.exp(6.63e-27*nu/(1.38e-16*T))-1)*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH)

def Lnu_kappanuBnu_abs(nu,T):
    if 4.8043478261E-11*nu/(T) > 100:
        return 0
    if 4.8043478261E-11*nu/(T) < 1e-3:
        return 2*nu**2*1.38e-16*T/(3e10)**2*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs)
    else:
        return 2*6.63e-27*nu**3/(3e10)**2/(np.exp(6.63e-27*nu/(1.38e-16*T))-1)*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs)


def Total_Lnu_kappanuBnu(nu,model,gamma):
    # return np.trapz([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)],model.r_arr)
    # integrate in log log space
    # return np.trapz(np.exp(np.log(np.array([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)]))+np.log(model.r_arr)),np.log(model.r_arr))
    return np.trapz([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)],model.r_arr)

def Total_Lnu_kappanuBnu_abs(nu,model,gamma):
    return np.trapz([Lnu_kappanuBnu_abs(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)],model.r_arr)


def ToBeIntegrated(nu,model,gamma):
    return np.array([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)])

# import time
# start1 = time.time()
# model_test = OrionLRD(n_0=40, gamma=0.2, L_UV=1e46, T_sub=1000, NH_target=7.5e22)
# cProfile.run(OrionLRD(n_0=40, gamma=0.2, L_UV=1e46, T_sub=1000, NH_target=7.5e22))
# end1 = time.time()
# print(model_test.interp(10))
# end2 = time.time()
# print(model_test.interp(20))
# end3 = time.time()
# print(end1-start1,end2-end1,end3-end2)

def plot_model(gamma,n0):
    _model = OrionLRD(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=T_sub, NH_target=7.5e22)
    _model_B87 = B87Model(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=T_sub, NH_target=7.5e22)
    _model_Lnu = np.array([Total_Lnu_kappanuBnu_abs(nu,_model,gamma) for nu in RestframeFrequency])
    _model_Lnu_B87 = np.array([Total_Lnu_kappanuBnu_abs(nu,_model_B87,gamma) for nu in RestframeFrequency])
    # _model_Lnu_abs = np.array([Total_Lnu_kappanuBnu_abs(nu,_model,gamma) for nu in RestframeFrequency])
    _ratio_3_10 = np.log10(_model_Lnu[Lnu_3micro_arg])-np.log10(_model_Lnu[Lnu_10micro_arg])
    _ratio_3_20 = np.log10(_model_Lnu[Lnu_3micro_arg])-np.log10(_model_Lnu[Lnu_20micro_arg])
    _ratio_3_300 = np.log10(_model_Lnu[Lnu_300micro_arg]) - np.log10(_model_Lnu[Lnu_3micro_arg])
    plt.plot(RestframeWavelength_micron,_model_Lnu,label=f'n0={n0}, gamma={gamma},T_sub={T_sub},\nratio_3_10={_ratio_3_10:.2f}, ratio_3_20={_ratio_3_20:.2f}, ratio_3_300={_ratio_3_300:.2f},\nr_in={_model.r_in:.2f} pc, r_out={_model.r_out/1e3:.2f} kpc, T_out={_model.T_arr[-1]:.2f} K')
    plt.plot(RestframeWavelength_micron,_model_Lnu_B87,linestyle='--')
    # plt.plot(RestframeWavelength_micron,_model_Lnu_abs,linestyle='--')
    plt.scatter([3,10,20,300],_model_Lnu[[Lnu_3micro_arg,Lnu_10micro_arg,Lnu_20micro_arg,Lnu_300micro_arg]],marker='x')
    plt.text(3,_model_Lnu[Lnu_3micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_3micro_arg])))
    plt.text(10,_model_Lnu[Lnu_10micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_10micro_arg])))
    plt.text(20,_model_Lnu[Lnu_20micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_20micro_arg])))
    plt.text(300,_model_Lnu[Lnu_300micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_300micro_arg])))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(1e-34,1e-26)
    plt.show()

# model = OrionLRD(n_0=40, gamma=0.4, L_UV=1e46, T_sub=1500, NH_target=7.5e22)
# r_arry = np.linspace(model.r_in,model.r_out,1000)
# model_UVflux = np.array([model.UV_Flux(r) for r in r_arry])
# plt.loglog(r_arry,model_UVflux)
# plt.show()
# plot_model(gamma=0.4,n0=40)  #TEMP

# biscection
left_hand = -1*np.trapz(4* np.pi * np.array([Lnu_kappanuBnu_abs(nu,T_sub) for nu in RestframeFrequency]),RestframeFrequency)


def L_temple_nu(nu):
    return np.interp(3e10/nu,TempleWavelength_5_AA*1e-8,Temple_NormalizedLnu_5)

def right_hand(r):
    return -1*np.trapz(np.array([L_temple_nu(nu)/r**2/4/np.pi*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs) for nu in TempleFrequency_5]),TempleFrequency_5)

def biscetion_f(r):
    _r = r*3.0856e18 # cm
    return left_hand - right_hand(_r)

def bisection(f, a, b, tol):
    """
    Bisection method for finding roots of a function f.

    Parameters:
    f : function
        The function for which we are trying to approximate a root.
    a, b : numbers
        The interval in which to search for a root. The function values at the endpoints must have opposite signs.
    tol : number
        The tolerance level for stopping the iteration.

    Returns:
    number
        The approximate root of the function.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("The function must have different signs at the endpoints a and b.")

    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint) == 0:
            return midpoint  # The midpoint is the root
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

    return (a + b) / 2

# import time
# start = time.time()
# print(bisection(biscetion_f,0.1,10,1e-4))
# end = time.time()
# print(end-start)




# print(-1*np.trapz(Temple_NormalizedLnu_5,3e10/(TempleWavelength_5_AA*1e-8)))


# model0 = B87Model(n_0=1e1, gamma=0, L_UV=1e46, T_sub=1000, NH_target=7.5e22)


# check the boundary for gamma > 1
# gamma_arr = np.linspace(1.01,2,100)
# n0_arr = np.logspace(5,0,1000)
# n0_boundary_arr = []
# for gamma in gamma_arr:
#     n0_boundary = 0
#     for n0 in n0_arr:
#         try:
#             model = B87Model(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=T_sub, NH_target=7.5e22)
#             n0_boundary = n0
#         except ValueError:
#             break
#     n0_boundary_arr.append(n0_boundary)

# plt.plot(gamma_arr,n0_boundary_arr,label='n0_boundary')
# plt.plot(gamma_arr,6.1e3*(gamma_arr-1),label='n0=6.1e3*(gamma-1)')
# plt.legend()
# plt.show()



# # import Orion opacity data
# CLOUDY_Path = '/Users/lzr/ProjectsFiles/SEDSimulation/KoheisCodeForSED/cloudy/model/LRDChangeISMGrain'

# OrionFile = 'Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc'

# OrionOpacityFile = os.path.join(CLOUDY_Path,OrionFile)
# OrionWavelength_Rydberg , Orion_Opacity_SigmaPerH , Orion_Opacity_SigmaPerH_abs = np.loadtxt(OrionOpacityFile,comments='#', delimiter='\t', usecols=(0, 1 ,2),unpack=True)

# OrionWavelength_micro = 911.27/OrionWavelength_Rydberg*1e-4 # micron
# OrionWavelength_x = 1/OrionWavelength_micro # rising order


# def Lnu_kappanuBnu_abs(nu,T):
#     if 4.8043478261E-11*nu/(T) > 100:
#         return 0
#     if 4.8043478261E-11*nu/(T) < 1e-3:
#         return 2*nu**2*1.38e-16*T/(3e10)**2*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs)
#     else:
#         return 2*6.63e-27*nu**3/(3e10)**2/(np.exp(6.63e-27*nu/(1.38e-16*T))-1)*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH_abs)

# def Total_Lnu_kappanuBnu_abs(nu,model,gamma):
#     return np.trapz([Lnu_kappanuBnu_abs(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)],model.r_arr)

# RestframeWavelength_micron = np.logspace(-0.5,3.5,200) # micron
# RestframeFrequency = 3e10/(RestframeWavelength_micron*1e-4) # Hz

# TempleWavelength_5_AA, TempleLlambda_5 = np.loadtxt('analysis/data/raw/Temple/quasar5.template',comments='#',delimiter=' ',usecols=(0,1),unpack=True)
# TempleFrequency_5 = 3e10/(TempleWavelength_5_AA*1e-8)
# TempleLnu_5 = TempleLlambda_5*TempleWavelength_5_AA**2/3e18
# TempleLnu_5_attenuated = TempleLnu_5*np.exp(np.array([-np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH) for nu in TempleFrequency_5]) * 7.5e22)
# L_temple = -1* np.trapz(TempleLnu_5,TempleFrequency_5)
# L_temple_attenuated = -1*np.trapz(TempleLnu_5_attenuated,TempleFrequency_5)

# arg_4000AA_Temple5 = np.argmin(np.abs(TempleWavelength_5_AA-4000))
# L_temple_4000AA = TempleLnu_5_attenuated[arg_4000AA_Temple5]

# # for gamma in [0,0.2,0.5]:
# for n0 in [1,10,100,1000]:
#     # n0 = 70
#     gamma = 0
#     model_plot = OrionLRD(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=1500, NH_target=7.5e22)
#     model_plot_Lnu = np.array([Total_Lnu_kappanuBnu_abs(nu,model_plot,gamma=gamma) for nu in RestframeFrequency])
#     model_L = -1*np.trapz(model_plot_Lnu,RestframeFrequency)
#     model_plot_Lnu = model_plot_Lnu/model_L*L_temple*3.4390405441

#     addupwave = np.logspace(np.log10(912),7,10000)
#     addupwave_fq = 3e10/(addupwave*1e-8)
#     Addup1 = np.array([np.interp(wave,TempleWavelength_5_AA,TempleLnu_5_attenuated,left=0,right=0) for wave in addupwave])
#     Addup2 = np.array([np.interp(wave*1e-4,RestframeWavelength_micron,model_plot_Lnu,left=0,right=0) for wave in addupwave])
#     Addup = Addup1+Addup2
#     # Addup2 = Addup2/L_temple_4000AA
#     Addup = Addup/L_temple_4000AA
#     # plt.plot(addupwave*1e-4,Addup2,ls = '--',label=f"n0={n0},dust only")
#     r_out = model_plot.r_out
#     plt.plot(addupwave*1e-4,Addup,label=f"$n_0$={n0},$\gamma$={gamma},$r_{{out}}$={r_out/1e3:.2f} kpc")


# plt.plot(TempleWavelength_5_AA*1e-4,TempleLnu_5/L_temple_4000AA,label='De-reddened, No Dust')

# PerazGonzalez_wavelength, PerazGonzalez_flux = np.loadtxt('analysis/data/raw/Perez-Gonzalez.csv',comments='#',delimiter=',',usecols=(0,1),unpack=True)
# ComsmosWeb_wavelength, ComsmosWeb_flux = np.loadtxt('analysis/data/raw/Cosmosweb.dat',comments='#',delimiter=' ',usecols=(0,1),unpack=True)

# arg_1 = (ComsmosWeb_wavelength<10) & (ComsmosWeb_wavelength>0.7)
# arg_2 = (ComsmosWeb_wavelength>10) | (ComsmosWeb_wavelength<0.7)
# ComsmosWeb_wavelength_1 = ComsmosWeb_wavelength[arg_1]/(1+6)
# ComsmosWeb_wavelength_2 = ComsmosWeb_wavelength[arg_2]/(1+6)

# ComsmosWeb_wavelength = ComsmosWeb_wavelength/(1+6)
# ComsmosWeb_flux_1 = ComsmosWeb_flux[arg_1]/ComsmosWeb_flux[np.argmin(np.abs(ComsmosWeb_wavelength-0.4))]
# ComsmosWeb_flux_2 = ComsmosWeb_flux[arg_2]/ComsmosWeb_flux[np.argmin(np.abs(ComsmosWeb_wavelength-0.4))]
# ComsmosWeb_flux = ComsmosWeb_flux/ComsmosWeb_flux[np.argmin(np.abs(ComsmosWeb_wavelength-0.4))]
# plt.plot(PerazGonzalez_wavelength, PerazGonzalez_flux,label='Stacked LRDs SED')
# plt.scatter(ComsmosWeb_wavelength_1,ComsmosWeb_flux_1,label='COSMOS Web')
# plt.scatter(ComsmosWeb_wavelength_2,ComsmosWeb_flux_2,label='COSMOS Web,upper bound',marker='v')

# # plt.plot(TempleWavelength_5_AA,TempleLnu_5_4000AA,label='tmpe5')
# # plt.plot(T_1e3_wavelength_micron*(1e4), Weighted_Addup_Jnu/TempleLnu_5[arg_4000AA_Temple5], label='CLOUDY Dust')
# # plt.plot(addupwave,Addup,label='Addup')
# # plt.plot(Temple5freq_Hz,TempleLnu_5,label='CLOUDY Dust')
# # plt.plot(Weighted_freq_Hz[arg_cutoff:],Weighted_Addup_Jnu[arg_cutoff:],label='Weighted Dust')
# plt.xlim(1e-1,1e3)
# plt.ylim(1e-2,1e5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Rest-frame Wavelength [$\mu$m]')
# plt.ylabel('Normalized Flux Density ($f_\\nu$/$f_\\nu$(0.4$\mu$m))')

# # Get the current axes
# ax = plt.gca()

# # Show ticks on all four axes
# ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
# # plt.ylabel('f_nu/f_nu(4000AA)')
# plt.legend(fontsize=7.7)
# plt.show()





# logratios_3_10 = []
# logratios_3_20 = []
# logratios_3_300 = []
# gammas = []
# n0s = []
# routes = []
# Touts = []
# flags = []

# for gamma < 1

# for gamma in np.linspace(0,0.6,10):
#     for n0 in np.logspace(0,2,10):
#         model = B87Model(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=T_sub, NH_target=7.5e22)
#         model_Lnu = np.array([Total_Lnu_kappanuBnu(nu,model,gamma) for nu in RestframeFrequency])
#         ratio_3_10 = np.log10(model_Lnu[Lnu_3micro_arg])-np.log10(model_Lnu[Lnu_10micro_arg])
#         ratio_3_20 = np.log10(model_Lnu[Lnu_3micro_arg])-np.log10(model_Lnu[Lnu_20micro_arg])
#         ratio_3_300 = - np.log10(model_Lnu[Lnu_3micro_arg]) + np.log10(model_Lnu[Lnu_300micro_arg])
#         gammas.append(gamma)
#         n0s.append(n0)
#         logratios_3_10.append(ratio_3_10)
#         logratios_3_20.append(ratio_3_20)
#         logratios_3_300.append(ratio_3_300)
#         routes.append(np.log10(model.r_out))
#         Touts.append(np.log10(model.T_arr[-1]))
#         if ratio_3_10 <  -1.85 and ratio_3_300 < 1.4:
#             flags.append(0)
#         else:
#             flags.append(1)

# gamma >1

# for gamma in np.linspace(1.01,2,10):
#     for n0 in np.logspace(1,4,10):
#         # if n0 < 1.9e4*(gamma-1):
#         if n0 < 6.1e3*(gamma-1):
#             continue
#         else:
#             model = B87Model(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=T_sub, NH_target=7.5e22)
#             model_Lnu = np.array([Total_Lnu_kappanuBnu(nu,model,gamma) for nu in RestframeFrequency])
#             ratio_3_10 = np.log10(model_Lnu[Lnu_3micro_arg])-np.log10(model_Lnu[Lnu_10micro_arg])
#             ratio_3_20 = np.log10(model_Lnu[Lnu_3micro_arg])-np.log10(model_Lnu[Lnu_20micro_arg])
#             ratio_3_300 = - np.log10(model_Lnu[Lnu_3micro_arg]) + np.log10(model_Lnu[Lnu_300micro_arg])
#             gammas.append(gamma)
#             n0s.append(n0)
#             logratios_3_10.append(ratio_3_10)
#             logratios_3_20.append(ratio_3_20)
#             logratios_3_300.append(ratio_3_300)
#             routes.append(np.log10(model.r_out))
#             Touts.append(np.log10(model.T_arr[-1]))
#             if ratio_3_10 <  -1.85 and ratio_3_300 < 1.4:
#                 flags.append(0)
#             else:
#                 flags.append(1)



# plt.scatter(gammas,n0s, c=logratios_3_10)
# plt.xlabel('gamma')
# plt.ylabel('n0')
# plt.yscale('log')
# # plt.xlim(1,2)
# # plt.ylim(1e1,1e4)
# plt.colorbar(label='log ratio 3micro/10micro')
# plt.title('T_sub={}'.format(T_sub))
# plt.clim(-1.85,0)
# plt.show()


# plt.scatter(gammas,n0s, c=logratios_3_20)
# plt.xlabel('gamma')
# plt.ylabel('n0')
# plt.yscale('log')
# # plt.xlim(1,2)
# # plt.ylim(1e1,1e4)
# plt.colorbar(label='log ratio 3micro/20micro')
# plt.clim(-2,0)
# plt.title('T_sub={}'.format(T_sub))
# plt.show()

# plt.scatter(gammas,n0s, c=logratios_3_300)
# plt.xlabel('gamma')
# plt.ylabel('n0')
# plt.yscale('log')
# # plt.xlim(1,2)
# # plt.ylim(1e1,1e4)
# plt.colorbar(label='log ratio 300micro/3micro')
# plt.title('T_sub={}'.format(T_sub))
# plt.clim(0,1.4)
# plt.show()

# plt.scatter(gammas,n0s, c=routes)
# plt.xlabel('gamma')
# plt.ylabel('n0')
# plt.yscale('log')
# # plt.xlim(1,2)
# # plt.ylim(1e1,1e4)
# plt.colorbar(label='log r_out')
# plt.clim(0,4)
# plt.title('T_sub={}'.format(T_sub))
# plt.show()

# plt.scatter(gammas,n0s, c=Touts)
# plt.xlabel('gamma')
# plt.ylabel('n0')
# plt.yscale('log')
# # plt.xlim(1,2)
# # plt.ylim(1e1,1e4)
# plt.colorbar(label='log T_out')
# plt.title('T_sub={}'.format(T_sub))
# plt.clim(0,3)
# plt.show()

# plt.scatter(gammas,n0s, c=flags)
# plt.xlabel('gamma')
# plt.ylabel('n0')
# plt.yscale('log')
# # plt.xlim(1,2)
# # plt.ylim(1e1,1e4)
# plt.colorbar(label='flags')
# plt.clim(0,1)
# plt.title('T_sub={}'.format(T_sub))
# plt.show()