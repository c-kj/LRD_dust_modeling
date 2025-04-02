import numpy as np
# from scipy import integrate
import matplotlib.pyplot as plt
import astropy.units as u
import os

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
        self.r_in = self.calc_r_in()  # in [pc]
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
CLOUDY_Path = '/Users/chenkejian/Library/Mobile Documents/com~apple~CloudDocs/北大/科研/JWST_LRD_SED/IR_spectrum/'

OrionFile = 'Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc'

OrionOpacityFile = os.path.join(CLOUDY_Path,OrionFile)
OrionWavelength_Rydberg , Orion_Opacity_SigmaPerH = np.loadtxt(OrionOpacityFile,comments='#', delimiter='\t', usecols=(0, 1),unpack=True)


OrionWavelength_micro = 911.27/OrionWavelength_Rydberg*1e-4 # micron
OrionWavelength_x = 1/OrionWavelength_micro # rising order
# print(np.all(np.diff(OrionWavelength_x)> 0))

# Define functions
def Lnu_kappanuBnu(nu,T):
    if 4.8043478261E-11*nu/(T) > 100:
        return 0
    if 4.8043478261E-11*nu/(T) < 1e-3:
        return 2*nu**2*1.38e-16*T/(3e10)**2*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH)
    else:
        return 2*6.63e-27*nu**3/(3e10)**2/(np.exp(6.63e-27*nu/(1.38e-16*T))-1)*np.interp(1/(3e10/nu*1e4),OrionWavelength_x,Orion_Opacity_SigmaPerH)

def Total_Lnu_kappanuBnu(nu,model,gamma):
    # return np.trapz([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)],model.r_arr)
    # integrate in log log space
    # return np.trapz(np.exp(np.log(np.array([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)]))+np.log(model.r_arr)),np.log(model.r_arr))
    return np.trapz([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)],model.r_arr)

def ToBeIntegrated(nu,model,gamma):
    return np.array([Lnu_kappanuBnu(nu,T)*r**(2-gamma) for (T,r) in zip(model.T_arr,model.r_arr)])

RestframeWavelength_micron = np.logspace(-0.5,3.5,200) # micron
RestframeFrequency = 3e10/(RestframeWavelength_micron*1e-4) # Hz
Lnu_3micro_arg = np.argmin(np.abs(RestframeWavelength_micron-3))
Lnu_10micro_arg = np.argmin(np.abs(RestframeWavelength_micron-10))
Lnu_20micro_arg = np.argmin(np.abs(RestframeWavelength_micron-20))
Lnu_300micro_arg = np.argmin(np.abs(RestframeWavelength_micron-300))
T_sub = 1500

def plot_model(gamma,n0):
    _model = B87Model(n_0=n0, gamma=gamma, L_UV=1e46, T_sub=T_sub, NH_target=7.5e22)
    _model_Lnu = np.array([Total_Lnu_kappanuBnu(nu,_model,gamma) for nu in RestframeFrequency])
    _ratio_3_10 = np.log10(_model_Lnu[Lnu_3micro_arg])-np.log10(_model_Lnu[Lnu_10micro_arg])
    _ratio_3_20 = np.log10(_model_Lnu[Lnu_3micro_arg])-np.log10(_model_Lnu[Lnu_20micro_arg])
    _ratio_3_300 = np.log10(_model_Lnu[Lnu_300micro_arg]) - np.log10(_model_Lnu[Lnu_3micro_arg])
    plt.plot(RestframeWavelength_micron,_model_Lnu,label=f'n0={n0}, gamma={gamma},T_sub={T_sub},\nratio_3_10={_ratio_3_10:.2f}, ratio_3_20={_ratio_3_20:.2f}, ratio_3_300={_ratio_3_300:.2f},\nr_out={_model.r_out/1e3:.2f} kpc, T_out={_model.T_arr[-1]:.2f} K')
    plt.scatter([3,10,20,300],_model_Lnu[[Lnu_3micro_arg,Lnu_10micro_arg,Lnu_20micro_arg,Lnu_300micro_arg]],marker='x')
    plt.text(3,_model_Lnu[Lnu_3micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_3micro_arg])))
    plt.text(10,_model_Lnu[Lnu_10micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_10micro_arg])))
    plt.text(20,_model_Lnu[Lnu_20micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_20micro_arg])))
    plt.text(300,_model_Lnu[Lnu_300micro_arg],'{:.1f}'.format(np.log10(_model_Lnu[Lnu_300micro_arg])))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylim(1e-34,1e-26)
    
    fig, ax = plt.gcf(), plt.gca()
    plt.show()
    return fig, ax

# model0 = B87Model(n_0=1e1, gamma=0, L_UV=1e46, T_sub=1000, NH_target=7.5e22)
plot_model(gamma=0.4,n0=40)

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








logratios_3_10 = []
logratios_3_20 = []
logratios_3_300 = []
gammas = []
n0s = []
routes = []
Touts = []
flags = []

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