from typing import override

import numpy as np
from scipy import integrate, special
import astropy.units as u
import astropy.constants as const


from .model_base import LRD_IR_ModelBase, Planck_B_nu, R_out_Error
from .utils import LogLogInterpolator
from .opacity import OpacityData


class B87Model(LRD_IR_ModelBase):
    sigma_H_UV_ext = 6.82e-23  #TEMP, from Orion sigma_H_ext maximum value，because the maximum is ~ 1 Ry = 13.6 eV, in UV range。用于计算 tau
    sigma_H_UV_abs = 5.44e-23  # 来自 Orion 的 sigma_H_abs 最大值，因为最大值约为 1 Ry = 13.6 eV，处于 UV 能量范围内。用于计算 IR Flux
    beta = 1.6 #TEMP

    @staticmethod
    def r_in_B87(L_UV_cgs: float, T_sub: float) -> float:
        L_46 = L_UV_cgs / 1e46
        T_1500 = T_sub / 1500
        return 1.3 * L_46**(1/2) * T_1500**(-2.8)

    def _calc_r_in(self):  # in [pc]
        return self.r_in_B87(self.L_UV, self.T_sub)

    def tau_UV_profile(self, r):
        """Optical depth profile"""
        return self.sigma_H_UV_ext * self.NH_profile(r) 

    def IR_Flux(self, T):
        c = const.c.cgs.value
        h = const.h.cgs.value
        k_B = const.k_B.cgs.value
        q_ir = 1.4e-24  #* 直接取 B87 所用的值
        beta = self.beta
        sigma_H_ir = q_ir * self.sigma_H_UV_abs

        return 4*np.pi * sigma_H_ir * 2*h/c**2 * (k_B * T/h)**(4+beta) * special.gamma(4+beta) * special.zeta(4+beta)

    def T_dust_B87(self, r):
        beta = self.beta
        L_46 = self.L_UV / 1e46
        tau = self.tau_UV_profile(r)
        #! Note that the coefficient 1650 also depends on beta (and the Q_nu relation). Will derived it by myself later.
        return 1650 * (L_46/r**2)**(1/(4+beta)) * np.exp(-tau / (4+beta))  # in [K]

    def T_dust_profile(self, r):
        """Temperature profile T(r)
        """
        return self.T_dust_B87(r)
