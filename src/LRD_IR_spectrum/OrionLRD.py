from typing import override

import numpy as np
from scipy import integrate, optimize
import astropy.units as u

from .model_base import Planck_B_nu

from .utils import LogLogInterpolator, quad_vec_log, trapz_log
from .opacity import OpacityData
from .incident_SED import IncidentSED
from .model_base import LRD_IR_ModelBase


# Orion_opacity = OpacityData('data/Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc')

#TODO: 类的名字重新起，加注释解释这个 model 的核心原理
class SemiOrionLRDModel(LRD_IR_ModelBase):

    T_floor = 0.0  # 温度的下限，低于这个温度的区域温度都设为这个值  #TODO 目前设为类的属性，以后可以考虑作为参数设定
    T_accuracy = 1.0      #* 精确到 1 K
    
    def __init__(
        self,
        n_0: float,
        gamma: float,
        *,   # 以下参数必须用关键字指定
        L_UV: float,
        T_sub: float,
        NH_target: float | None,
        opacity: OpacityData,
        tau_ph = 1.0   # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
    ):
        #? 近似：UV 波段的 sigma_H 直接取 opacity 的最大值了，这对吗？
        self.sigma_H_UV_ext = opacity.sigma_H_ext.max().cgs.value
        self.sigma_H_UV_abs = opacity.sigma_H_abs.max().cgs.value
        self.tau_ph = tau_ph
        super().__init__(n_0=n_0, gamma=gamma, L_UV=L_UV, T_sub=T_sub, NH_target=NH_target, opacity=opacity)

    def tau_UV_profile(self, r):
        """Optical depth profile"""
        return self.sigma_H_UV_ext * self.NH_profile(r)
    
    def tau_UV_profile_inverse(self, tau):
        """tau_UV_profile 的逆函数。给定tau，返回r。"""
        NH = (tau / self.sigma_H_UV_ext)
        r = self.NH_profile_inverse(NH)
        return r
    
    @property
    def r_ph(self):
        #* 这里用的是 UV band 处的 sigma_H。也就是说，求的是 UV band 处 tau == tau_ph 的 r。
        return self.tau_UV_profile_inverse(self.tau_ph)

    def UV_Flux(self, r, tau=None):
        """计算方程左端，即 UV Flux。
        r 可以是标量，或者 numpy 数组。返回值与其形状一致。
        """
        if tau is None:  #* 允许指定 tau，从而可以设 tau=0 以计算无遮挡的情况，比如计算 r_in、T_ph 时。
            tau = self.tau_UV_profile(r)
        return self.L_UV * self.sigma_H_UV_abs * np.exp( -tau ) / (4 * np.pi * (r * u.pc.to(u.cm))**2) 
    

    def IR_Flux(self, T, method:str = 'trapz_log'):
        """计算方程右端，即 IR Flux。
        T 可以是标量，或者 numpy 数组。返回值始终为 array
        """
        nu_cgs = self.opacity.nu_cgs

        def integrand(nu, T):
            return self.opacity.interp_abs(nu * u.Hz).cgs.value * Planck_B_nu(nu, T)

        if method == 'quad':
            flux = integrate.quad_vec(lambda nu: integrand(nu, T), nu_cgs.min(), nu_cgs.max())[0]
        elif method == 'quad_log':
            flux = quad_vec_log(lambda nu: integrand(nu, T), nu_cgs.min(), nu_cgs.max())[0]  #? 似乎非常慢？为什么？
        elif method == 'trapz':
            flux = integrate.trapezoid(integrand(nu_cgs[:, None], T, ), nu_cgs, axis=0)
        elif method == 'trapz_log':
            flux = trapz_log(integrand(nu_cgs[:, None], T, ), nu_cgs[:, None], axis=0)  #* 由于 trapz_log 中有 f(x) * x，所以 nu_cgs 也要变成二维才能 broadcast
        else: 
            raise ValueError(f"method {method} is invalid! ")

        flux *= 4*np.pi

        if flux.size == 1:
            flux = flux.item()  # 把标量 array 变成标量。同时适用于 array(1) 和 array([])
        return flux

    def _T_dust_eqn(self, r, T, **kwargs):
        return self.UV_Flux(r) - self.IR_Flux(T, **kwargs)


    def _calc_r_in(self):
        IR_Flux = self.IR_Flux(self.T_sub)
        return np.sqrt(self.L_UV * self.sigma_H_UV_abs / (4 * np.pi * IR_Flux)) * u.cm.to(u.pc)
    

    def T_dust_profile_brentq(self, r):
        if isinstance(r, (int, float)):
            return optimize.brentq(lambda T: self._T_dust_eqn(r, T), 0, self.T_sub, xtol=self.T_accuracy)
        else:
            return np.array([self.T_dust_profile_brentq(r) for r in r])

    def T_dust_profile(self, r):
        T_array = np.linspace(1, self.T_sub, int((self.T_sub - 1) / self.T_accuracy) + 1)
        T_array_low = np.geomspace(1e-10, 1, 10, endpoint=False)   # 在 < 1 K 的范围，用 log 尺度取几个点，从而避免直接把下界取 0 导致 log 插值错误的问题，也避免这里在 log scale 下间隔过大。
        T_array = np.concatenate([T_array_low, T_array]) 
        
        IR_Flux_array = self.IR_Flux(T_array)
        interp = LogLogInterpolator(IR_Flux_array, T_array, bounds_error=False, fill_value='extrapolate')  #* 在越界时不报错，而是外插。由于数值误差，UV_Flux(r_in) 可能会轻微地超出 IR_Flux(T_sub)。这样可以避免报错。
        UV_Flux = self.UV_Flux(r)  # 形状与 r 相同，可能是标量，也可能是数组。
        #* 处理 feedback 造成的效应：等效地，我们按照原来的 dust 密度、温度分布，只是在计算 r_ph 以内的 UV Flux 时，将其设为 r_ph 处的 UV Flux 值，并设 tau=0，因为 r_ph 以内不再有 dust 遮蔽。这样，这部分 dust 的辐射谱贡献就等效于 r_ph 处的 thin shell 了。
        # 取 r < r_ph 而非 <= ，这样在严格的 r = r_ph 处（thin shell 的外沿）相当于仍有 tau = 1，从而更贴近真实情况。
        UV_Flux = np.where(r < self.r_ph, self.UV_Flux(self.r_ph, tau=0), UV_Flux)   # tau = 0 是上界。若令 tau = 1 则为下界。
        
        ret = np.where(UV_Flux == 0, 0.0, interp(UV_Flux))  # 如果 IR_Flux 为 0，那么 T = 0，而非使用插值，因为插值可能给出 nan.
        return np.maximum(ret, self.T_floor)  # 低于 T_floor 的温度都设为 T_floor
    
    
class OrionLRDModel(SemiOrionLRDModel):
    
    def __init__(
        self,
        n_0: float,
        gamma: float,
        *,  # 以下参数必须用关键字指定
        L_UV: float = None,  #TODO 这个模型实际上不需要 L_UV 了，可以考虑删去。乃至在基类中删去。
        T_sub: float,
        NH_target: float | None,
        opacity: OpacityData,
        incident_SED: IncidentSED,
        tau_ph = 1.0,  # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
    ):
        self.incident_SED = incident_SED
        #* 暂时继承 SemiOrionLRDModel 的 __init__ 方法，从而继承其 tau_UV_profile 所需要的 self.sigma_H_UV_ext
        super().__init__(n_0=n_0, gamma=gamma, L_UV=L_UV, T_sub=T_sub, NH_target=NH_target, opacity=opacity, tau_ph=tau_ph)
        
    def tau_nu_profile(self, nu, r):
        """Optical depth profile"""
        return self.opacity.interp_ext(nu).cgs.value * self.NH_profile(r) 
    
    @override
    def UV_Flux(self, r, tau=None):
        """计算方程左端，即 UV Flux。
        r 可以是标量，或者 numpy 数组。返回值与其形状一致。
        """
        #TODO 整理，写注释
        
        r = np.array(r)
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array).cgs.value
        
        r_arr = r[:, None] if np.size(r) > 1 else r
        if tau is None:
            tau = self.tau_nu_profile(nu_array, r_arr)
            
        #TEMP 目前临时处理单位的转换。由于 spectral 等效，这里必须谨慎。
        integral = trapz_log(incident_SED.L_nu.to(u.erg/u.s/u.Hz).value * sigma_H * np.exp(-tau), nu_array.cgs.value)
        # print(f"{integral=}")
        #TEMP 之前没考虑到 nu 的升降序，临时补丁
        if np.all(integral < 0):
            integral = -integral
        return integral / (4*np.pi * (r * u.pc.to(u.cm))**2)
    
    @override
    def _calc_r_in(self):
        IR_Flux = self.IR_Flux(self.T_sub)
        
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array).cgs.value
        
        #TEMP 目前临时处理单位的转换。由于 spectral 等效，这里必须谨慎。
        integral = trapz_log(incident_SED.L_nu.to(u.erg/u.s/u.Hz).value * sigma_H, nu_array.cgs.value)
        # print(f"IR_Flux: {IR_Flux}, sigma_H: {sigma_H}, nu_array: {nu_array}, incident_SED.L_nu: {incident_SED.L_nu}, {integral=}")
        
        #TEMP 之前没考虑到 nu 的升降序，临时补丁
        if integral < 0:
            integral = -integral
        
        return np.sqrt(integral / (4 * np.pi * IR_Flux)) * u.cm.to(u.pc)
    
    @override
    def _repr_latex_(self):
        return rf"""{self.__class__.__name__}($n_0=${self.n_0:.2e}, $\gamma=${self.gamma}, 
    $T_{{\rm sub}}=${self.T_sub}, 
    $r_{{\rm in}}=${self.r_in:.2e}, $r_{{\rm out}}=${self.r_out:.2e}, $T_{{\rm out}}=${self.T_out:.1f})
    """