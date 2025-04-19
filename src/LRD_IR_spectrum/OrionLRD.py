from functools import partial
from typing import override

import numpy as np
from scipy import integrate, optimize
import astropy.units as u
from astropy.units import Quantity

from .model_base import Planck_B_nu

from .utils import LogLogInterpolator, trapz_log, quad_vec_log, quad_vec_unit, quantity_to_latex
from .opacity import OpacityData
from .incident_SED import SED
from .model_base import LRD_IR_ModelBase


# Orion_opacity = OpacityData('data/Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc')

#TODO: 类的名字重新起，加注释解释这个 model 的核心原理
#FUTURE 这个 SemiOrionLRDModel 有点像个半成品。回头考虑一下有没有意义，要不要删去。
class SemiOrionLRDModel(LRD_IR_ModelBase):
    """IR 出射按照具体的 opacity 计算，而 UV 入射仍是直接给定 L_UV"""
    
    T_floor = 0.0 * u.K  # 温度的下限，低于这个温度的区域温度都设为这个值  #TODO 目前设为类的属性，以后可以考虑作为参数设定
    T_accuracy = 1.0 * u.K      #* 精确到 1 K
    
    @u.quantity_input
    def __init__(
        self,
        n_0: Quantity['number density'],
        gamma: float,
        *,   # 以下参数必须用关键字指定
        L_UV: Quantity['power'] | None,
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'] | None,
        opacity: OpacityData,
        tau_ph: float = 1.0   # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
    ):
        #? 近似：UV 波段的 sigma_H 直接取 opacity 的最大值了，这对吗？
        self.sigma_H_UV_ext = opacity.sigma_H_ext.max()
        self.sigma_H_UV_abs = opacity.sigma_H_abs.max()
        self.tau_ph = tau_ph
        super().__init__(n_0=n_0, gamma=gamma, L_UV=L_UV, T_sub=T_sub, NH_target=NH_target, opacity=opacity)

    @u.quantity_input
    def tau_UV_profile(self, r: Quantity['length']) -> Quantity['']:
        """Optical depth profile"""
        return self.sigma_H_UV_ext * self.NH_profile(r)
    
    @u.quantity_input
    def tau_UV_profile_inverse(self, tau: Quantity['']) -> Quantity[u.pc]:
        """tau_UV_profile 的逆函数。给定tau，返回r。"""
        NH = tau / self.sigma_H_UV_ext
        r = self.NH_profile_inverse(NH)
        return r
    
    @property
    @u.quantity_input
    def r_ph(self) -> Quantity[u.pc]:
        #* 这里用的是 UV band 处的 sigma_H。也就是说，求的是 UV band 处 tau == tau_ph 的 r。
        return self.tau_UV_profile_inverse(self.tau_ph)

    @u.quantity_input
    def UV_Flux(self, r: Quantity['length'], tau: Quantity[''] = None) -> Quantity[u.erg / u.s]:
        """计算方程左端，即 UV Flux。
        r 可以是标量，或者 numpy 数组。返回值与其形状一致。
        """
        if tau is None:  #* 允许指定 tau，从而可以设 tau=0 以计算无遮挡的情况，比如计算 r_in、T_ph 时。
            tau = self.tau_UV_profile(r)
        return self.L_UV * self.sigma_H_UV_abs * np.exp( -tau ) / (4 * np.pi * r**2) 
    

    @u.quantity_input
    def IR_Flux(self, T: Quantity['temperature'], method: str = 'trapz_log') -> Quantity[u.erg / u.s]:
        """计算方程右端，即 IR Flux。
        T 可以是标量，或者 numpy 数组。返回值始终为 array
        """
        nu_array = self.opacity.nu

        @u.quantity_input  #TEMP 为了速度，这里可以不做检查。而且返回值也不该转换单位
        def integrand(nu: Quantity['frequency'], T: Quantity['temperature']) -> Quantity[u.erg / u.s / u.Hz]:
            return self.opacity.interp_abs(nu) * Planck_B_nu(nu, T)

        if method == 'quad':  # 这个方法默认情况下会给出 0 值，可能是因为自适应算法没有注意到峰的位置。
            flux = quad_vec_unit(lambda nu: integrand(nu, T), nu_array.min(), nu_array.max())[0]
        elif method == 'quad_log':  #! 似乎是接近准确的，但非常慢 (单个计算 1min)
            flux = quad_vec_log(lambda nu: integrand(nu, T), nu_array.min(), nu_array.max())[0]
        elif method == 'trapz':
            flux = integrate.trapezoid(integrand(nu_array[:, None], T), nu_array, axis=0)
        elif method == 'trapz_log':
            flux = trapz_log(integrand(nu_array[:, None], T), nu_array[:, None], axis=0)  #* 由于 trapz_log 中有 f(x) * x，所以 nu_array 也要变成二维才能 broadcast
        else: 
            raise ValueError(f"method {method} is invalid! ")

        flux *= 4*np.pi

        if flux.size == 1:
            flux = flux.item()  # 把标量 array 变成标量。同时适用于 array(1) 和 array([])
        return flux


    @u.quantity_input
    def _calc_r_in(self) -> Quantity[u.pc]:
        IR_Flux = self.IR_Flux(self.T_sub)
        return np.sqrt(self.L_UV * self.sigma_H_UV_abs / (4 * np.pi * IR_Flux))
    

    @u.quantity_input
    def _T_dust_eqn(self, r: Quantity['length'], T: Quantity['temperature'], **kwargs) -> Quantity[u.erg / u.s]:
        return self.UV_Flux(r) - self.IR_Flux(T, **kwargs)

    @u.quantity_input
    def T_dust_profile_brentq(self, r: Quantity['length']) -> Quantity[u.K]:
        if isinstance(r, (int, float)) or r.size == 1:
            T_min = 0 * u.K
            T_max = self.T_sub
            return optimize.brentq(lambda T_val: self._T_dust_eqn(r, T_val * u.K).value, T_min.value, T_max.value, xtol=self.T_accuracy.value) * u.K
        else:
            return np.array([self.T_dust_profile_brentq(r_i) for r_i in r])

    @u.quantity_input
    def T_dust_profile(self, r: Quantity['length']) -> Quantity[u.K]:
        
        # 准备一个从 IR_flux 到 T 的插值表，从而加速计算
        T_array = np.linspace(1*u.K, self.T_sub, int((self.T_sub - 1*u.K) / self.T_accuracy) + 1)
        T_array_low = np.geomspace(1e-10 * u.K, 1 * u.K, 10, endpoint=False)  # 在 < 1 K 的范围，用 log 尺度取几个点，从而避免直接把下界取 0 导致 log 插值错误的问题，也避免这里在 log scale 下间隔过大。
        #FUTURE 这里的低温下界似乎太低了，造成 IR_Flux 非常小，出现 log(0)，不是很有必要。考虑改大一点
        T_array = np.concatenate([T_array_low, T_array]) 
        
        IR_Flux_array = self.IR_Flux(T_array)
        interp = LogLogInterpolator(IR_Flux_array, T_array, bounds_error=False, fill_value='extrapolate')  #* 在越界时不报错，而是外插。由于数值误差，UV_Flux(r_in) 可能会轻微地超出 IR_Flux(T_sub)。这样可以避免报错。
        
        UV_Flux = self.UV_Flux(r)  # 形状与 r 相同，可能是标量，也可能是数组。
        #* 处理 feedback 造成的效应：等效地，我们按照原来的 dust 密度、温度分布，只是在计算 r_ph 以内的 UV Flux 时，将其设为 r_ph 处的 UV Flux 值，并设 tau=0，因为 r_ph 以内不再有 dust 遮蔽。这样，这部分 dust 的辐射谱贡献就等效于 r_ph 处的 thin shell 了。
        # 取 r < r_ph 而非 <= ，这样在严格的 r = r_ph 处（thin shell 的外沿）相当于仍有 tau = 1，从而更贴近真实情况。
        UV_Flux = np.where(r < self.r_ph, self.UV_Flux(self.r_ph, tau=0), UV_Flux)   # tau = 0 是上界。若令 tau = 1 则为下界。
        
        T_dust = np.where(UV_Flux == 0, 0.0, interp(UV_Flux))  # 如果 IR_Flux 为 0，那么 T = 0，而非使用插值，因为插值可能给出 nan.
        return np.maximum(T_dust, self.T_floor)  # 低于 T_floor 的温度都设为 T_floor
    
    
class OrionLRDModel(SemiOrionLRDModel):
    
    @u.quantity_input
    def __init__(
        self,
        n_0: Quantity['number density'],
        gamma: float,
        *,  # 以下参数必须用关键字指定
        L_UV: Quantity['power'] | None = None,  #TODO 这个模型实际上不需要 L_UV 了，可以考虑删去。乃至在基类中删去。
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'] | None,
        opacity: OpacityData,
        incident_SED: SED,
        tau_ph: float = 1.0,  # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
    ):
        self.incident_SED = incident_SED
        #* 暂时继承 SemiOrionLRDModel 的 __init__ 方法，从而继承其 tau_UV_profile 所需要的 self.sigma_H_UV_ext
        super().__init__(n_0=n_0, gamma=gamma, L_UV=L_UV, T_sub=T_sub, NH_target=NH_target, opacity=opacity, tau_ph=tau_ph)
        
    @u.quantity_input
    def tau_nu_profile(self, nu: Quantity['frequency'], r: Quantity['length']) -> Quantity['']:
        """Optical depth profile"""
        return self.opacity.interp_ext(nu) * self.NH_profile(r) 
    
    @override
    @u.quantity_input
    def UV_Flux(self, r: Quantity['length'], tau: Quantity[''] = None) -> Quantity[u.erg / u.s]:
        """计算方程左端，即 UV Flux。
        r 是 Quantity。其形状可以是 scalar or 一维数组，返回值与其形状一致。
        tau 要么是一个 scalar，要么是一个二维数组，其形状 = (len(nu_array), len(r))
        """
        #TODO 整理，写注释
        r_arr = r[..., None]  # 在 r 的最后一个 axis 上添加一个维度。如果 r 是标量则转化为一维数组，r 是以为数组则转化为二维数组。
        
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array)
        
        if tau is None:
            tau = self.tau_nu_profile(nu_array, r_arr)  # tau 的形状：第 0 轴上和 nu_array 一样长，第 1 轴上和 r_arr 一样长
            
        integral = trapz_log(incident_SED.L_nu * sigma_H * np.exp(-tau), nu_array)
        
        #TEMP 之前没考虑到 nu 的升降序，临时补丁
        if np.all(integral < 0):
            integral = -integral
        return integral / (4*np.pi * r**2)
    
    @override
    @u.quantity_input
    def _calc_r_in(self) -> Quantity[u.pc]:
        IR_Flux = self.IR_Flux(self.T_sub)
        
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array)
        
        integral = trapz_log(incident_SED.L_nu * sigma_H, nu_array)
        
        #TEMP 之前没考虑到 nu 的升降序，临时补丁
        if integral < 0:
            integral = -integral
        
        return np.sqrt(integral / (4 * np.pi * IR_Flux))
    
    @override
    def _repr_latex_(self):
        fmt = partial(quantity_to_latex, p=4)
        return rf"""{self.__class__.__name__}( $n_0=$ {fmt(self.n_0)}, $\gamma={self.gamma}$, 
        $N_{{\rm H}}=$ {fmt(self.NH_target)}, $\tau_{{\rm ph}}={self.tau_ph}$, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm ph}}=$ {fmt(self.r_ph)}, $r_{{\rm out}}=$ {fmt(self.r_out)}, 
        $T_{{\rm sub}}=$ {fmt(self.T_sub)}, $T_{{\rm out}}=$ {fmt(self.T_out)} )
        """