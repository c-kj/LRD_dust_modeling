from abc import abstractmethod
from functools import partial
from typing import override

import numpy as np
from scipy import integrate, optimize
import astropy.units as u
from astropy.units import Quantity

from .model_base import Planck_B_nu

from .utils import LogLogInterpolator, trapz_log, quad_vec_log, quad_vec_unit, trapz_mapping, quantity_to_latex
from .opacity import OpacityData
from .incident_SED import SED
from .model_base import LRD_IR_ModelBase


class EnergyBalanceModel(LRD_IR_ModelBase):
    """根据 吸收-再发射能量平衡 计算 T_dust_profile 的模型。并考虑 feedback 效应。  
    
    抽象方法：
    - UV_Flux 的计算 (w/o feedback)
    - sigma_H_Prad_eff 的计算
    
    具体实现了：
    - r_ph 的计算（使用 等效辐射压截面 sigma_H_Prad_eff），用于计算 feedback 效应的影响
    - IR_Flux 的计算（都是根据黑体谱 & opacity 积分，很统一）
    - 根据 UV_Flux == IR_Flux 计算 T_dust_profile 的方法（反插表 or brentq）
    - 其他工具方法：
        - 用于检验 UV_Flux 和 IR_Flux 总功率的方法
    """
    
    config = LRD_IR_ModelBase.config | {  # 从父类的 config 上拓展
        'IR_Flux.integrator': 'trapz_log',  # IR_Flux 的积分方法
        'T_floor': 0.0 * u.K,  # 温度的下限，低于这个温度的区域温度都设为这个值
        'T_accuracy': 1.0 * u.K,  # 温度精确到 1 K。这决定了 T_dust_profile 中反插表的间距，或者 brentq 的 xtol
    }
    
    @u.quantity_input
    def __init__(
        self,
        *,   # 以下参数必须用关键字指定
        n_0: Quantity['number density'],
        gamma: float,
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'],
        opacity: OpacityData,
        tau_ph: float = 1.0,   # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
        config: dict = {},
    ):
        self.tau_ph = tau_ph
        super().__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, config=config)
    
    
    @property
    @abstractmethod
    def sigma_H_Prad_eff(self) -> Quantity[u.cm**2]:
        """等效辐射压截面"""
        pass
    
    def tau_Prad_profile_inverse(self, tau: Quantity['']) -> Quantity[u.pc]:
        """tau_Prad_profile 的逆函数。给定tau，返回r。用于计算 r_ph。  
        tau_Prad_profile 是对应于 等效辐射压截面 (sigma_H_Prad_eff) 的光深
        """
        NH: Quantity = tau / self.sigma_H_Prad_eff
        r = self.NH_profile_inverse(NH)
        return r
    
    @property
    @u.quantity_input
    def r_ph(self) -> Quantity[u.pc]:
        # 使用 等效辐射压截面 计算 tau，找到 tau == tau_ph 处的 r
        return self.tau_Prad_profile_inverse(self.tau_ph)
    
    @override
    @u.quantity_input
    def r_with_feedback(self, r: Quantity['length']) -> Quantity[u.pc]:
        """考虑 feedback 反馈效果时，从尘埃原位置 r 到新位置 r' 的映射。  
        类似于流体的 Lagrangian 坐标。  
        
        输入：r 是某个尘埃颗粒原先（无反馈）时的位置。  
        返回：r' 是考虑 feedback 后，尘埃所处的新位置。  
        r 可以是标量，也可以是数组。返回值与其形状相同。
        """
        return np.maximum(r, self.r_ph)  # 目前对 feedback 的处理：原先在 r_ph 以内的尘埃都会被扫到 r_ph 处堆积，而其余位置不变。

    @u.quantity_input
    def IR_Flux(self, T: Quantity['temperature']) -> Quantity[u.erg / u.s]:
        """计算方程右端的 IR Flux，即尘埃颗粒再发射的能流通量。  
        T 可以是标量，或者 numpy 数组。返回值始终为 array
        """
        nu_array = self.opacity.nu

        @u.quantity_input  #TEMP 为了速度，这里可以不做检查。而且返回值也不该转换单位
        def integrand(nu: Quantity['frequency'], T: Quantity['temperature']) -> Quantity[u.erg / u.s / u.Hz]:
            return self.opacity.interp_abs(nu) * Planck_B_nu(nu, T)

        method: str = self.config['IR_Flux.integrator']  # 这里用 trapz 或 trapz_log 区别很小。因为 nu_array 来自 opacity，足够密集。
        if method == 'quad':  # 这个方法默认情况下会给出 0 值，可能是因为自适应算法没有注意到峰的位置。
            flux = quad_vec_unit(lambda nu: integrand(nu, T), nu_array.min(), nu_array.max())[0]
        elif method == 'quad_log':  #! 似乎是接近准确的，但非常慢 (单个计算 1min)
            flux = quad_vec_log(lambda nu: integrand(nu, T), nu_array.min(), nu_array.max())[0]
        elif method == 'trapz':
            flux = integrate.trapezoid(integrand(nu_array[:, None], T), nu_array, axis=0)
        elif method == 'trapz_log':
            flux = trapz_log(integrand(nu_array[:, None], T), nu_array[:, None], axis=0)  #* 由于 trapz_log 中有 f(x) * x，所以 nu_array 也要变成二维才能 broadcast
        else: 
            raise ValueError(f"method {method = } is invalid! ")

        flux *= 4*np.pi

        if flux.size == 1:
            flux = flux.item()  # 把标量 array 变成标量。同时适用于 array(1) 和 array([])
        return flux

    @abstractmethod
    def UV_Flux(self, r: Quantity['length'], tau: Quantity[''] = None) -> Quantity[u.erg / u.s]:
        pass
    
    @abstractmethod
    def UV_Flux_with_feedback(self, r: Quantity['length']) -> Quantity[u.erg / u.s]:
        pass
    

    @u.quantity_input
    def _T_dust_eqn(self, r: Quantity['length'], T: Quantity['temperature'], **kwargs) -> Quantity[u.erg / u.s]:
        return self.UV_Flux_with_feedback(r) - self.IR_Flux(T, **kwargs)

    @u.quantity_input
    def T_dust_profile_brentq(self, r: Quantity['length']) -> Quantity[u.K]:
        if isinstance(r, (int, float)) or r.size == 1:
            T_min = 0 * u.K
            T_max = self.T_sub
            return optimize.brentq(lambda T_val: self._T_dust_eqn(r, T_val * u.K).value, T_min.value, T_max.value, xtol=self.config['T_accuracy'].value) * u.K
        else:
            return np.array([self.T_dust_profile_brentq(r_i) for r_i in r])

    @u.quantity_input
    def T_dust_profile(self, r: Quantity['length']) -> Quantity[u.K]:
        
        # 准备一个从 IR_flux 到 T 的插值表，从而加速计算
        T_array = np.linspace(1*u.K, self.T_sub, int((self.T_sub - 1*u.K) / self.config['T_accuracy']) + 1)
        T_array_low = np.geomspace(1e-10 * u.K, 1 * u.K, 10, endpoint=False)  # 在 < 1 K 的范围，用 log 尺度取几个点，从而避免直接把下界取 0 导致 log 插值错误的问题，也避免这里在 log scale 下间隔过大。
        #FUTURE 这里的低温下界似乎太低了，造成 IR_Flux 非常小，出现 log(0)，不是很有必要。考虑改大一点
        T_array = np.concatenate([T_array_low, T_array]) 
        
        IR_Flux_array = self.IR_Flux(T_array)
        interp = LogLogInterpolator(IR_Flux_array, T_array, bounds_error=False, fill_value='extrapolate')  #* 在越界时不报错，而是外插。由于数值误差，UV_Flux(r_in) 可能会轻微地超出 IR_Flux(T_sub)。这样可以避免报错。
        
        UV_Flux = self.UV_Flux_with_feedback(r) # 考虑了 feedback 的效应，对 UV_Flux 做了修正。 # 形状与 r 相同，可能是标量，也可能是数组。
        
        T_dust = np.where(UV_Flux == 0, 0.0, interp(UV_Flux))  # 如果 IR_Flux 为 0，那么 T = 0，而非使用插值，因为插值可能给出 nan.
        return np.maximum(T_dust, self.config['T_floor'])  # 低于 T_floor 的温度都设为 T_floor
    
    
    # 用于计算吸收总功率 L 的方法
    @u.quantity_input
    def calc_L_from_UV_Flux(self) -> Quantity[u.erg / u.s]:
        """从 UV_Flux 计算 dust 吸收的总功率，但不考虑 feedback 效应"""
        r_array = self.get_r_array()
        return trapz_log(self.UV_Flux(r_array) * self.n_profile(r_array) * 4*np.pi * r_array**2, r_array)
    
    @u.quantity_input
    def calc_L_from_UV_Flux_with_feedback(self) -> Quantity[u.erg / u.s]:
        """从 UV_Flux 计算 dust 吸收的总功率，考虑 feedback 效应"""
        r_array = self.get_r_array()
        # 要把 4πr^2 中的 r 改为 dust 颗粒实际所在的位置，也即 r_with_feedback。而 T 和 n 的 profile 中所使用的仍是原先的 r。
        r_with_feedback = self.r_with_feedback(r_array)
        return trapz_log(self.UV_Flux_with_feedback(r_array) * self.n_profile(r_array) * 4*np.pi * r_with_feedback**2, r_array)
    
    # 用于计算发射总功率 L 的方法
    @u.quantity_input
    def calc_L_from_IR_Flux(self) -> Quantity[u.erg / u.s]:
        """从 IR_Flux 计算 dust 发射的总功率"""
        r_array = self.get_r_array()
        # 要把 4πr^2 中的 r 改为 dust 颗粒实际所在的位置，也即 r_with_feedback。而 T 和 n 的 profile 中所使用的仍是原先的 r。
        r_with_feedback = self.r_with_feedback(r_array)
        return trapz_log(self.IR_Flux(self.T_dust_profile(r_array)) * self.n_profile(r_array) * 4*np.pi * r_with_feedback**2, r_array)


class L_UV_Model(EnergyBalanceModel):
    """UV_Flux 通过 L_UV 直接给出，不考虑入射光谱形状
    
    将入射光谱看作 UV 波段的常数 or delta function。  
    相应地使用该波段的等效 opacity 计算 extinction、absorption 和 radiation pressure。
    """
    
    @u.quantity_input
    def __init__(
        self,
        *,   # 以下参数必须用关键字指定
        n_0: Quantity['number density'],
        gamma: float,
        L_UV: Quantity['power'],
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'],
        opacity: OpacityData,
        tau_ph: float = 1.0,   # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
        config: dict = {},
    ):
        #? 近似：UV 波段的 sigma_H 直接取 opacity 的最大值了，这对吗？
        self.sigma_H_UV_ext = opacity.sigma_H_ext.max()  # 用于 tau_UV_profile 和其逆
        self.sigma_H_UV_abs = opacity.sigma_H_abs.max()  # 用于 UV_Flux 和 _calc_r_in
        
        self.L_UV = L_UV
        super().__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, tau_ph=tau_ph, config=config)
    
    
    @u.quantity_input
    def tau_UV_profile(self, r: Quantity['length']) -> Quantity['']:
        """UV 波段光深作为 r 的函数。
        用于计算 UV_Flux 中的指数衰减。
        """
        return self.sigma_H_UV_ext * self.NH_profile(r)
    
    @override
    @property
    def sigma_H_Prad_eff(self) -> Quantity[u.cm**2]:
        """等效辐射压截面"""
        #* 这里用的是 UV band 处的 sigma_H。也就是说，r_ph 求的是使得 tau_UV_ext(r) == tau_ph 的 r。
        return self.sigma_H_UV_ext

    @override
    @u.quantity_input
    def UV_Flux(self, r: Quantity['length'], tau: Quantity[''] = None) -> Quantity[u.erg / u.s]:
        """计算方程左端的 UV Flux，即尘埃颗粒吸收的能流通量。  
        r 可以是标量，或者 numpy 数组。返回值与其形状一致。
        """
        if tau is None:  #* 允许指定 tau，从而可以设 tau=0 以计算无遮挡的情况，比如计算 r_in、T_ph 时。
            tau = self.tau_UV_profile(r)
        return self.L_UV * self.sigma_H_UV_abs * np.exp( -tau ) / (4 * np.pi * r**2) 
    
    @override
    @u.quantity_input
    def UV_Flux_with_feedback(self, r: Quantity['length']) -> Quantity[u.erg / u.s]:
        """计算方程左端的 UV Flux，即尘埃颗粒吸收的能流通量。考虑 feedback 效应。  
        目前对 feedback 的近似处理是：假定 feedback 把 dust 扫到 r_ph 处，堆积成一个 thin shell。每个尘埃颗粒所处的光深 tau 不变，只是位置（1/4πr^2 中的 r）变为 r_ph。  
        当 tau_ph 设为 0 时，r_ph == r_in，回退到没有 feedback 的情况。  
        r 可以是标量，或者 numpy 数组。返回值与其形状一致。
        """
        # 处理 feedback 造成的效应：将 1/4πr^2 中的 r 使用 feedback 调整后的值，而 tau 则使用原有的 r。相当于只在 r<r_ph 的地方把 1/4πr^2 中的 r 替换为 r_ph。
        return self.UV_Flux(self.r_with_feedback(r), tau=self.tau_UV_profile(r))
    
    @override
    @u.quantity_input
    def _calc_r_in(self) -> Quantity[u.pc]:
        IR_Flux = self.IR_Flux(self.T_sub)
        return np.sqrt(self.L_UV * self.sigma_H_UV_abs / (4 * np.pi * IR_Flux))



class OrionLRDModel(SemiOrionLRDModel):
    
    config = SemiOrionLRDModel.config | {  # 从父类的 config 上拓展


class OrionLRDModel(EnergyBalanceModel):
    """使用入射光谱 incident_SED 计算 UV_Flux 并考虑 feedback 效应。
    
    区分不同 nu 上的 tau。
    """

    config = EnergyBalanceModel.config | {  # 从父类的 config 上拓展
        'UV_Flux.integrator': 'trapz_log',  # UV_Flux 的积分方法
    }

    @u.quantity_input
    def __init__(
        self,
        *,  # 以下参数必须用关键字指定
        n_0: Quantity['number density'],
        gamma: float,
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'],
        opacity: OpacityData,
        incident_SED: SED,
        tau_ph: float = 1.0,  # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
        config: dict = {},
    ):
        self.incident_SED = incident_SED
        # * 暂时继承 SemiOrionLRDModel 的 __init__ 方法，从而继承其 tau_UV_profile 所需要的 self.sigma_H_UV_ext
        super().__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, tau_ph=tau_ph, config=config)

    @u.quantity_input
    def tau_nu_profile(self, nu: Quantity['frequency'], r: Quantity['length']) -> Quantity['']:
        """光深 tau_nu 作为 r 的函数。  
        用于计算 UV_Flux 中的指数衰减。
        """
        return self.opacity.interp_ext(nu) * self.NH_profile(r) 

    @override
    @property
    def sigma_H_Prad_eff(self) -> Quantity[u.cm**2]:
        """等效辐射压截面  
        计算方法由 config['sigma_H_Prad_eff.method'] 控制。
        """
        return self.opacity.sigma_H_ext.max()  #TEMP 与之前的处理保持一致，取截面最大值。即将更改为可以使用 incident SED 加权平均
    
    @override
    @u.quantity_input
    def UV_Flux(self, r: Quantity['length'], tau: Quantity[''] = None) -> Quantity[u.erg / u.s]:
        """计算方程左端的 UV Flux，即尘埃颗粒吸收的能流通量。  
        r 是 Quantity。其形状可以是 scalar or 一维数组，返回值与其形状一致。
        tau 要么是一个 scalar，要么是一个二维数组，其形状 = (len(nu_array), len(r))
        """
        r_arr = r[..., None]  # 在 r 的最后一个 axis 上添加一个维度。如果 r 是标量则转化为一维数组，r 是以为数组则转化为二维数组。
        
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu  # 采用 incident_SED 的 nu，因为它的范围比 opacity 更窄。#* 如果担心采样点不够密集，应当在传入时调用 SED.refine 方法。
        sigma_H = self.opacity.interp_abs(nu_array)
        
        if tau is None:
            tau = self.tau_nu_profile(nu_array, r_arr)  # tau 的形状：第 0 轴上和 nu_array 一样长，第 1 轴上和 r_arr 一样长
            
        integrator = trapz_mapping[self.config['UV_Flux.integrator']]
        integral = integrator(incident_SED.L_nu * sigma_H * np.exp(-tau), nu_array)
        
        return integral / (4*np.pi * r**2)
    
    @override
    @u.quantity_input
    def UV_Flux_with_feedback(self, r: Quantity['length']) -> Quantity[u.erg / u.s]:
        """计算方程左端的 UV Flux，即尘埃颗粒吸收的能流通量。考虑 feedback 效应。  
        目前对 feedback 的近似处理是：假定 feedback 把 dust 扫到 r_ph 处，堆积成一个 thin shell。每个尘埃颗粒所处的光深 tau 不变，只是位置（1/4πr^2 中的 r）变为 r_ph。  
        当 tau_ph 设为 0 时，r_ph == r_in，回退到没有 feedback 的情况。  
        r 可以是标量，或者 numpy 数组。返回值与其形状一致。
        """
        r_arr: Quantity = r[..., None]             # 在 r 的最后一个 axis 上添加一个维度。如果 r 是标量则转化为一维数组，r 是以为数组则转化为二维数组。
        nu_array: Quantity = self.incident_SED.nu  # 与 UV_Flux 中使用的 nu_array 保持一致，这样 tau 的形状才能一致。
        tau = self.tau_nu_profile(nu_array, r_arr)
        # 处理 feedback 造成的效应：将 1/4πr^2 中的 r 使用 feedback 调整后的值，而 tau 则使用原有的 r。相当于只在 r<r_ph 的地方把 1/4πr^2 中的 r 替换为 r_ph。
        return self.UV_Flux(self.r_with_feedback(r), tau=tau)
    
    @override
    @u.quantity_input
    def _calc_r_in(self) -> Quantity[u.pc]:
        """计算 r_in。  
        来自于在 r == r_in 处的能量平衡 UV_Flux == IR_Flux，而 UV_Flux 由于 tau == 0 所以随 r 是平方反比的，可以直接从中解出 r_in 的值。  
        目前是仿照 UV_Flux 单独写了一遍。也许可以改为调用 UV_Flux 方法，从而确保一致性。
        """
        IR_Flux = self.IR_Flux(self.T_sub)
        
        incident_SED = self.incident_SED
        nu_array = incident_SED.nu
        sigma_H = self.opacity.interp_abs(nu_array)
        
        integrator = trapz_mapping[self.config['UV_Flux.integrator']]
        integral = integrator(incident_SED.L_nu * sigma_H, nu_array)
        
        return np.sqrt(integral / (4 * np.pi * IR_Flux))
    
    @override
    def _repr_latex_(self):
        fmt = partial(quantity_to_latex, p=4)
        return rf"""{self.__class__.__name__}( $n_0=$ {fmt(self.n_0)}, $\gamma={self.gamma}$, 
        $N_{{\rm H}}=$ {fmt(self.NH_target)}, $\tau_{{\rm ph}}={self.tau_ph}$, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm ph}}=$ {fmt(self.r_ph)}, $r_{{\rm out}}=$ {fmt(self.r_out)}, 
        $T_{{\rm sub}}=$ {fmt(self.T_sub)}, $T_{{\rm out}}=$ {fmt(self.T_out)} )
        """