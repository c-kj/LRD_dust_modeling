from typing import override

import numpy as np
from scipy import integrate, special
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity


from .model_base import LRD_IR_ModelBase, Planck_B_nu, R_out_Error
from .utils import LogLogInterpolator
from .opacity import OpacityData

#FUTURE 把系数对 beta （以及 q_ir）的依赖推导出来，从而让 B87 适用于任意 beta 和 power-law opacity 的情况
class B87Model(LRD_IR_ModelBase):
    """Barvainis 1987 paper 中的模型  
    假设 & 近似：见文档
    """
    
    # 类属性
    sigma_H_UV_ext: Quantity[u.cm**2] = 6.82e-23 * u.cm**2  #TEMP, from Orion sigma_H_ext maximum value，because the maximum is ~ 1 Ry = 13.6 eV, in UV range。用于计算 tau
    sigma_H_UV_abs: Quantity[u.cm**2] = 5.44e-23 * u.cm**2  # 来自 Orion 的 sigma_H_abs 最大值，因为最大值约为 1 Ry = 13.6 eV，处于 UV 能量范围内。用于计算 IR Flux
    sigma_geometric = sigma_H_UV_abs  # 几何截面 π a^2。 sigma_nu = π a^2 * Q_nu。 根据 B87 中的近似，UV 波段的 Q_nu = 1，即吸收截面近似为几何截面。
    
    # 来自 B87 文章的参数。注意如果更改 beta，q_ir 也要相应更改。
    beta = 1.6  # IR 波段 Q_nu 对 nu 的幂指数。
    q_ir = 1.4e-24  # 来自 B87 paper。是无量纲数，相当于 nu = 1 Hz 对应的 Q_nu （虽然这个 power-law 并不延伸到那么远）
    
    @u.quantity_input
    def __init__(
        self, 
        *,  # 以下参数必须用关键字指定
        n_0: Quantity['number density'],
        gamma: float,
        L_UV: Quantity['power'],  # UV 波段的功率
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'] | None,
        opacity: OpacityData,
        config: dict = {},  # 注意 config 的默认值是可变的，不要修改它！
    ):
        self.L_UV = L_UV
        super().__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, config=config)
    
    def Q_nu_abs(self, nu: Quantity['frequency']) -> np.ndarray:
        """B87 文章中 Q_nu 的近似公式：Q_nu = q_ir * nu_cgs**beta"""
        nu_cgs = nu.to_value(u.Hz)  # 转换为 cgs 单位
        return self.q_ir * nu_cgs**self.beta  # B87 中采用 cgs 单位制，所以把 B87 公式中的 nu 解释为 nu_cgs

    @staticmethod
    @u.quantity_input
    def r_in_B87(L_UV: Quantity['power'], T_sub: Quantity['temperature']) -> Quantity[u.pc]:
        L_46: float = L_UV.to_value(1e46 * u.erg / u.s)
        T_1500 = T_sub / (1500 * u.K)
        #FUTURE 这里的 -2.8 这个数字应该来自 beta 的值！需要推导
        return 1.3 * u.pc * L_46**(1/2) * T_1500**(-2.8)

    def _calc_r_in(self) -> Quantity[u.pc]:  # in [pc]
        return self.r_in_B87(self.L_UV, self.T_sub)

    @u.quantity_input
    def tau_UV_profile(self, r: Quantity['length']) -> Quantity['']:
        """Optical depth profile"""
        return self.sigma_H_UV_ext * self.NH_profile(r) 
    
    #FUTURE 实现 UV_Flux，用于和其他模型比较。其实和 SemiOrionLRDModel 是一样的，最好是能抽出来，让两个模型都采用这个函数，保证一致性。
    @u.quantity_input
    def UV_Flux(self, r: Quantity['length'], tau=None) -> Quantity[u.erg / u.s]:
        raise NotImplementedError("有待实现！")
    
    @u.quantity_input
    def IR_Flux(self, T: Quantity['temperature']) -> Quantity[u.erg / u.s]:
        from astropy.constants import c, h, k_B
        beta = self.beta
        nu_0 = 1 * u.Hz  # B87 中使用 cgs 单位制，隐含了 nu 是以 1Hz 归一化的
        
        # B87 公式 3。用的是 Q_nu 而非 sigma_nu，所以量纲是单位面积的功率
        # 这里我加入了 n_0 使得量纲平衡。B87 中的公式把 nu_0 省略了。
        IR_flux_density = 4*np.pi * self.q_ir * 2*h/c**2 * nu_0**4 * (k_B*T / (h*nu_0))**(4+beta) * special.gamma(4+beta) * special.zeta(4+beta)  
        
        return IR_flux_density * self.sigma_geometric  # 乘上截面（把 Q_nu 转换为我们用的 sigma_nu），得到 Flux，量纲为功率

    @u.quantity_input
    def T_dust_B87(self, r: Quantity['length']) -> Quantity[u.K]:
        beta = self.beta
        L_46: float = self.L_UV.to_value(1e46 * u.erg / u.s)
        r_pc: float = r.to_value(u.pc)
        tau = self.tau_UV_profile(r)
        # TODO 推导这里的系数
        #! Note that the coefficient 1650 also depends on beta (and the Q_nu relation). Will derived it by myself later.
        return 1650 * u.K * (L_46/r_pc**2)**(1/(4+beta)) * np.exp(-tau / (4+beta))

    @override
    @u.quantity_input
    def T_dust_profile(self, r: Quantity['length']) -> Quantity[u.K]:
        """Temperature profile T(r)
        """
        return self.T_dust_B87(r)
