from abc import ABC, abstractmethod
from functools import partial
import logging

import numpy as np
from scipy import integrate, special
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity

from .utils import trapz_log, quad_vec_log, quad_vec_unit, quantity_to_latex
from .opacity import OpacityData

class R_out_Error(Exception):
    pass

@u.quantity_input
def Planck_B_nu(nu: Quantity['frequency'], T: Quantity['temperature']) -> Quantity[u.erg / u.s / u.cm**2 / u.Hz]:
    """Planck B_nu function. Returns spectral radiance with proper units.
    """
    from astropy.constants import c, h, k_B
    x = h * nu / (k_B * T)
    return (2 * h / c**2) * nu**3 / (np.exp(x) - 1)

#TODO 重新考虑 抽象基类 该怎么写，最好只提供接口。实现放到 mixin？
class LRD_IR_ModelBase(ABC):
    paras_name = ['n_0', 'gamma', 'L_UV', 'T_sub']

    @u.quantity_input
    def __init__(
        self,
        n_0: Quantity['number density'],
        gamma: float,
        *,  # 以下参数必须用关键字指定
        L_UV: Quantity['power'] | None,
        T_sub: Quantity['temperature'],
        NH_target: Quantity['column density'] | None,
        opacity: OpacityData,  #TODO 这里似乎不应该要求输入 opacity。L_UV 也应该重新考虑
    ):
        """
        Initialize the model with these parameters. The units are in cgs unless specified otherwise.
        r_in is in [pc]
        
        These input paras should NOT be changed after initialization
        """
        self.n_0 = n_0
        self.gamma = gamma
        self.L_UV = L_UV
        self.T_sub = T_sub
        self.NH_target = NH_target
        self.opacity = opacity
        # r_in 目前在初始化时计算。子类应当覆写 _calc_r_in 方法。
        self.r_in = self._calc_r_in()   # in [pc]  #* 可以在运行时手动修改 r_in
    

    @property
    def paras(self):
        """提取这个模型所有必要的参数。可用于生成新的模型、做 Hash 等。子类可以覆写类属性 paras_name。"""
        return {self.__dict__[name] for name in self.paras_name}

    # 目前暂时废弃 __repr__，避免维护成本。反正 Jupyter notebook 里显示用的也是 latex 格式
    # def __repr__(self):
    #     return f"{self.__class__.__name__}(n_0={self.n_0}, gamma={self.gamma}, L_UV={self.L_UV}, T_sub={self.T_sub}, r_in={self.r_in})"

    def _repr_latex_(self):
        """在 Jupyter Notebook 的单元输出中渲染时所调用的方法"""
        fmt = partial(quantity_to_latex, p=4)
        
        return rf"""{self.__class__.__name__}($n_0=$ {fmt(self.n_0)} , $\gamma={self.gamma}$, 
        $L_{{\rm UV}}=$ {fmt(self.L_UV)} , $T_{{\rm sub}}=$ {fmt(self.T_sub)}, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm out}}=$ {fmt(self.r_out)} , $T_{{\rm out}}=$ {fmt(self.T_out)})
        """

    def __format__(self, format_spec: str) -> str:
        """usage: ` f"{model:latex}" `
        """
        if format_spec == 'latex':
            return self._repr_latex_()
        else:
            return super().__format__(format_spec)

    # @staticmethod
    # def r_in_Kohei(L_UV_cgs, T_sub):
    #     """Kohei 的 Fortran 程序中所使用的计算 r_in 的公式。不太懂为啥，可能是错的，以后废弃。"""
    #     sigma_SB = const.sigma_sb.cgs.value
    #     return np.sqrt(L_UV_cgs/(16.0e0*np.pi*sigma_SB)/(T_sub**4.0e0)) * (u.cm.to(u.pc))
    

    @abstractmethod  # 抽象方法：子类必须实现这个方法
    def _calc_r_in(self):  # in [pc]
        pass

    @u.quantity_input
    def n_profile(self, r: Quantity['length']) -> Quantity[u.cm**-3]:
        r_ratio = r / self.r_in
        return self.n_0 * r_ratio**(-self.gamma)

    @u.quantity_input
    def NH_profile(self, r: Quantity['length']) -> Quantity[u.cm**-2]:
        """Column density profile in cgs"""
        gamma = self.gamma  # for brevity
        r_ratio: Quantity[''] = r / self.r_in
        if gamma == 1:  # special case: gamma == 1
            factor = np.log(r_ratio)
        else:
            factor = (r_ratio**(1 - gamma) - 1) / (1 - gamma)
        return self.n_0 * self.r_in * factor
    
    @u.quantity_input
    def NH_profile_inverse(self, NH: Quantity['column density']) -> Quantity[u.pc]:
        """Inverse function of NH_profile. For given NH, find r.
        
        NH in cgs unit, r in pc.
        """
        gamma = self.gamma
        factor: Quantity[''] = NH / (self.n_0 * self.r_in)
        #* 如果 gamma > 1，那么 NH 的最大值是 n_0 * r_in / (gamma-1)。如果 NH 大于这个值，那么 r 无法找到。
        if gamma > 1 and factor > 1/(gamma-1):
            raise R_out_Error(f"The {NH = } is larger than possible in this NH_profile with {gamma = }, cannot find r")  # raise error，留给外部处理。
        
        if gamma == 1:
            r_ratio = np.exp(factor)
        else:
            r_ratio = (factor * (1-gamma) + 1) ** (1/(1-gamma))
        return self.r_in * r_ratio
    
    
    # r_out 的处理：调用 _calc_r_out() 计算并缓存结果。可以手动通过 self.r_out = value 来修改其值。可以通过 del self.r_out 来删除缓存，下次调用时重新计算。
    @property
    def r_out(self):
        if not hasattr(self, '_r_out'):
            self._r_out = self._calc_r_out()
        return self._r_out
    
    @r_out.setter
    def r_out(self, value):
        self._r_out = value
    
    @r_out.deleter
    def r_out(self):
        del self._r_out

    def _calc_r_out(self):
        """calc the r_out that could give the specified NH"""
        NH_target = self.NH_target
        if NH_target is None:
            raise ValueError("NH_target is not specified yet!")
        r_out = self.NH_profile_inverse(NH_target)
        
        if np.isinf(r_out):        #! 注意，在极端参数下，r_out 可能超过浮点数上界，变为 inf。
            logging.warning("r_out is inf due to float overflow.")
            
        return r_out


    @u.quantity_input
    def T_dust_power_law(self, p: float, r: Quantity['length']) -> Quantity[u.K]:
        return self.T_sub * (r / self.r_in) ** (-p)

    @abstractmethod
    def T_dust_profile(self, r: Quantity['length']) -> Quantity[u.K]:
        """Temperature profile T(r)"""
        pass

    @property
    def T_in(self) -> Quantity[u.K]:
        return self.T_dust_profile(self.r_in)

    @property
    def T_out(self) -> Quantity[u.K]:
        return self.T_dust_profile(self.r_out)

    method_L_nu = 'trapz_log'  # 默认方法是 'trapz_log'，因为 trapz 要显著快于 quad，而 log scale 收敛性显著更好。
    @u.quantity_input
    def calc_L_nu(
        self,
        nu_array: Quantity['frequency'] | None = None,
        r_sample_num: int = 100,
    ) -> Quantity[u.erg / u.s / u.Hz]:
        """calc L_nu at given frequency array, using the given opacity data

        Parameters
        ----------
        nu_array : Quantity['frequency'] | None, optional
            If not specified (default = None), will use opacity.nu.
            可以是任何可以转换为频率的数组，比如波长数组，但是要通过 equivalencies 转换为频率单位。

        Returns
        -------
        L_nu : Quantity[u.erg / u.s / u.Hz]
            对应于 nu_array 的 L_nu。
        """
        if nu_array is None:  # if nu_array is not given, use the nu array in opacity
            nu_array = self.opacity.nu
            sigma_H = self.opacity.sigma_H_abs  # then, sigma_H directly from data, don't need interpolation
        else:
            nu_array = nu_array.to(u.Hz, copy=False)  # 转换为频率单位
            sigma_H = self.opacity.interp_abs(nu_array)

        # 被积函数
        @u.quantity_input #TODO 为了速度，这里可以不做检查。而且返回值也不该转换单位
        def func(nu: Quantity['frequency'], r: Quantity['length']) -> Quantity[u.erg / u.s / u.Hz / u.cm**3]:
            return Planck_B_nu(nu, self.T_dust_profile(r)) * self.n_profile(r) * 4 * np.pi * r**2  # 这里的 4pi 是 dV = 4 pi r^2 dr 的系数

        # * 主要是采样方式 (log / linear) 对积分的收敛性影响较大。积分是用 trapz 还是 trapz_log 影响较小。
        if self.method_L_nu == 'quad':
            L_nu = quad_vec_unit(lambda r: func(nu_array, r), self.r_in, self.r_out)[0]
        elif self.method_L_nu == 'quad_log':
            L_nu = quad_vec_log(lambda r: func(nu_array, r), self.r_in, self.r_out)[0]
        elif self.method_L_nu == 'trapz':
            r_array = np.linspace(self.r_in, self.r_out, r_sample_num)   # 实际上，linspace 采样是很不合适的，收敛性很差。
            integrand_array = func(nu_array[:, None], r_array)
            L_nu = integrate.trapezoid(integrand_array, r_array)
        elif self.method_L_nu == 'trapz_log':
            r_array = np.geomspace(self.r_in, self.r_out, r_sample_num)  # sample in log scale
            integrand_array = func(nu_array[:, None], r_array)
            L_nu = trapz_log(integrand_array, r_array)
        else:
            raise ValueError(f"method {self.method_L_nu = } is invalid! ")

        L_nu *= 4 * np.pi * sigma_H   # 最后，乘上积分前面统一的 factor。sigma_H 这个关于 nu 的数组是提到积分外面来的。这里的 4pi 是出射立体角 Omega

        return L_nu
    
    @u.quantity_input
    def calc_nu_L_nu(
        self,
        nu_array: Quantity['frequency'] | None = None,
        r_sample_num: int = 100,
    ) -> Quantity[u.erg / u.s]:
        if nu_array is None:  # if nu_array is not given, use the nu array in opacity
            nu_array = self.opacity.nu
        else:
            nu_array = nu_array.to(u.Hz, copy=False)
        
        return self.calc_L_nu(nu_array, r_sample_num) * nu_array
