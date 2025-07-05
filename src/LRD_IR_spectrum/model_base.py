from abc import ABC, abstractmethod
from functools import partial
import logging
from collections import ChainMap

import numpy as np
from scipy import integrate, special
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity

from .utils import Planck_B_nu, trapz_log, quad_mapping, trapz_mapping, quantity_to_latex
from .opacity import OpacityData

class R_out_Error(Exception):
    pass

class IPyChainMap(ChainMap):
    """A subclass of ChainMap that allows for IPython tab completion."""
    def _ipython_key_completions_(self):  # IPython 检查这个方法来支持在 tab 补全中显示各个 keys
        return list(self)  # 返回各个 key 组成的列表


#FUTURE 重新考虑 抽象基类 该怎么写，最好只提供接口。实现放到 mixin？
class LRD_IR_ModelBase(ABC):
    """各个 Model 的抽象基类。  
    
    抽象方法：
    - r_in 的计算（根据 T_sub）
    - T_dust_profile 的计算
    
    具体实现了：
    - dust 的分布
        - n_0 和 gamma 参数
        - r_out 的计算（根据 NH_target）
        - dust 密度的幂律分布、相应的 NH_profile 和其逆
    - 指定 opacity
    - dust re-emission 光谱的计算（根据 T_dust_profile, n_profile 和 opacity 积分）
    - 其他一些工具
        - _repr_latex_ 和 __format__
        - r_with_feedback
        - check_Luminosity
        - ...
    """
    
    # 类属性
    # 类的 config 会被所有实例共享。对其的修改会直接影响所有实例（包括已创建的），因为实例的 config 是 ChainMap。
    config = {
        'calc_L_nu.integrator': 'trapz_log',  # calc_L_nu 的积分方法。默认方法是 'trapz_log'，因为 trapz 要显著快于 quad，而 log scale 收敛性显著更好。
        'r_sample_num': 1000,  # 对 r 积分时，从 r_in 到 r_out 之间采样的点数。
        'r_sample_scale': 'log',  # get_r_array 采样的 scale。可选 'linear' 或 'log'。
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
        config: dict = {},  # 注意 config 的默认值是可变的，不要修改它！
    ):
        self.n_0 = n_0
        self.gamma = gamma
        self.T_sub = T_sub
        self.NH_target = NH_target
        self.opacity = opacity
        
        # 用 ChainMap 把传入的 config 和类属性 config 串联，这样实例对 config 的修改会保存在前面；对类属性 config 的修改也会实时更新到实例上（因为 ChainMap 是 View ）
        self.config = IPyChainMap(config, self.__class__.config)  
        
        # r_in 目前在初始化时计算。子类应当覆写 _calc_r_in 方法。
        self.r_in = self._calc_r_in()  #* 可以在运行时手动修改 r_in
    

    # 目前暂时废弃 __repr__，避免维护成本。反正 Jupyter notebook 里显示用的也是 latex 格式
    # def __repr__(self):
    #     return f"{self.__class__.__name__}(n_0={self.n_0}, gamma={self.gamma}, L_UV={self.L_UV}, T_sub={self.T_sub}, r_in={self.r_in})"

    def _repr_latex_(self):
        """在 Jupyter Notebook 的单元输出中渲染时所调用的方法"""
        fmt = partial(quantity_to_latex, p=4)
        
        # 其实基类中并没有 self.L_UV 属性。但目前几个具体子类要么接收 L_UV 参数，要么 override 了这个 repr，所以这样写没事。
        return rf"""{self.__class__.__name__}( $n_0=$ {fmt(self.n_0)}, $\gamma={self.gamma}$, 
        $L_{{\rm UV}}=$ {fmt(self.L_UV)}, $N_{{\rm H}}=$ {fmt(self.NH_target)}, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm out}}=$ {fmt(self.r_out)} , 
        $T_{{\rm sub}}=$ {fmt(self.T_sub)}, $T_{{\rm out}}=$ {fmt(self.T_out)} )
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
    def r_with_feedback(self, r: Quantity['length']) -> Quantity[u.pc]:
        """考虑 feedback 反馈效果时，从尘埃原位置 r 到新位置 r' 的映射。  
        对于无反馈的模型，r' = r。
        """
        return r

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
        # if NH_target is None:  # 以前似乎考虑 NH_target 可以传入 None 的情况，但现在要求必须传入值了。
        #     raise ValueError("NH_target is not specified yet!")
        r_out = self.NH_profile_inverse(NH_target)
        
        if np.isinf(r_out):        #! 注意，在极端参数下，r_out 可能超过浮点数上界，变为 inf。
            logging.warning("r_out is inf due to float overflow.")
            
        return r_out
    
    def get_r_array(self, r_sample_num: int | None = None, r_sample_scale: str | None = None) -> Quantity[u.pc]:
        """给出 r_in 和 r_out 之间采样的 r_array。  
        因为经常使用，所以抽出来作为一个方法，从而统一控制。  
        选项如果不指定，则 fallback 到 self.config 中的默认值。
        """
        if r_sample_num is None:  # 如果没有指定采样点数，则使用 config 中的默认值
            r_sample_num = self.config['r_sample_num']
        if r_sample_scale is None:  # 如果没有指定采样 scale，则使用 config 中的默认值
            r_sample_scale = self.config['r_sample_scale']
            
        if r_sample_scale == 'log':
            return np.geomspace(self.r_in, self.r_out, num=r_sample_num)
        elif r_sample_scale == 'linear':  # 实际上，linspace 采样是很不合适的，收敛性很差。
            return np.linspace(self.r_in, self.r_out, num=r_sample_num)
        else:
            raise ValueError(f"get_r_array: {r_sample_scale = } is not supported! ")


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

    @u.quantity_input(equivalencies=u.spectral())  # 兼容各种可以在 u.spectral() 等效下转换为频率的物理量
    def calc_L_nu(
        self,
        nu_array: Quantity['frequency'] | None = None,
        r_sample_num: int | None = None,
    ) -> Quantity[u.erg / u.s / u.Hz]:
        """calc L_nu at given frequency array, using the given opacity data

        Parameters
        ----------
        nu_array : Quantity['frequency'] | None, optional
            If not specified (default = None), will use opacity.nu.
            可以是任何可以在 u.spectral() 等效下转换为频率的数组，比如波长数组。
        r_sample_num : int | None, optional
            采样点数。默认值不指定时，取 self.config['r_sample_num']。
        
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
        def integrand(nu: Quantity['frequency'], r: Quantity['length']) -> Quantity[u.erg / u.s / u.Hz / u.cm**3]:
            # note: 这里的 self 实际上是从函数外部直接引用的，没有传入
            # 要把 4πr^2 中的 r 改为 dust 颗粒实际所在的位置，也即 r_with_feedback。而 T 和 n 的 profile 中所使用的仍是原先的 r。
            r_with_feedback = self.r_with_feedback(r)
            return Planck_B_nu(nu, self.T_dust_profile(r)) * self.n_profile(r) * 4 * np.pi * r_with_feedback**2  # 这里的 4pi 是 dV = 4 pi r^2 dr 的系数

        #* 主要是采样方式 (log / linear) 对积分的收敛性影响较大。积分是用 trapz 还是 trapz_log 影响较小。
        L_nu_integrator: str = self.config['calc_L_nu.integrator']
        if L_nu_integrator in {'quad', 'quad_log'}:  # 从函数计算积分（自适应）
            integrator = quad_mapping[L_nu_integrator]  # 挑选对应的积分器
            L_nu = integrator(lambda r: integrand(nu_array, r), self.r_in, self.r_out)[0]
        elif L_nu_integrator in {'trapz', 'trapz_log'}:  # 从采样点计算积分
            integrator = trapz_mapping[L_nu_integrator]  # 挑选对应的积分器
            r_array = self.get_r_array(r_sample_num=r_sample_num)  # 对 r 采样。r_sample_scale 需要从 config 中指定。
            integrand_array = integrand(nu_array[:, None], r_array)
            L_nu = integrator(integrand_array, r_array)
        else:
            raise ValueError(f"method {L_nu_integrator = } is invalid! ")

        L_nu *= 4 * np.pi * sigma_H   # 最后，乘上积分前面统一的 factor。sigma_H 这个关于 nu 的数组是提到积分外面来的。这里的 4pi 是出射立体角 Omega

        return L_nu
    
    @u.quantity_input(equivalencies=u.spectral())   # 兼容各种可以在 u.spectral() 等效下转换为频率的物理量
    def calc_nu_L_nu(
        self,
        nu_array: Quantity['frequency'] | None = None,
        r_sample_num: int | None = None,
    ) -> Quantity[u.erg / u.s]:
        """与 calc_L_nu 类似，但计算的是 nu * L_nu  
        参见 calc_L_nu 的文档
        """
        if nu_array is None:  # if nu_array is not given, use the nu array in opacity
            nu_array = self.opacity.nu
        else:
            nu_array = nu_array.to(u.Hz, copy=False)
        
        return self.calc_L_nu(nu_array, r_sample_num) * nu_array
    
    
    # 用于计算发射总功率 L 的方法
    @u.quantity_input
    def calc_L_from_L_nu(self, *, r_sample_num: int | None = None) -> Quantity[u.erg / u.s]:
        """从 L_nu 计算 dust 发射的总功率"""
        # 目前没加入可以指定的 nu_array，因为好像意义不大。
        return trapz_log(self.calc_L_nu(r_sample_num=r_sample_num), self.opacity.nu)
    
    def check_Luminosity(self):
        """调用模型上所有计算总光度 L 的方法，用于检查其一致性。  
        返回值为一个字典，key 为方法名，value 为对应的 L 值。
        """
        Luminosity_dict: dict[str, Quantity['power']] = {
            method_name : getattr(self, method_name)() 
            for method_name in self.__dir__() if method_name.startswith('calc_L_from')  # 通过方法名的开头来识别「计算总光度的方法」。这里不用 dir(self) 是因为它按字母序排列。
        }
        return Luminosity_dict