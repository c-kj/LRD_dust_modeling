from functools import partial
from typing import override, Callable, Protocol
from numbers import Real

import numpy as np
import astropy.units as u
from astropy.units import Quantity, Magnitude

from .opacity import OpacityData
from .incident_SED import SED
from .OrionLRD import OrionLRDModel
from .utils import quantity_to_latex, trapz_log

MagnitudeLike = Quantity[u.mag] | Quantity['']  # 可以是纯数字 (Real，包括 int|float) 或单位与 magnitude 兼容的 Quantity (xxx * u.mag、xxx * u.mag()、xxx * u.dex 等)。单个数字或 np.ndarray 均可。


def get_value_in_mag(A: MagnitudeLike) -> Real | np.ndarray:
    """把 A 转换为以 magnitude 为单位的值。如果本身就是纯数，则直接返回。  
    A 也可以是数组。
    
    兼容 float, int, u.Magnitude, u.mag, u.dex 等类型。  
    不支持 1 * u.one 这种单位。
    """
    if isinstance(A, Quantity):
        A_mag = A.to_value(u.mag)  # downcast 到 u.mag 为单位的值上。这样 u.Magnitude, u.mag(), u.mag, u.dex 等都可以兼容
    elif isinstance(A, Real | np.ndarray):  # 纯数字或 numpy array
        A_mag = A  # 直接返回
    else:
        raise TypeError(f"A 的类型不支持：{type(A)}。A 应当是 MagnitudeLike 类型。")
    
    return A_mag

def to_magnitude(A: MagnitudeLike) -> Magnitude:
    """把 MagnitudeLike 类型统一转换为 Magnitude 类型的量。"""
    return Magnitude(get_value_in_mag(A))


def tau_from_A(A: MagnitudeLike) -> float:
    """把消光 A 转换为光深 tau。  
    
    依据：exp(-tau) == 10**(-0.4 * A/mag) == L_obs / L_intrinsic
    可推得 A/mag = 1.086 * tau， 这个 1.086 实际上是 1/(0.4 log(10))
    """
    A_mag = get_value_in_mag(A)  # 转换为以 magnitude 为单位的值
    
    return A_mag * 0.4*np.log(10) 

def A_from_tau(tau: float) -> Magnitude:
    A_float = float(tau / (0.4*np.log(10)))  # 必须要转换为 float，否则 Magnitude 对于无量纲量给出的不是直接的数值，而是当做量纲为空的 Quantity 处理
    return Magnitude(A_float)  # 返回以 magnitude 为单位的值


def N_H_from_A_V(A_V: MagnitudeLike, opacity: OpacityData) -> Quantity[u.cm**-2]:
    """Convert A_V to N_H
    
    :param A_V: extinction in magnitudes
    :param opacity: OpacityData object for interpolation
    :return: hydrogen column density in cm^-2
    
    Examples
    --------
    >>> N_H_from_A_V(3, Orion_opacity)  
    <Quantity 7.42834955e+22 1 / cm2>
    """
    tau_V = tau_from_A(A_V)  # optical depth in V band
    return tau_V / opacity.sigma_H_ext_V


def de_redden_SED(*, observed_SED: SED, A_V: MagnitudeLike, opacity: OpacityData) -> SED:
    """对于观测到的 observed_SED，根据 A_V 和指定的 opacity 计算 de-reddened SED"""
    A_V = to_magnitude(A_V)  # 把 A_V 统一为 Magnitude 表示
    A_lambda: Magnitude = A_V * (opacity.interp_ext(observed_SED.nu) / opacity.sigma_H_ext_V)  # A_lambda 是对应于 observed_SED.nu 的各个波长的消光值数组。# 可以考虑改名为 A_nu
    de_reddened_L_nu = observed_SED.L_nu / A_lambda.physical  # .physical 把 A_lambda 转化为消光的比率

    return SED(nu=observed_SED.nu, L_nu=de_reddened_L_nu)



class A_V_Model(OrionLRDModel):
    """根据指定的 A_V，从 observed_SED 还原出 incident_SED，并计算相应的 NH_target 的模型。  
    用于从观测数据限制 A_V 的上限。  
    
    observed_SED 实际上是用于 de-redden 的。不应当包含 non-detection 的数据。
    """
    
    @u.quantity_input
    def __init__(self,
        *,  # 以下参数必须用关键字指定
        n_0: Quantity['number density'],
        gamma: float,
        T_sub: Quantity['temperature'],
        A_V: MagnitudeLike,
        opacity: OpacityData,
        observed_SED: SED,  #* 这里必须输入的是 rest frame 的 nu 和 L_nu！ #TODO 强调这里必须是 rest frame
        tau_ph = 1, # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
        config: dict = {},
        ):
        
        A_V = to_magnitude(A_V)  # 把 A_V 统一为 Magnitude 表示
        
        # 记录参数
        self.A_V = A_V
        self.observed_SED = observed_SED
        
        # 计算需要传递给父类的参数
        NH_target = N_H_from_A_V(A_V, opacity=opacity)
        de_reddened_SED = de_redden_SED(observed_SED=observed_SED, A_V=A_V, opacity=opacity)  # 计算 de-reddened SED
        
        # 调用父类的 __init__ 
        super().__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, incident_SED=de_reddened_SED, tau_ph=tau_ph, config=config)
        
    @override
    def _repr_latex_(self):
        fmt = partial(quantity_to_latex, p=4)
        return rf"""{self.__class__.__name__}( $n_0=$ {fmt(self.n_0)}, $\gamma={self.gamma}$, 
        $A_V=$ {fmt(self.A_V.to(u.mag))}, $N_{{\rm H}}=$ {fmt(self.NH_target)}, $\tau_{{\rm ph}}={self.tau_ph}$, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm ph}}=$ {fmt(self.r_ph)}, $r_{{\rm out}}=$ {fmt(self.r_out)}, 
        $T_{{\rm sub}}=$ {fmt(self.T_sub)}, $T_{{\rm out}}=$ {fmt(self.T_out)} )
        """
        
    # 用于计算吸收总功率 L 的方法
    @u.quantity_input
    def calc_L_from_extinction(self) -> Quantity[u.erg / u.s]:
        """根据消光前后的 de-reddened SED vs observed SED 计算吸收总功率"""
        # 红外 SED 数据点比较稀疏，不论是用 trapz_log 还是如何插值采样，都会有较大不确定性。但因为红外部分消光很弱，所以这部分的差距其实不显著。
        nu_array = self.incident_SED.nu
        return trapz_log(self.incident_SED.L_nu - self.observed_SED.L_nu, nu_array)
    
            
    # @u.quantity_input(equivalencies=u.spectral())   # 兼容各种可以在 u.spectral() 等效下转换为频率的物理量
    def calc_nu_L_nu_total(
        self,
        nu_array: Quantity['frequency'] | None = None,
        r_sample_num: int | None = None,
    ) -> Quantity[u.erg / u.s]:
        """计算 observed SED + re-emission SED 的总功率谱。  
        但目前并不投入使用（被注释掉了），因为论文 Fig.2 中目前不画 observed SED + re-emission SED。
        """
        if nu_array is None:  # if nu_array is not given, use the nu array in opacity
            nu_array = self.opacity.nu
        else:
            nu_array = nu_array.to(u.Hz, copy=False)
            
        observed_SED = self.observed_SED
        observed_nu_L_nu = observed_SED.interp_nu_L_nu_zeropad(nu_array)  # 需要在插值范围外补 0

        return self.calc_nu_L_nu(nu_array=nu_array, r_sample_num=r_sample_num) + observed_nu_L_nu
    
    
    # M_dust 的限制
    def A_V_out_from_M_gas(self, M_gas: Quantity[u.Msun]) -> Magnitude:
        """对于输入的 M_gas 计算对应的 r_out 处的 A_V，即 A_V_out。  
        可以用于根据 M_dust_max 限制 A_V_max。
        """
        N_H_out = self.NH_out_from_M_gas(M_gas)
        tau_V_out = N_H_out * self.opacity.sigma_H_ext_V
        return A_from_tau(tau_V_out)

    @property
    def A_V_max_from_NH_max(self):
        N_H_max = self.NH_max  # 有可能是 np.inf * u.cm**-2，但同样能正确计算
        tau_V_max = N_H_max * self.opacity.sigma_H_ext_V
        return A_from_tau(tau_V_max)

# A_V_Model 的 factory 的类型定义
class Partial_A_V_Model(Protocol):
    """一个 Protocol，表示一个类似 partial 对象的东西，接收 A_V 以及可选的其他 kwargs，返回一个 A_V_Model 对象。
    """
    def __call__(self, A_V: MagnitudeLike, /, **kwargs) -> A_V_Model:
        ...
        
A_V_ModelFactory = Callable[[MagnitudeLike], A_V_Model]  # 一个类型，只接收 A_V 一个参数，返回一个 A_V_Model 对象



# ----------------------------------- 带有缝隙的 model ----------------------------------- #

class A_V_Model_with_Gap(A_V_Model):
    """在 A_V_Model 的基础上，增加了缝隙 (gap) 的功能。  
    主要是为了模拟 Sakiko 的 ID830 源。
    
    两个成分：视线方向的 A_V_gap，和非视线方向的 A_V_torus。实际上这俩名字不太好，不该叫 gap，就应该叫 line-of-sight 之类的。
    计算 incident_SED 时，只用视线方向的 A_V；计算 re-emission 时，用视线方向和非视线方向的 A_V 的加权和（分别乘以各自的 covering factor）
    #? 这俩成分应该有共同的 gamma, n_0 吗？
    #* 应该分成俩 model 来计算，但目前临时，只考虑 gap 的情形，也就是 line-of-sight 组分的 A_V 很小而且 covering factor 也很小。
    #* 这样，re-emission 中只考虑了非视线方向的贡献（f_cover ~ 1）
    """
    

    @u.quantity_input
    def __init__(self,
        *,  # 以下参数必须用关键字指定
        n_0: Quantity['number density'],
        gamma: float,
        T_sub: Quantity['temperature'],
        A_V_torus: MagnitudeLike,
        A_V_gap: MagnitudeLike,
        opacity: OpacityData,
        observed_SED: SED,  #* 这里必须输入的是 rest frame 的 nu 和 L_nu！ #TODO 强调这里必须是 rest frame
        tau_ph = 1, # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
        config: dict = {},
        ):
        
        A_V_torus = to_magnitude(A_V_torus)  # 把 A_V 统一为 Magnitude 表示
        A_V_gap = to_magnitude(A_V_gap)  # 把 A_V 统一为 Magnitude 表示
        
        # 记录参数
        self.A_V_torus = A_V_torus
        self.A_V_gap = A_V_gap
        self.observed_SED = observed_SED
        
        # 计算需要传递给父类的参数
        NH_target = N_H_from_A_V(A_V_torus, opacity=opacity)
        de_reddened_SED = de_redden_SED(observed_SED=observed_SED, A_V=A_V_gap, opacity=opacity)  # 计算 de-reddened SED
        
        # 调用父类的 __init__ 
        super(A_V_Model, self).__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, incident_SED=de_reddened_SED, tau_ph=tau_ph, config=config)

        #TEMP 目前没有实现 f_cover
        
    @override
    def _repr_latex_(self):
        fmt = partial(quantity_to_latex, p=4)
        return rf"""{self.__class__.__name__}( $n_0=$ {fmt(self.n_0)}, $\gamma={self.gamma}$, 
        $A_{{V, torus}}=$ {fmt(self.A_V_torus.to(u.mag))}, $A_{{V, gap}}=$ {fmt(self.A_V_gap.to(u.mag))}, $N_{{\rm H}}=$ {fmt(self.NH_target)}, $\tau_{{\rm ph}}={self.tau_ph}$, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm ph}}=$ {fmt(self.r_ph)}, $r_{{\rm out}}=$ {fmt(self.r_out)}, 
        $T_{{\rm sub}}=$ {fmt(self.T_sub)}, $T_{{\rm out}}=$ {fmt(self.T_out)} )
        """