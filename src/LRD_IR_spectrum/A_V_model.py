from functools import partial
from typing import override
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
    
    @u.quantity_input
    def __init__(self,
        n_0: Quantity['number density'],
        gamma: float,
        *,  # 以下参数必须用关键字指定
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
        #! 这里用 SED 的 nu 可能太粗糙了！也许插值更好
        # 红外 SED 数据点比较稀疏，不论是用 trapz_log 还是如何插值采样，都会有较大不确定性。但因为红外部分消光很弱，所以这部分的差距其实不显著。
        nu_array = self.incident_SED.nu
        return trapz_log(self.incident_SED.L_nu - self.observed_SED.L_nu, nu_array)