from functools import partial
from typing import override

import numpy as np
import astropy.units as u
from astropy.units import Quantity, Magnitude

from .opacity import OpacityData
from .incident_SED import SED
from .OrionLRD import OrionLRDModel
from .utils import quantity_to_latex

MagnitudeType = Quantity[u.mag] | Quantity['']  # 可以是纯数字、xxx * u.mag、xxx * u.mag()


def get_value_in_mag(A: MagnitudeType) -> float:
    """把 A 转换为以 magnitude 为单位的值。如果本身就是纯数，则直接返回。
    
    兼容 float, int, u.Magnitude, u.mag, u.dex 等类型。  
    不支持 1 * u.one 这种单位。
    """
    if isinstance(A, Quantity):
        A_mag = A.to_value(u.mag)  # downcast 到 u.mag 为单位的值上。这样 u.Magnitude, u.mag(), u.mag, u.dex 等都可以兼容
    elif isinstance(A, float | int):
        A_mag = A  # 如果是数字直接返回
    else:
        raise TypeError(f"A 的类型不支持：{type(A)}。A 应当是 float, int, u.Magnitude, u.mag, u.dex 等")
    
    return A_mag

def to_magnitude(A: MagnitudeType) -> Magnitude:
    return u.Magnitude(get_value_in_mag(A))  # 转换为 u.Magnitude 类型的值


def tau_from_A(A: MagnitudeType) -> float:
    """把消光 A 转换为光深 tau。  
    
    依据：exp(-tau) == 10**(-0.4 * A/mag) == L_obs / L_intrinsic
    可推得 A/mag = 1.086 * tau， 这个 1.086 实际上是 1/(0.4 log(10))
    """
    A_mag = get_value_in_mag(A)  # 转换为以 magnitude 为单位的值
    
    return A_mag * 0.4*np.log(10) 


def N_H_from_A_V(A_V: MagnitudeType, opacity: OpacityData) -> Quantity[u.cm**-2]:
    """Convert A_V to N_H
    
    :param A_V: extinction in magnitudes
    :param opacity: OpacityData object for interpolation
    :return: hydrogen column density in cm^-2
    
    Examples
    --------
    >>> N_H_from_A_V(3, Orion_opacity)  
    <Quantity 7.42834955e+22 1 / cm2>
    """
    nu_V: Quantity['frequency'] = (5470 * u.AA).to(u.Hz, equivalencies=u.spectral())  # frequency of V band
    sigma_H_ext_V = opacity.interp_ext(nu_V)  # sigma_H_ext (extinction cross-section per H) in V band
    tau_V = tau_from_A(A_V)  # optical depth in V band
    return tau_V / sigma_H_ext_V

#TODO 改造为一个类，继承 OrionLRDModel

class A_V_Model(OrionLRDModel):
    
    @u.quantity_input
    def __init__(self,
        n_0: Quantity['number density'],
        gamma: float,
        *,  # 以下参数必须用关键字指定
        T_sub: Quantity['temperature'],
        A_V: MagnitudeType,
        opacity: OpacityData,
        observed_SED: SED,  #* 这里必须输入的是 rest frame 的 nu 和 L_nu！ #TODO 强调这里必须是 rest frame
        tau_ph = 1, # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
        ):
        
        A_V = to_magnitude(A_V)  # 把 A_V 统一为 Magnitude 表示
        
        # 计算各种参数
        NH_target = N_H_from_A_V(A_V, opacity=opacity)
        
        A_lambda: Magnitude = A_V * (opacity.interp_ext(observed_SED.nu) / opacity.sigma_H_ext_V)  # A_lambda 是对应于 observed_SED.nu 的各个波长的消光值数组。# 可以考虑改名为 A_nu
        de_reddened_L_nu = observed_SED.L_nu / A_lambda.physical  # .physical 把 A_lambda 转化为消光的比率

        de_reddened_SED = SED(nu=observed_SED.nu, L_nu=de_reddened_L_nu)
        
        # 记录参数
        self.A_V = A_V
        self.observed_SED = observed_SED
        
        # 调用父类的 __init__ 
        super().__init__(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, incident_SED=de_reddened_SED, tau_ph=tau_ph)
        
    @override
    def _repr_latex_(self):
        fmt = partial(quantity_to_latex, p=4)
        return rf"""{self.__class__.__name__}( $n_0=$ {fmt(self.n_0)}, $\gamma={self.gamma}$, 
        $A_V=$ {fmt(self.A_V.to(u.mag))}, $N_{{\rm H}}=$ {fmt(self.NH_target)}, $\tau_{{\rm ph}}={self.tau_ph}$, 
        $r_{{\rm in}}=$ {fmt(self.r_in)}, $r_{{\rm ph}}=$ {fmt(self.r_ph)}, $r_{{\rm out}}=$ {fmt(self.r_out)}, 
        $T_{{\rm sub}}=$ {fmt(self.T_sub)}, $T_{{\rm out}}=$ {fmt(self.T_out)} )
        """