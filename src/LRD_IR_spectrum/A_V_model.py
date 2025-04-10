import numpy as np
import astropy.units as u
from astropy.units import Quantity


from .opacity import OpacityData
from .incident_SED import SED
from .OrionLRD import OrionLRDModel


def tau_from_A(A: float) -> float:
    """把消光 A 转换为光深 tau。  
    A 是以 magnitude 为单位的数字，返回值 tau 是纯数字。
    
    依据：exp(-tau) == 10**(-0.4 * A)
    可推得 A = 1.086 * tau， 这个 1.086 实际上是 1/(0.4 log(10))
    """
    return A * 0.4*np.log(10) 

@u.quantity_input
def N_H_from_A_V(A_V: float, opacity: OpacityData) -> Quantity[u.cm**-2]:
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
@u.quantity_input
def A_V_model(
    n_0: Quantity['number density'],
    gamma: float,
    *,  # 以下参数必须用关键字指定
    T_sub: Quantity['temperature'],
    A_V: float,
    opacity: OpacityData,
    observed_SED: SED,  #* 这里必须输入的是 rest frame 的 nu 和 L_nu！ #TODO 强调这里必须是 rest frame
    tau_ph = 1, # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
    ) -> OrionLRDModel:
    
    NH_target = N_H_from_A_V(A_V, opacity=opacity)
    
    with u.set_enabled_equivalencies(u.spectral()):
        sigma_H_V = opacity.interp_ext(5470 * u.AA)
    
    # A_nu = opacity.interp_ext(observed_SED.nu) * NH_target * 1.086 #TODO 考虑用哪种写法来计算 A_nu
    A_lambda = A_V * opacity.interp_ext(observed_SED.nu) / sigma_H_V 
    de_reddened_L_nu = observed_SED.L_nu * (10**(0.4 * A_lambda))  #TODO 把这里改为用 u.Magnitude.physical 计算？
    
    de_reddened_SED = SED(nu=observed_SED.nu, 
                          L_nu=de_reddened_L_nu)
    
    model = OrionLRDModel(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target, opacity=opacity, incident_SED=de_reddened_SED, tau_ph=tau_ph)
    return model