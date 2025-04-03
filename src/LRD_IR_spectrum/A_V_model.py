import astropy.units as u
from astropy.units import Quantity


from .opacity import OpacityData
from .incident_SED import SED
from .OrionLRD import OrionLRDModel


#FUTURE 加上更详细的 type hint，并用 @u.quantity_input 检查。考虑把 A_V 改为 magnitude 单位
def N_H_from_A_V(A_V: float, opacity: OpacityData) -> Quantity:
    """Convert A_V to N_H
    
    :param A_V: extinction in magnitudes
    :param opacity: OpacityData object for interpolation
    :return: hydrogen column density in cm^-2
    
    Examples
    --------
    >>> N_H_from_A_V(3, Orion_opacity)  
    <Quantity 7.42834955e+22 1 / cm2>
    """
    nu_V: Quantity = (5470 * u.AA).to(u.Hz, equivalencies=u.spectral())  # frequency of V band
    sigma_H_ext_V = opacity.interp_ext(nu_V)  # sigma_H_ext (extinction cross-section per H) in V band
    return A_V / (1.086 * sigma_H_ext_V)

#TODO 改造为一个类，继承 OrionLRDModel
def A_V_model(
    n_0: float,
    gamma: float,
    *,  # 以下参数必须用关键字指定
    T_sub: float,
    A_V: float,
    opacity: OpacityData,
    observed_SED: SED,  #* 这里必须输入的是 rest frame 的 nu 和 L_nu！ #TODO 强调这里必须是 rest frame
    tau_ph = 1, # feedback 将内区的 dust 吹到某个 r_ph 位置堆积。这是对应的光深。
    ) -> OrionLRDModel:
    
    NH_target = N_H_from_A_V(A_V, opacity=opacity)
    
    with u.set_enabled_equivalencies(u.spectral()):
        sigma_H_V = opacity.interp_ext(5470 * u.AA)
    
    A_lambda = A_V * opacity.interp_ext(observed_SED.nu) / sigma_H_V 
    de_reddened_L_nu = observed_SED.L_nu * (10**(0.4 * A_lambda))
    
    de_reddened_SED = SED(nu=observed_SED.nu, 
                          L_nu=de_reddened_L_nu)
    
    model = OrionLRDModel(n_0=n_0, gamma=gamma, T_sub=T_sub, NH_target=NH_target.cgs.value, opacity=opacity, incident_SED=de_reddened_SED, tau_ph=tau_ph)
    return model