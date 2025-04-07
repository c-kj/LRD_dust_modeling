import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity

from .utils import LogLogInterpolator

#TODO 目前是一个简陋的实现，可以考虑改为 dataclass，并作为 IncidentSED 的基类？
class SED:
    @u.quantity_input
    def __init__(self, nu: Quantity['frequency'], L_nu: Quantity[u.erg/u.s/u.Hz]):
        self.nu = nu
        self.L_nu = L_nu

#TODO 统一使用带单位的量，不要中途使用不带单位的数字
#TODO 可以考虑作为 SED 的子类 or 作为一个「另种构造方法」？
#TODO 把 bolometric correction 作为可选项，参数可以传入
class IncidentSED:
    def __init__(self, filename: str):
        # the incident SED data need to be in the format of wavelength (Angstrom) and L_lambda (erg/s/Angstrom)
        #* 目前假定文件中的数据是有序且无重复的
        data = np.loadtxt(filename, comments='#', delimiter=' ', usecols=(0, 1))
        #* 目前把 wavelength 和 L_lambda 都反转，从而使得最终的 nu 递增，而 wavelength 对应地递减
        self.wavelength = data[::-1, 0] * u.AA  # 假定文件中的 wavelength 单位是 Angstrom
        L_lambda = data[::-1, 1] * u.erg/u.s/u.AA
        
        self.nu: Quantity['frequency'] = self.wavelength.to(u.Hz, u.spectral())  # 带单位 Hz
        L_nu = L_lambda * self.wavelength**2 / const.c

        # 进行归一化，使得 bolometric luminosity 为 L_bol
        # 2.0675e+15 是 1450 Angstrom 对应的频率
        L_bol = 1e46 * u.erg/u.s     # erg/s
        f_bol_UV = 4.4   #  从 bolometric 到 1450 Ang 处的 L_nu 的 correction 系数。 bolometric correction factor of f_{bol,UV} = 4.4 (Richards et al. 2006) 
        nu_1450AA =  (1450 * u.AA).to(u.Hz, u.spectral())  # 1450 Angstrom 对应的频率
        NormalizedFactorAt1450 = L_bol / f_bol_UV / nu_1450AA
        
        self.L_nu: Quantity[u.erg/u.s/u.Hz] = L_nu / LogLogInterpolator(self.wavelength, L_nu)(1450 * u.AA) * NormalizedFactorAt1450  # 归一化后，L_nu 在 1450 Angstrom 处的值为 NormalizedFactorAt1450