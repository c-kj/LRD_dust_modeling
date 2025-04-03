import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity

from .utils import LogLogInterpolator

#TODO 目前是一个简陋的实现，可以考虑改为 dataclass，并作为 IncidentSED 的基类？
class SED:
    def __init__(self, nu: Quantity, L_nu: Quantity):
        self.nu = nu
        self.L_nu = L_nu

#TODO 统一使用带单位的量，不要中途使用不带单位的数字
class IncidentSED:
    def __init__(self, filename: str):
        # the incident SED data need to be in the format of wavelength (Angstrom) and L_lambda (erg/s/Angstrom)
        #* 目前假定文件中的数据是有序且无重复的
        data = np.loadtxt(filename, comments='#', delimiter=' ', usecols=(0, 1))
        #* 目前把 wavelength 和 L_lambda 都反转，从而使得最终的 nu 递增，而 wavelength 对应地递减
        self.wavelength = data[::-1, 0] * u.AA  # 假定文件中的 wavelength 单位是 Angstrom
        L_lambda = data[::-1, 1] * u.erg/u.s/u.AA
        
        self.nu = self.wavelength.to(u.Hz, u.spectral())  # 带单位 Hz
        L_nu = L_lambda * self.wavelength**2 / const.c
        L_nu = L_nu.to(u.erg/u.s/u.Hz).value

        # 进行归一化，使得 bolometric luminosity 为 L_bol
        # 2.0675e+15 是 1450 Angstrom 对应的频率
        L_bol = 1e46     # erg/s
        f_bol_UV = 4.4   #  从 bolometric 到 1450 Ang 处的 L_nu 的 correction 系数。 bolometric correction factor of f_{bol,UV} = 4.4 (Richards et al. 2006) 
        NormalizedFactorAt1450 = L_bol/f_bol_UV/2.0675e+15 # erg/s/Hz
        
        self.L_nu = L_nu / LogLogInterpolator(self.wavelength, L_nu)(1450 * u.AA) * NormalizedFactorAt1450  # 归一化后，L_nu 在 1450 Angstrom 处的值为 NormalizedFactorAt1450
        # 目前 L_nu 是不带单位的纯数值，单位应该是 erg/s/Hz
        
        #TEMP 把 L_nu 给带上单位
        self.L_nu = self.L_nu * u.erg/u.s/u.Hz