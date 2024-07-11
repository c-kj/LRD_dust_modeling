import numpy as np
import astropy.units as u
import astropy.constants as const

from .utils import LogLogInterpolator

class IncidentSED:
    def __init__(self, filename: str):
        #* 目前假定文件中的数据是有序且无重复的
        data = np.loadtxt(filename, comments='#', delimiter=' ', usecols=(0, 1))
        #* 目前把 wavelength 和 L_lambda 都反转，从而使得最终的 nu 递增，而 wavelength 对应地递减
        self.wavelength = data[::-1, 0] * u.AA  # 假定文件中的 wavelength 单位是 Angstrom
        L_lambda = data[::-1, 1]
        
        self.nu = self.wavelength.to(u.Hz, u.spectral())  # 带单位 Hz
        L_nu = L_lambda * self.wavelength.cgs.value**2 / const.c.cgs.value  # 未归一化
        
        #TODO magic number 不太理解，有待修改
        L_bol = 2.4e46 # erg/s
        NormalizedFactorAt1450 = L_bol/4.4/2.0675e+15 # erg/s/Hz
        
        # 目前 L_nu 是不带单位的纯数值，单位应该是 erg/s/Hz
        self.L_nu = L_nu / LogLogInterpolator(self.wavelength, L_nu)(1450 * u.AA) * NormalizedFactorAt1450