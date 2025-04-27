from dataclasses import dataclass
from pathlib import Path
from typing import Self
from warnings import warn

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity

from .utils import LogLogInterpolator

class SED:
    """SED: 表示光谱 (nu, L_nu) 的类。  
    输入: nu, L_nu，都必须带单位且具有兼容的量纲。
    L_nu 与 nu 逐点对应，会自动按 nu 的升序排序。
    wavelength, nu_L_nu, L_lambda 等量在每次调用时计算。  
    
    from_file 方法用于从文件中读取。对文件格式有特定的要求，见该方法的 docstring。
    """
    
    @u.quantity_input
    def __init__(self, *, 
                 nu: Quantity['frequency'], 
                 L_nu: Quantity[u.erg/u.s/u.Hz]):
        # 按照频率升序排列数据
        indices = np.argsort(nu)
        self.nu: Quantity = nu[indices]
        self.L_nu: Quantity = L_nu[indices]
    
    
    def __eq__(self, other):  # 不用 dataclass 定义的 __eq__，因为它不能处理 numpy 数组的相等
        if not isinstance(other, self.__class__):
            warn(f"比较的对象类型为 {type(other)}，与 {self.__class__} 不同，需要谨慎考虑比较是否合理")
        return np.all(self.nu == other.nu) and np.all(self.L_nu == other.L_nu)
    
    
    @property
    def wavelength(self) -> Quantity[u.AA] :
        return self.nu.to(u.AA, equivalencies=u.spectral())
    
    @property
    def nu_L_nu(self) -> Quantity[u.erg/u.s]:
        return self.nu * self.L_nu

    @property
    def L_lambda(self) -> Quantity[u.erg/u.s/u.AA]:
        return self.nu_L_nu / self.wavelength
    
    # L_nu 和其等价表示 (nu_L_nu 等) 的插值函数，以 nu 为自变量。
    # 目前不缓存它们，因为调用次数不多。这样可以避免更改时缓存不更新的潜在问题。
    @property
    def interp_L_nu(self):
        return LogLogInterpolator(self.nu, self.L_nu)
    
    @property
    def interp_nu_L_nu(self):
        return LogLogInterpolator(self.nu, self.nu_L_nu)
    
    @property
    def interp_L_lambda(self):
        return LogLogInterpolator(self.nu, self.L_lambda)
    
    
    #TODO 把 bolometric correction 作为可选项，参数可以传入
    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """从文件中读取 SED 数据。  
        文件格式要求为：第一列为 wavelength (Angstrom)，第二列为 L_lambda (erg/s/Angstrom)。
        
        以前是 IncidentSED 类，现在整合为 SED 的类方法
        """
        #* 目前假定文件中的数据是有序且无重复的
        data = np.loadtxt(filename, comments='#', delimiter=' ', usecols=(0, 1))
        # 这里不用在乎升降序，最后传到 __init__ 中时会自动排序
        wavelength = data[:, 0] * u.AA  # 假定文件中的 wavelength 单位是 Angstrom
        L_lambda = data[:, 1] * u.erg/u.s/u.AA
        
        nu: Quantity['frequency'] = wavelength.to(u.Hz, u.spectral())  # 带单位 Hz
        L_nu = L_lambda * wavelength**2 / const.c

        # 进行归一化，使得 bolometric luminosity 为 L_bol
        # 2.0675e+15 是 1450 Angstrom 对应的频率
        L_bol = 1e46 * u.erg/u.s     # erg/s
        f_bol_UV = 4.4   #  从 bolometric 到 1450 Ang 处的 L_nu 的 correction 系数。 bolometric correction factor of f_{bol,UV} = 4.4 (Richards et al. 2006) 
        nu_1450AA =  (1450 * u.AA).to(u.Hz, u.spectral())  # 1450 Angstrom 对应的频率
        NormalizedFactorAt1450 = L_bol / f_bol_UV / nu_1450AA
        
        L_nu: Quantity[u.erg/u.s/u.Hz] = L_nu / LogLogInterpolator(wavelength, L_nu)(1450 * u.AA) * NormalizedFactorAt1450  # 归一化后，L_nu 在 1450 Angstrom 处的值为 NormalizedFactorAt1450
        
        return cls(nu=nu, L_nu=L_nu)


def IncidentSED(filename: str):
    raise DeprecationWarning("IncidentSED is deprecated, please use SED.from_file instead.") 
        