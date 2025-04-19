from dataclasses import dataclass
from typing import Self
from warnings import warn

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity

from .utils import LogLogInterpolator

@dataclass(frozen=True, kw_only=True)
class SED:
    """SED: 表示光谱 (nu, L_nu) 的类。  
    输入: nu, L_nu，都必须带单位且具有兼容的量纲。nu 最好是递增的。  
    wavelength, nu_L_nu, L_lambda 等量在每次调用时计算。  
    
    from_file 方法用于从文件中读取。对文件格式有特定的要求，见该方法的 docstring。
    
    一个坑：虽然 frozen，但如果做类似 `sed.nu *= 2` 的操作，即便会报错，但也会改变原来的值。
    """
    nu: Quantity['frequency']
    L_nu: Quantity[u.erg/u.s/u.Hz]
    
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
    
    
    #TODO 把 bolometric correction 作为可选项，参数可以传入
    @classmethod
    def from_file(cls, filename: str) -> Self:
        """从文件中读取 SED 数据。  
        文件格式要求为：第一列为 wavelength (Angstrom)，第二列为 L_lambda (erg/s/Angstrom)。
        
        以前是 IncidentSED 类，现在整合为 SED 的类方法
        """
        # the incident SED data need to be in the format of wavelength (Angstrom) and L_lambda (erg/s/Angstrom)
        #* 目前假定文件中的数据是有序且无重复的
        data = np.loadtxt(filename, comments='#', delimiter=' ', usecols=(0, 1))
        #* 目前把 wavelength 和 L_lambda 都反转，从而使得最终的 nu 递增，而 wavelength 对应地递减
        wavelength = data[::-1, 0] * u.AA  # 假定文件中的 wavelength 单位是 Angstrom
        L_lambda = data[::-1, 1] * u.erg/u.s/u.AA
        
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

SED.__init__ = u.quantity_input(SED.__init__)  # 给 __init__ 方法添加单位检查。因为 SED 是 dataclass 不好直接装饰到 __init__ 上，所以在这里单独处理



def IncidentSED(filename: str):
    raise DeprecationWarning("IncidentSED is deprecated, please use SED.from_file instead.") 
        