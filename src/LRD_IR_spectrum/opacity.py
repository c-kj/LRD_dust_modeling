from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from .utils import LogLogInterpolator


@dataclass(frozen=True, kw_only=True)
class OpacityData:
    """Wrapper for opacity data.
    
    Init parameters (keyword-only)：
    - nu: 频率数组
    - sigma_H_ext, sigma_H_abs: 与 nu 对应的 extinction & absorption cross section
    - filename: Optional, 用于记录数据来源的文件名。
    
    Alternative constructor:
    - `from_file`: 从文件中读取数据，创建 OpacityData 对象。
    
    Other API:
    - `energy, wavelength` : nu 数组的等价表示
    - `nu_cgs` : nu in Hz, for convenience
    - `interp_ext, interp_abs` : interpolators for the opacity data
    
    
    Usage:
    ```python
    Orion_opacity = OpacityData.from_file('Orion_Tdust20_Sigma_23_Thickness_16_Hden_07.opc')
    ```    
    """
    
    filename: str | None = None  # optional。如果不是从文件中读取，则 filename 为 None
    nu: Quantity['frequency']
    sigma_H_ext: Quantity['area']
    sigma_H_abs: Quantity['area']
        
    def __eq__(self, other):
        return np.all(self.nu == other.nu) \
            and np.all(self.sigma_H_ext == other.sigma_H_ext) \
            and np.all(self.sigma_H_abs == other.sigma_H_abs)
    
    
    @property
    def wavelength(self):
        return self.nu.to(u.um, equivalencies=u.spectral())
    
    @property
    def energy(self):
        return self.nu.to(u.Ry, equivalencies=u.spectral())
    
    @property
    def nu_cgs(self):
        return self.nu.to_value(u.Hz)
    
    #* 注意 @cached_property 要保证 nu, sigma_H_ext, sigma_H_abs 是不可变的，因为缓存无法随之变化。
    @cached_property
    def interp_ext(self):
        return LogLogInterpolator(self.nu, self.sigma_H_ext)

    @cached_property
    def interp_abs(self):
        return LogLogInterpolator(self.nu, self.sigma_H_abs)
    
    
    @classmethod
    def from_file(cls, filename: str | Path, factor: float = 1.0) -> Self:
        """从 CLOUDY output file 中读取数据，从而创建 OpacityData 对象。
        文件格式要求为：第一列为 nu (Rydberg)，第二列为 sigma_H_ext (cm^2)，第三列为 sigma_H_abs (cm^2)。
        """
        data = np.loadtxt(filename, comments='#', delimiter='\t', usecols=(0, 1, 2))  # read from file
        data = np.unique(data, axis=0)  # remove duplicate values, and sort by the first column
        
        energy = data[:, 0] * u.Ry  # in [Rydberg]
        nu = energy.to(u.Hz, equivalencies=u.spectral())
        sigma_H_ext = factor * data[:, 1] * u.cm**2
        sigma_H_abs = factor * data[:, 2] * u.cm**2
        
        return cls(filename=filename, nu=nu, sigma_H_ext=sigma_H_ext, sigma_H_abs=sigma_H_abs)

OpacityData.__init__ = u.quantity_input(OpacityData.__init__)  # 给 __init__ 方法添加单位检查。因为 OpacityData 是 dataclass 不好直接装饰到 __init__ 上，所以在这里单独处理