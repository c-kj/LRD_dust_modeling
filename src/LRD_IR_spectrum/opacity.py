from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from dust_extinction.baseclasses import BaseExtModel

from .utils import LogLogInterpolator


class OpacityData:
    """Wrapper for opacity data.
    
    #! 注意：此类的对象不应在创建后修改其数据，否则缓存的 interpolator 将与更新后的数据不一致。
    
    Init parameters (keyword-only)：
    - nu: 频率数组
    - sigma_H_ext, sigma_H_abs: 与 nu 对应的 extinction & absorption cross section
    - filename: Optional, 用于记录数据来源的文件名。
    
    Alternative constructor:
    - `from_file`: 从文件中读取数据，创建 OpacityData 对象。
    - `from_extinction_data`: 从 extinction data 创建 OpacityData 对象。
    - `from_extinction_model`: 从 dust_extinction 包所提供的 extinction model，创建 OpacityData 对象。
    
    Other API:
    - `energy, wavelength` : nu 数组的等价表示
    - `nu_cgs` : nu in Hz, for convenience
    - `interp_ext, interp_abs` : interpolators for the opacity data
    
    
    Usage:
    ```python
    Orion_opacity = OpacityData.from_file('data/opacity_law/Orion.opc')
    opacity = OpacityData.from_extinction_model(ext_model, sigma_H_V=...)
    ```    
    """
    
    def __init__(self, *, 
                 nu: Quantity['frequency'], 
                 sigma_H_ext: Quantity['area'], 
                 sigma_H_abs: Quantity['area'] | None = None, 
                 filename: str | None = None,):
        self.filename = filename
        
        # 按照频率升序排列数据
        indices = np.argsort(nu)
        self.nu = nu[indices]
        self.sigma_H_ext = sigma_H_ext[indices]
        self.sigma_H_abs = sigma_H_abs[indices] if sigma_H_abs is not None else self.sigma_H_ext  # 如果没有提供 absorption cross section，则默认与 extinction cross section 相同。
        
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
    
    @property
    def sigma_H_ext_V(self) -> Quantity['area']:
        """V band extinction cross section per H (sigma_H_ext)"""
        nu_V: Quantity['frequency'] = (5470 * u.AA).to(u.Hz, equivalencies=u.spectral())
        return self.interp_ext(nu_V)
    
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
    
    @classmethod
    @u.quantity_input(equivalencies=u.spectral())
    def from_extinction_data(cls, *,
                             x: Quantity['wavenumber'], 
                             A_rel: Quantity[''], 
                             sigma_H_V: Quantity['area']   #TODO 这里应该选什么值？
                             ) -> Self:
        """从 extinction data 创建 OpacityData 对象。  
        x 为频率 or 波长 or 波数等，在 spectral() 等效下转换。  
        A_rel 为与 x 对应的 A_lambda / A_V，即以 V band 归一化的消光值。  
        sigma_H_V 为 V band (5470 AA) 处的 extinction 截面 sigma per H。需要据此来计算其他波长上的 A_lambda。
        """
        nu = x.to(u.Hz, equivalencies=u.spectral())
        sigma_H_ext = A_rel * sigma_H_V

        return cls(nu=nu, sigma_H_ext=sigma_H_ext)
        
    @classmethod
    @u.quantity_input(equivalencies=u.spectral())
    def from_extinction_model(cls, 
                              model: BaseExtModel, 
                              *, 
                              sigma_H_V: Quantity['area'], 
                              x: Quantity['wavenumber'] | None = None, 
                              x_sample_num: int = 1000
                              ) -> Self:
        """从 dust_extinction 包所提供的 extinction model，创建 OpacityData 对象。
        参见 from_extinction_data 方法的 docstring。
        除非人为指定 x，否则 x 在 extinction model 的 x_range 内以 log scale 等距采样。
        """
        if x is None:
            x_range: tuple[float, float] = model.x_range
            x = np.geomspace(*x_range, num=x_sample_num) * (1/u.micron)
        return cls.from_extinction_data(x=x, A_rel=model(x), sigma_H_V=sigma_H_V)

OpacityData.__init__ = u.quantity_input(OpacityData.__init__)  # 给 __init__ 方法添加单位检查。因为 OpacityData 是 dataclass 不好直接装饰到 __init__ 上，所以在这里单独处理