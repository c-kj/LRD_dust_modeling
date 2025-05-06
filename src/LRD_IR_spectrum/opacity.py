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
    
    Constructor:
    - `__init__`: 从数据创建 OpacityData 对象。详见其 docstring。
    - `from_file`: 从文件中读取数据，创建 OpacityData 对象。
    - `from_extinction_data`: 从 extinction data 创建 OpacityData 对象。
    - `from_extinction_model`: 从 dust_extinction 包所提供的 extinction model，创建 OpacityData 对象。
    
    API:
    - `nu, wavelength, energy` : 横轴，nu 数组的等价表示
    - `sigma_H_ext, sigma_H_abs, sigma_H_Prad` : extinction/absorption/radiation-pressure cross section per H
    - `interp_ext, interp_abs, interp_Prad` : interpolators for the opacity data
    
    
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
        """从数据创建 OpacityData 对象。  
        
        Init parameters (keyword-only)：
        - nu: 频率数组
        - sigma_H_ext, sigma_H_abs: 与 nu 对应的 extinction/absorption cross section per H
            - sigma_H_abs 如果为 None（默认），则与 sigma_H_ext 相同。
        - filename: Optional, 用于记录数据来源的文件名。
        """
        
        self.filename = filename
        
        # 按照频率升序排列数据
        indices = np.argsort(nu)
        self.nu = nu[indices]
        self.sigma_H_ext = sigma_H_ext[indices]
        self.sigma_H_abs = sigma_H_abs[indices] if sigma_H_abs is not None else self.sigma_H_ext  # 如果没有提供 absorption cross section，则默认与 extinction cross section 相同。
        
    def __eq__(self, other):
        try:
            return bool(np.all(self.nu == other.nu)
                and np.all(self.sigma_H_ext == other.sigma_H_ext)
                and np.all(self.sigma_H_abs == other.sigma_H_abs))
        except AttributeError:  # other 不具有要求的属性
            return False
        except ValueError:  # 长度不一致引起的
            return False
    
    
    @property
    def wavelength(self):
        return self.nu.to(u.um, equivalencies=u.spectral())
    
    @property
    def energy(self):
        return self.nu.to(u.Ry, equivalencies=u.spectral())
    
    
    @property
    def sigma_H_Prad(self) -> Quantity['area']:
        """Radiation pressure cross section per H (sigma_H_Prad)  
        
        sigma_H_Prad = sigma_H_abs + (1- <cos(theta)>) * sigma_H_sca  
        <cos(theta)> 的观测值（ISM）参见 Fig 21.4 in Draine 2011。对于长波（Rayleigh 散射）为 0，对于 optical 波段约为 0.5.
        """
        cos_theta_avg = 0.5
        sigma_H_sca = self.sigma_H_ext - self.sigma_H_abs  # 散射截面
        return self.sigma_H_abs + (1 - cos_theta_avg) * sigma_H_sca
    
    
    #* 注意 @cached_property 要保证 nu, sigma_H_ext, sigma_H_abs 是不可变的，因为缓存无法随之变化。
    @cached_property
    def interp_ext(self):
        return LogLogInterpolator(self.nu, self.sigma_H_ext)

    @cached_property
    def interp_abs(self):
        return LogLogInterpolator(self.nu, self.sigma_H_abs)
    
    @cached_property
    def interp_Prad(self):
        return LogLogInterpolator(self.nu, self.sigma_H_Prad)
    
    @property
    def sigma_H_ext_V(self) -> Quantity['area']:
        """extinction cross section per H (sigma_H_ext) in V band"""
        nu_V: Quantity['frequency'] = (5470 * u.AA).to(u.Hz, equivalencies=u.spectral())
        return self.interp_ext(nu_V)
    
    @classmethod
    def from_file(cls, filename: str | Path, factor: float = 1.0, ignore_scatter: bool = False) -> Self:
        """从 CLOUDY output file 中读取数据，从而创建 OpacityData 对象。
        文件格式要求为：第一列为 nu (Rydberg)，第二列为 sigma_H_ext (cm^2)，第三列为 sigma_H_abs (cm^2)。
        """
        data = np.loadtxt(filename, comments='#', delimiter='\t', usecols=(0, 1, 2))  # read from file
        data = np.unique(data, axis=0)  # remove duplicate values, and sort by the first column
        
        energy = data[:, 0] * u.Ry  # in [Rydberg]
        nu = energy.to(u.Hz, equivalencies=u.spectral())
        sigma_H_ext = factor * data[:, 1] * u.cm**2
        
        if ignore_scatter:
            sigma_H_abs = None  # 向 __init__ 传入 None 表示取 absorption 与 extinction 截面相同
        else:
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

        return cls(nu=nu, sigma_H_ext=sigma_H_ext)  # 由于 extinction data 只有 extinction 截面，所以只能认为 absorption == extinction，即忽略 scattering 截面。
        
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