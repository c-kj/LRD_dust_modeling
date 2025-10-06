# 定义宇宙学参数，用于观测到的 f_nu 与 L_nu 之间的转换 （需要光度距离）

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.cosmology import Cosmology

def f_nu_from_L_nu_rest(*, L_nu_rest: Quantity[u.Lsun/u.Hz], cosmology: Cosmology, z: float, magnification: float = 1):
    D_L = cosmology.luminosity_distance(z)
    f_nu = L_nu_rest * (1 + z) / (4 * np.pi * D_L**2) * magnification
    return f_nu
    
def L_nu_rest_from_f_nu(*, f_nu: Quantity[u.uJy], cosmology: Cosmology, z: float, magnification: float = 1):
    D_L = cosmology.luminosity_distance(z)
    L_nu_rest = f_nu * (4 * np.pi * D_L**2) / (1 + z) / magnification
    return L_nu_rest