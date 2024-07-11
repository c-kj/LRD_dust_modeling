import numpy as np
import astropy.units as u
from scipy import interpolate, integrate


try:
    import astropy.units as u
    INSTALLED_astropy = True
except ImportError:
    INSTALLED_astropy = False
try:
    import unyt
    INSTALLED_unyt = True
except ImportError:
    INSTALLED_unyt = False


class ScaledInterpolator:
    @staticmethod
    def _identity(x):
        return x
    
    @staticmethod
    def get_value_unit(x):
        if INSTALLED_astropy and isinstance(x, u.Quantity):  # if astropy is installed, then enable the Quantity support
            return x.value, x.unit
        elif INSTALLED_unyt and isinstance(x, unyt.unyt_array):  # if unyt is installed, then enable the unyt_array support
            return x.value, x.units
        return x, 1
    
    def __init__(self, x, y, *, scale_x=(_identity, _identity), scale_y=(_identity, _identity), **kwargs):
        self.x = x
        self.y = y
        self.f_x, self.inv_f_x = scale_x
        self.f_y, self.inv_f_y = scale_y
        
        x_value, self.x_unit = self.get_value_unit(x)
        y_value, self.y_unit = self.get_value_unit(y)
        
        #* 对于 kwargs 中的 fill_value，需要单独处理。如果指定为具体的值，要用 f_y 转换。
        if 'fill_value' in kwargs:
            fill_value = kwargs['fill_value']
            if fill_value == 'extrapolate':   # 如果是 'extrapolate'，则不做处理；
                pass
            elif isinstance(fill_value, tuple) and len(fill_value) == 2:   # 如果是一个 tuple，则对 tuple 中的每个元素做转换，仍然给出一个 tuple
                fill_value = (self.f_y(fill_value[0]), self.f_y(fill_value[1]))
            else:   # 否则对整个 fill_value 做转换。
                fill_value = self.f_y(fill_value)
            kwargs['fill_value'] = fill_value  # 用转换后的 fill_value 替换原来的 fill_value
        
        
        self._interp = interpolate.interp1d(self.f_x(x_value), self.f_y(y_value), **kwargs)
        
    def __call__(self, x):
        unit = self.get_value_unit(x)[1] 
        if self.x_unit != 1 and unit != 1:   # if input x has a unit
            x = x.to_value(self.x_unit)   # convert to the same internal unit. Note the .to_value() method is the same in astropy and unyt
        elif self.x_unit == 1 and unit == 1:
            pass
        else:  # if x is unitless, but the interpolator has a unit
            raise ValueError(f"The input x should have the same unit as {self.x_unit = } in the interpolator")
        
        return self.inv_f_y(self._interp(self.f_x(x))) * self.y_unit
    
    @property
    def inverse(self):
        """Return a new interpolator with the x and y swapped. The scale_x and scale_y are also swapped."""
        return __class__(self.y, self.x, scale_x=(self.f_y, self.inv_f_y), scale_y=(self.f_x, self.inv_f_x))


class LogLogInterpolator(ScaledInterpolator):
    """以 log-log 尺度插值。
    
    注意：np.log 和 np.exp 在浮点数运算时不是严格的反函数，会有很小的数值误差。
    """
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, scale_x=(np.log, np.exp), scale_y=(np.log, np.exp), **kwargs)
        
class LogLinearInterpolator(ScaledInterpolator):
    def __init__(self, x, y, **kwargs):
        _identity = self._identity
        super().__init__(x, y, scale_x=(np.log, np.exp), scale_y=(_identity, _identity), **kwargs)
        
        
        
# 辅助函数，用于在自变量 log 尺度下做积分

def trapz_log(y, x, *args, **kwargs):
    return integrate.trapezoid(y * x, np.log(x), *args, **kwargs)

def quad_vec_log(f, a, b, *args, **kwargs):
    def integrand(log_x):
        x = np.exp(log_x)
        return f(x) * x
    return integrate.quad_vec(integrand, np.log(a), np.log(b), *args, **kwargs)
        
        
        