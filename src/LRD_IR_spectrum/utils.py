from functools import wraps
from typing import Callable

import numpy as np
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


def get_value_unit(x, target_unit=None):
    """把一个量拆分成 value 和 unit。兼容 astropy、unyt 和 纯数字。
    
    target_unit 是一个单位，如果指定，则将 x 在该单位下拆分为 value 和 unit。
    """
    if INSTALLED_astropy and isinstance(x, u.Quantity):  # if astropy is installed, then enable the Quantity support
        if target_unit is not None:
            x <<= target_unit  # 原地转换到目标单位
        return x.value, x.unit
    
    elif INSTALLED_unyt and isinstance(x, unyt.unyt_array):  # if unyt is installed, then enable the unyt_array support
        if target_unit is not None:
            x.convert_to_units(target_unit)  # 原地转换到目标单位
        return x.value, x.units
    
    else:  # 纯数字。但目前没有处理 x 既不是 astropy 或 unyt 的量，也不是数字的情况。
        if target_unit is not None:
            raise ValueError("只有当 x 是 astropy 或 unyt 的量时，才能指定 target_unit。")
        return x, 1

def get_value(x):  # 目前为了简洁性，不接收 target_unit 参数
    """如果 x 是 astropy 或 unyt 的量，则返回其 value，否则直接返回 x"""
    return getattr(x, 'value', x)  # 利用了 astropy 和 unyt 的量都有 .value 属性，就不用分别处理了

class ScaledInterpolator:
    @staticmethod
    def _identity(x):
        return x
    
    def __init__(self, x, y, *, scale_x=(_identity, _identity), scale_y=(_identity, _identity), **kwargs):
        self.x = x
        self.y = y
        self.f_x, self.inv_f_x = scale_x
        self.f_y, self.inv_f_y = scale_y
        self.kwargs = kwargs
        
        x_value, self.x_unit = get_value_unit(x)
        y_value, self.y_unit = get_value_unit(y)
        
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
        unit = get_value_unit(x)[1] 
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
        return __class__(self.y, self.x, scale_x=(self.f_y, self.inv_f_y), scale_y=(self.f_x, self.inv_f_x), **self.kwargs)


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
        
        



def integrator_unit_support(integrator: callable):
    """装饰一个 用于函数积分 的函数 integrator(f,a,b, ...) -> (res, err, ...)，使其参数 f,a,b 和返回值都支持带单位。
    integrator 例如 scipy.integrate 下的 quad, quad_vec 等
    """
    @wraps(integrator)
    def wrapped_integrator(f, a, b, x_unit=None, *args, **kwargs):
        """包装 integrator，使其参数 f,a,b 和返回值都支持带单位。
        
        x_unit 用于指定积分时，自变量 x 应当在什么单位下分解为 value 和 unit （原则上不影响结果）。默认为 None 时，采用 a 的单位。
        """
        # 剥离 x 轴的值和单位。如果给定了 x_unit 则按照它来剥离，否则使用 a 的单位。
        a_value, x_unit = get_value_unit(a, target_unit=x_unit)  # 至此，x_unit 不再可能是 None。要么是某个单位，要么是 1
        b_value, b_unit = get_value_unit(b, target_unit=x_unit)  #* 必须保证 b 和 a 的单位是一致的！所以转换到 x_unit
        
        f_unit = get_value_unit(f(a))[1]  # 从 f(a) 中获取 f 的单位（假定 f 对于以 x_unit 的所有输入都给出同样的输出单位）
        res_unit = f_unit * x_unit
        
        # 包装 f，使其接收和返回值都不带单位。不用区分 astropy 和 unyt，因为它们都有 .value 属性
        if f_unit == 1:
            integral_func = lambda x: f(x * x_unit)
        else:
            integral_func = lambda x: f(x * x_unit).value
        
        res, err, *rest = integrator(integral_func, a_value, b_value, *args, **kwargs) # 调用 integrator 执行积分

        # 处理返回值：把 res 和 err 都附上单位
        res *= res_unit
        err *= res_unit
        
        return res, err, *rest
    
    return wrapped_integrator



# 辅助函数，用于在自变量 log 尺度下做积分
# \int f dx = \int f * x d(ln(x))
# 适用于自变量天然地以 log 尺度分布的情况，普通的积分在 log(x) 轴的 sample 很不均匀

def trapz_log(y, x, *args, **kwargs):  # 目前只支持了 x 给定的情况。没有支持 x 不给出，依靠 dx 计算的情况。
    """scipy 的 trapezoid 积分函数的变种，在 x 轴的 log 尺度下做积分。适用于 x 数组天然地以 log 尺度分布的情况。

    y 和 x 都可以含有单位，支持 astropy 和 unyt。  
    x 应当是正数。但目前不做检查（为了性能）。
    """
    
    # scipy 的 trapezoid 函数本身就支持带单位输入。只是 np.log 不支持，所以只需要把 x 拆开即可。
    # 又因为 log(x) 的差分与 x 的单位（或乘数）无关，所以只需要任意单位下 x 数组的值即可。单位由 y * x 决定。
    x_value = get_value(x)
    return integrate.trapezoid(y * x, np.log(x_value), *args, **kwargs)

quad_vec_unit: Callable = integrator_unit_support(integrate.quad_vec)

@integrator_unit_support
def quad_vec_log(f, a, b, *args, **kwargs):
    """scipy 的 quad_vec 积分函数的变种，在 x 轴的 log 尺度下做积分。  
    适用于 x 变量天然地跨越多个数量级的情况。  
    积分限 a, b 都应该是正数。
    """
    if a <= 0 or b <= 0:
        raise ValueError("Both 'a' and 'b' must be positive for log-scale integration.")
        
    def integrand(log_x):
        x = np.exp(log_x)
        return f(x) * x
    return integrate.quad_vec(integrand, np.log(a), np.log(b), *args, **kwargs)



# 用于格式的 utils

from astropy.units import Quantity
from typing import Literal

def quantity_to_latex(quan: Quantity,
              formatter=None,
              p=None,
              *,
              format: Literal['latex'] | Literal['latex_inline'] | None = 'latex_inline',
              subfmt=None,
              unit=None,
              ) -> str:
    """便捷地把 Quantity 转换为 LaTeX 字符串  
    是 Quantity.to_string 的一个包装，在参数的顺序和默认值上做了一些调整
    
    p 为 precision 参数的简写
    format 默认为 'latex_inline'
    """
    
    return quan.to_string(format=format, unit=unit, precision=p, subfmt=subfmt, formatter=formatter)