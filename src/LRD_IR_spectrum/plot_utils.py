from matplotlib.axes import Axes 
from matplotlib.axis import Axis
import astropy.units as u



def set_axis_label_with_unit(axis: Axis,
                             label: str | None = None,
                             bracket: str = '[]',
                             ):
    #* 需要开启 astropy 的 quantity_support()
    
    assert len(bracket) == 2, f"bracket should be a string with length 2, but got {bracket}"
    lb, rb = bracket
    
    if label is not None:
        if axis.get_units().physical_type == 'dimensionless':
            tail = ''
        else:
            tail = f" {lb}{axis.get_label_text()}{rb}"
        axis.set_label_text(f"{label}{tail}")

def set_xylabel_with_unit(ax: Axes, 
                          xlabel: str | None = None, 
                          ylabel: str | None = None, 
                          bracket: str = '[]'):
    
    set_axis_label_with_unit(ax.xaxis, xlabel, bracket=bracket)
    set_axis_label_with_unit(ax.yaxis, ylabel, bracket=bracket)
    

# 用于加另一个横轴的函数

def wavelength_to_temperature(x):
    """波长和温度的相互转换。 x 既可以是波长，也可以是温度"""
    #* 目前这个函数只接受 micron or K 作为单位，用的是值而非带单位的量
    from astropy.constants import c, h, k_B
    return (h * c / k_B).to_value(u.K * u.micron) / x

def add_T_xaxis(ax: Axes, *, location: float | str = 'top'):
    
    T_axis = ax.secondary_xaxis(location=location, functions=(wavelength_to_temperature, wavelength_to_temperature))
    T_axis.set_xlabel(r"$T$ [K]")
    return T_axis
    
def add_wavelength_obs_xaxis(ax: Axes, *, z: float, location: float | str = 'top'):
    """添加一个新的 x 轴，表示观测波长"""
    wavelength_obs_axis = ax.secondary_xaxis(location=location, functions=(lambda x: x * (1+z), lambda x: x / (1+z)))
    wavelength_obs_axis.set_xlabel(r"Observed $\lambda$ [$\mu m$]")  #TEMP 这里 label 写死了
    return wavelength_obs_axis


# 自定义 legend handler

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

class HandlerTupleVertical(HandlerTuple):
    """legend key 纵向堆叠
    用 copilot 写的，回头再细看
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 两条线分别画在 legend 图标的上半和下半
        lines = []
        for i, line in enumerate(orig_handle):
            # y 坐标：上半和下半
            y = ydescent + height * (0.75 if i == 0 else 0.25)
            l = Line2D([xdescent, xdescent + width], [y, y],
                       color=line.get_color(),
                       linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth(),
                       alpha=line.get_alpha())
            lines.append(l)
        return lines