# Fig.2: Typical SEDs with different parameters

# * 由之前的纯过程式版本重构而来。渐进式重构，需要哪一部分灵活可变时，再拆出来单独调用。不需要改变的部分就写死在内部。


from functools import partial
from typing import Any

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table, QTable

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from ..A_V_limits import calc_A_V_max_for_paras
from ..A_V_model import Partial_A_V_Model
from ..incident_SED import SED, get_SED_detection
from ..opacity import OpacityData
from ..utils import quantity_to_latex
from .plot_utils import add_wavelength_obs_xaxis, set_xylabel_with_unit

np.seterr(over='ignore', divide='ignore')  # 忽略 overflow 和 divide by zero 的警告

type detectionType = bool


def plot_observed_table(
    ax: Axes,
    *,
    x: str,
    y: str,
    table: QTable,
    observed_sed_line_label: str = 'Observed SED (A2744-45924)',
    marker_props_dict: dict[detectionType, dict] = {},
):
    """绘制观测数据表。分别画 detection 和 non-detection 的 marker 以及连起来的粗线
    
    table 要包含 x,y, 'detection' 列，table.meta 中要含有 'z' 信息
    """
    
    # marker 的默认属性，分别设置 detection 和 non-detection 的
    marker_props_default_dict: dict[detectionType, dict[str, Any]] = {
        True : dict(marker='o', color='r', edgecolor='k', lw=1, s=6**2,),
        False: dict(marker='v', color='r', edgecolor='k', lw=1, s=6**2 * 1.3),
    }

    scatter_handles: dict[detectionType, Artist] = {}  # 收集 scatter 的 handles

    # 观测数据
    for is_detected in [True, False]:  # 分别绘制 detection 和 non-detection 的数据
        mask = table['detection'] == is_detected
        # 使用不同的 marker 属性
        marker_props = marker_props_default_dict[is_detected]
        marker_props |= marker_props_dict.get(is_detected, {})  # 覆盖默认属性

        _scatter_handle = ax.scatter(
            x=x, y=y, data=table[mask],
            # s=6**2,
            **marker_props,
            label={True: 'detection', False: 'non-detection'}[is_detected],
            zorder=5
        )

        scatter_handles |= {is_detected: _scatter_handle}

    # NIRCam 和 MIRI 的 detection 数据，连成粗线
    mask_sed_line = table['detection']
    observed_sed_line, = ax.plot(x, y, data=table[mask_sed_line],
            color='orange', alpha=.3, lw=6,
            label=observed_sed_line_label
            )

    # 设定坐标轴的范围
    ax.set_xscale('log')
    ax.set_yscale('log')

    # ax.grid(which='both', alpha=.1)  # 添加网格线

    ax.yaxis.set_ticks_position('both')  # 让右侧 y 轴也显示 ticks

    return observed_sed_line, scatter_handles[True], scatter_handles[False]


class TypicalSEDPlot:

    def __init__(
        self,
        *, 
        model_factory: Partial_A_V_Model,  # model_factory 的 opacity 和 observed_SED 参数会被覆盖
        opacity: OpacityData,
        SED_table: QTable,
        spec_table: QTable | None = None,
        fmt = partial(quantity_to_latex, formatter='e'),  # 用于格式化 Quantity
        cmap=plt.get_cmap('viridis'),  # 使用默认的 colormap
        norm=plt.Normalize(vmin=0, vmax=1.15),  #TEMP 这里的归一化范围需要调整
    ):
        
        # 配置在这张图（多个 panel）之间通用的参数
        
        self.opacity = opacity
        self.SED_table = SED_table
        self.spec_table = spec_table
        
        self.fmt = fmt
        self.cmap = cmap
        self.norm = norm

        # 配置局部使用的 model_factory 和 _calc_A_V_max
        self.model_factory = partial(
            model_factory,
            opacity=opacity,
            observed_SED=get_SED_detection(SED_table),
        )

        self._calc_A_V_max = partial(
            calc_A_V_max_for_paras,
            model_factory=self.model_factory,
            constraint_SED=SED.from_QTable(SED_table),
        )

        
    def set_style(self):
        plt.rcParams['font.size'] = 14
        plt.rcParams['legend.fontsize'] = 'small'
        plt.rcParams['legend.title_fontsize'] = 'small'
        plt.rcParams['legend.edgecolor'] = 'none'
        
    
    def plot_background(self, ax: Axes):
        """用于在这两个 panel 中绘图"""
        table = self.SED_table
        
        ret = plot_observed_table(ax, x='wavelength_rest', y='nu_L_nu', table=table)
        
        if self.spec_table is not None:
            [self.handel_spec] = ax.step('wavelength_rest', 'nu_L_nu', data=self.spec_table, 
                    label='JWST PRISM', 
                    where='pre', color='k', alpha=.3, lw=1, zorder=-100)  # 放在最后面
        
        ax.set_xlim(*[1e-1, 3e2] * u.um)
        ax.set_ylim(*[1e43, 1e47] * u.erg/u.s)
        add_wavelength_obs_xaxis(ax, z=table.meta['z'])  # 顶部的观测波长轴
        set_xylabel_with_unit(ax, xlabel=r"Rest-frame $\lambda$", ylabel=r"$\nu L_\nu$")  # 坐标轴 label

        return ret
    
    def plot_panel_A(
        self,
        *,
        ax: Axes,
        A_V_list: np.ndarray = np.arange(.2, 1.+1e-8, .2),
        n_0: Quantity,
        gamma: float,
        A_V_high: float | None = None,  # 设定一个值，展示高 A_V 有多「糟糕」
    ):
        """Changing A_V. (n_0, gamma) 保持不变"""
        
        fmt = self.fmt
        cmap = self.cmap
        norm = self.norm

        props_incident = dict(alpha=1, ls='--')
        props_reprocessed = dict(alpha=1, ls='-')

        handles_ax0_right = [] 

        for A_V_val in A_V_list if A_V_high is None else np.append(A_V_list, A_V_high):  # 如果指定了 A_V_high，则也绘制它
            model = self.model_factory(A_V=A_V_val, n_0=n_0, gamma=gamma)  # 固定 (n_0, gamma)
            color = cmap(norm(A_V_val))
            
            ax.plot(model.incident_SED.wavelength, model.incident_SED.nu_L_nu.to(u.erg/u.s), color=color, **props_incident)  # de-redden 入射光谱
            
            _line, = ax.loglog(model.opacity.wavelength, model.calc_nu_L_nu(), color=color, label=rf'{A_V_val:.1f} mag', **props_reprocessed)  # IR re-emission
            if A_V_val == A_V_high:  # 把 A_V_high 的线单独设定样式
                _line.set(alpha=1, ls=':')
            
            handles_ax0_right.append(_line)
            
            

        ax.set_title(rf"Changing $A_V$. $(\gamma, n_0)=$ ({model.gamma}, {fmt(model.n_0)})")  # 直接取最后一个模型的参数，反正它们都一样

        # 绘制背景并获取图例句柄
        _line_observed_sed, _handle_detection, _handle_nondetection = self.plot_background(ax)
        _handle_incident_sed = Line2D([], [], color='k', **props_incident, label='incident SED')
        _handle_reprocessed_sed = Line2D([], [], color='k', **props_reprocessed, label='re-emitted SED') 

        handles_ax0_left = [
            _line_observed_sed, 
            _handle_detection, 
            _handle_nondetection,
            _handle_incident_sed, 
            _handle_reprocessed_sed, 
        ]
        
        if self.spec_table is not None:
            handles_ax0_left.insert(1, self.handel_spec)

        # 双图例
        bbox_to_anchor_y = 0.985
        legend_left = ax.legend(handles=handles_ax0_left, loc='upper left', bbox_to_anchor=(0.10, bbox_to_anchor_y)) 
        ax.add_artist(legend_left)   # 手动添加，从而能添加另一个图例

        legend_right = ax.legend(handles=handles_ax0_right[::-1], # 反序，这样图例的上下顺序与 SED 曲线一致
                    loc='upper right', bbox_to_anchor=(0.98, bbox_to_anchor_y), title=r"$A_V$", )
        ax.add_artist(legend_right)
        
        return handles_ax0_left, handles_ax0_right
    

    def plot_panel_B(
        self,
        *, 
        ax: Axes,
        paras_list: list[tuple[float, Quantity]],
    ):
        """绘制 Panel B 的内容
        
        paras_list: (gamma, n_0) 列表
        """
        bbox_to_anchor_y = 0.985

        fmt = self.fmt
        cmap = self.cmap
        norm = self.norm

        handles_ax1_right = []

        ax.set_title(rf"Maximizing $A_V$ for different $(\gamma, n_0)$") 

        for gamma, n_0 in paras_list:
            A_V = self._calc_A_V_max(gamma=gamma, n_0=n_0).A_V_max
            model = self.model_factory(n_0=n_0, gamma=gamma, A_V=A_V)
            color = cmap(norm(A_V))

            _line_2, = ax.loglog(model.opacity.wavelength, model.calc_nu_L_nu(), 
                          label=rf'{A_V:.2f} mag ($\gamma$={gamma:.2f}, $n_0$={fmt(n_0)})', color=color)
            handles_ax1_right.append(_line_2)

            # IR re-emission + observed SED:
            if False: 
                props_total = dict(alpha=.8, ls='--')
                # axs[1].loglog(model.opacity.wavelength, model.calc_nu_L_nu_total(), color=color, **props_total)
                interp_nu_L_nu = LogLogInterpolator(model.observed_SED.wavelength, model.observed_SED.nu_L_nu, fill_value='extrapolate')  #TEMP 用 log-log 外插
                axs[1].loglog(model.opacity.wavelength, model.calc_nu_L_nu() + interp_nu_L_nu(model.opacity.wavelength), color=color, **props_total)
                axs[1].plot([], [], c='k', **props_total, label='observed + re-emission SED')  # 为了给 summation 的线条添加图例

        self.plot_background(ax) 

        # 图例
        ax.legend(handles=handles_ax1_right, loc='upper left', bbox_to_anchor=(0.02, bbox_to_anchor_y), title=r"maximum allowed $A_V$", )
        
        return ax
    
    def add_colorbar(
        self,
        *, 
        fig: Figure,
        axs: list[Axes],
        A_V_list: np.ndarray = np.arange(.2, 1.+1e-8, .2),
        label: str = r"$A_V$ [mag]",
        pad: float = 0.01,
    ):
        """添加 colorbar"""
        cmap = self.cmap
        norm = self.norm

        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(A_V_list) 
        cbar = fig.colorbar(mappable, ax=axs, label=label, pad=pad)
        
        return cbar
    
    def plot_figure(
        self, 
        *, 
        A_V_list = np.arange(.2, 1.+1e-8, .2),  # panel A 中的 A_V 列表
        panel_A_paras = dict(n_0 = 1e3 * u.cm**-3, gamma = 0.5),
        paras_list: list[tuple[float, Quantity]],  # (gamma, n_0) 列表
        A_V_high: float | None = None,  # 设定一个值，展示高 A_V 有多「糟糕」
        ):
        """组装各部分，从头绘制图 2."""
        
        self.set_style()
        
        #* 目前只是创建，但并不持有 fig, axs，因为暂时没有这个需求
        fig, axs = plt.subplots(ncols=2, figsize=(8*2, 6), layout='constrained',)
        axs: list[Axes]
        
        self.plot_panel_A(ax=axs[0], A_V_list=A_V_list, **panel_A_paras, A_V_high=A_V_high)
        self.plot_panel_B(ax=axs[1], paras_list=paras_list)
        
        if False:  # 目前不启用 colorbar
            self.add_colorbar(fig, axs, label=r"$A_V$ [mag]", pad=0.01)
            
        return fig, axs
