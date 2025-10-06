# 图 3

from functools import partial

import numpy as np
import astropy.units as u
from astropy.units import Quantity

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.colors import LogNorm


from ..A_V_limits import Constraint
from ..utils import quantity_to_latex


class ParasSurveyPlot: 
    def __init__(
        self,
        *,
        gamma_array: np.ndarray,
        n_0_array: Quantity,
        
    ):
        self.gamma_array = gamma_array
        self.n_0_array = n_0_array

    def set_style(self):
        plt.rcdefaults()
        plt.rcParams['font.size'] = 12
        plt.rcParams['legend.fontsize'] = 'small'

    def plot_A_V_max(self, *, fig: Figure, ax: Axes, A_V_max_array: np.ndarray[float],):
        gamma_array = self.gamma_array
        n_0_array = self.n_0_array

        # 用颜色表示 A_V max
        p = ax.pcolormesh(gamma_array, np.log10(n_0_array.value), 
                        A_V_max_array, 
                        # shading='auto', 
                        shading='gouraud', 
                        cmap='viridis',
                        vmin=0,
                        # vmax=2, #TEMP 尝试统一 SMC 和 Orion 绘图的 A_V 范围
                        # edgecolors='face',
                        )
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\log_{10} n_0$ [${\rm cm}^{-3}$]")

        # colorbar
        cbar = fig.colorbar(p, ax=ax, label=r"$A_V$ limit [mag]")

        return p, cbar

    def plot_crit_wavelength(self, *, ax: Axes, crit_index_array: np.ndarray, NH_MAX_mask: np.ndarray[bool]):
        gamma_array = self.gamma_array
        n_0_array = self.n_0_array

        # 关键限制波长的分界线
        # ax.contour(gamma_array, np.log10(n_0_array.value), critical_wavelength_array.to_value(u.um), colors='k', levels=[1, 3, 5, 100])
        all_crit_indices = np.unique(crit_index_array)
        ax.contour(gamma_array, np.log10(n_0_array.value),
                #   np.reshape(crit_index_list, X.shape), 
                np.ma.array(crit_index_array, mask=NH_MAX_mask),
                levels=(all_crit_indices[1:] + all_crit_indices[:-1]) / 2,
                colors='k', linewidths=1.3,
                )

        return

    def plot_region_NH_MAX(self, ax: Axes, constraint_array: np.ndarray, ):
        gamma_array = self.gamma_array
        n_0_array = self.n_0_array

        # TEMP 画出由 NH_MAX 限制的区域
        cntr_NH_MAX = ax.contour(gamma_array, np.log10(n_0_array.value),
                constraint_array,
                levels=[(Constraint.NH_MAX+Constraint.RE_EMISSION)/2, ],
                colors="#ff0000", linewidths=1,
                )
        # cntr_NH_MAX.set(path_effects=[patheffects.withTickedStroke(angle=60, length=1, spacing=10)])

        return

    def add_text(self, ax: Axes,):
        # 用文本标出各个区域的关键限制波段
        # if True:
        #    _text_props = dict(ha='center', va='center', color='w', fontsize=11)
        #    ax.text(.16, 0.3, "ALMA", **_text_props)
        #    ax.text(.16, 3.6, "MIRI\nF2100W", **_text_props)
        #    ax.text(.16, 4.6, "MIRI\nF1000W", **_text_props)
        #    ax.text(1.5, 1, r"$r_{\rm out} \geq 10~{\rm kpc}$", **_text_props)  #TEMP

        if True:
            _text_props = dict(ha='center', va='center', color='w', fontsize=10)
            ax.text(.16, 0.3, "ALMA", **_text_props)
            ax.text(.16, 4.0, "MIRI\nF2100W", **_text_props)
            ax.text(.16, 4.75, "MIRI\nF1000W", **_text_props)
            ax.text(1.5, 1, r"$r_{\rm out} \geq 10~{\rm kpc}$", **_text_props)  #TEMP

        return

    def plot_M_dust_contour(self, *, ax: Axes, M_dust_array: Quantity, NH_MAX_mask: np.ndarray[bool],):
        gamma_array = self.gamma_array
        n_0_array = self.n_0_array

        # 对应于最大 A_V 的 M_dust 的 contour
        cntr = ax.contour(gamma_array, np.log10(n_0_array.value),
                #   np.where(X<1.35, M_dust_array.to_value(u.Msun), np.nan),
                np.ma.array(M_dust_array.to_value(u.Msun), mask=NH_MAX_mask, fill_value=np.nan),
                #    cmap='Reds', 
                levels=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                colors='white',
                linewidths=.6,
                linestyles='--',
                norm=LogNorm(),
                )
        fmt = partial(quantity_to_latex, formatter='e')  # 用于格式化 Quantity  # 取 e 是
        _clabel_props = dict(use_clabeltext=True, fontsize=10,)
        cntr.clabel(levels=cntr.levels[:1], fmt=lambda M: rf'$M_{{\rm dust}}$ = {fmt(M * u.Msun)}', **_clabel_props)  # 把第一条线单独拎出来，写全变量和单位。后面就只写数字
        cntr.clabel(levels=cntr.levels[1:], 
                    fmt=lambda M: fmt(M * u.one), 
                    # fmt=lambda M: f'{M:.0e}',  # 只写数字
                    # fmt=lambda M: print(M),
                    **_clabel_props, 
                    # manual=[[.2, 1], [.2, 1.5], [.2, 2], [.2, 2.4], [.2, 2.7],],  # 手动指定标签位置,
                    # manual=[]  # 手动指定标签位置,
                    )  #TODO: 怎么让它只显示 10^x，不要 k * 10^x ? cntr.cvalues

    def add_markers_for_paras_list(self, ax: Axes, paras_list: list[tuple[float, Quantity]],):
        typical_gammas, typical_n_0s = zip(*paras_list)
        typical_n_0s = Quantity(typical_n_0s)
        ax.scatter(typical_gammas, np.log10(typical_n_0s.value), 
                        marker='*', facecolors='none', color="#ff9650", lw=1.5, s=10**2, 
                        label=r'$(\gamma, n_0)$ in Fig.2b', zorder=100, 
                        clip_on=False,
                        )

        ax.legend(loc='best', framealpha=.8, bbox_to_anchor=(.99, .99))

    def plot_figure(
        self,
        *,
        A_V_max_array: np.ndarray[float],
        crit_index_array: np.ndarray[int],
        constraint_array: np.ndarray[Constraint],
        M_dust_array: Quantity,
        paras_list: list[tuple[float, Quantity]] | None = None,  # (gamma, n_0) 列表
    ):

        NH_MAX_mask = (constraint_array==Constraint.NH_MAX)
        # NH_MAX_mask = ((constraint_array==Constraint.NH_MAX) | (X>1.5)) & (Y<1e4*u.cm**-3)

        self.set_style()

        fig, ax = plt.subplots(
            layout='constrained',
            #   figsize=(6.4, 4.8)
        )

        p, cbar = self.plot_A_V_max(fig=fig, ax=ax, A_V_max_array=A_V_max_array)
        self.plot_region_NH_MAX(ax=ax, constraint_array=constraint_array)
        self.plot_M_dust_contour(ax=ax, M_dust_array=M_dust_array, NH_MAX_mask=NH_MAX_mask)
        self.plot_crit_wavelength(ax=ax, crit_index_array=crit_index_array, NH_MAX_mask=NH_MAX_mask)

        # self.add_text(ax=ax)
        
        # 与 Fig.2b 中的几个典型参数对应的点
        if paras_list is not None:
            self.add_markers_for_paras_list(ax=ax, paras_list=paras_list,)

        # #TEMP 尝试 T_out
        # ax.contour(gamma_array, np.log10(n_0_array.value),
        #            T_out_array.to_value(u.K),
        #         #    cmap='Reds',
        #          #   levels=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
        #          levels=[30, 50, 100, 200, 300, 400, 500],
        #            colors='orange',
        #            linewidths=.6,
        #            linestyles='--',
        #            norm=LogNorm(),
        #            )

        # * 需要根据不同的图在外部进行的操作：set_title, savefig, 加上 text
        
        return fig, ax

        # fig.savefig(f'figures/paras_survey/{_opacity_name}_Tfloor=30_large_6.png', bbox_inches='tight')
        # fig.savefig(f'figures/paras_survey/{_opacity_name}_Tfloor=30_large_6.pdf', bbox_inches='tight')

        # fig.savefig(f'figures/paras_survey/ignore_UV/{_opacity_name}_Tfloor=30_ignore_UV_6.pdf', bbox_inches='tight')
        # fig.savefig(f'figures/paras_survey/ignore_UV/{_opacity_name}_Tfloor=30_ignore_UV_6.png', bbox_inches='tight')

        # fig.savefig(f'figures/paras_survey/Delvecchio_{_opacity_name}_Tfloor=30_1.pdf', bbox_inches='tight')
