# Fig.6: PRIMA plot

# * 由之前的纯过程式版本重构而来。渐进式重构，需要哪一部分灵活可变时，再拆出来单独调用。不需要改变的部分就写死在内部。


from functools import partial
from typing import Any

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table, QTable
from astropy.cosmology import Cosmology


import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


from ..cosmology import f_nu_from_L_nu_rest
from ..A_V_limits import calc_A_V_max_for_paras
from ..A_V_model import Partial_A_V_Model
from ..incident_SED import SED, get_SED_detection
from ..opacity import OpacityData
from ..utils import quantity_to_latex
from .plot_utils import add_wavelength_obs_xaxis, set_xylabel_with_unit
from .typical_SED_plot import plot_observed_table


class PRIMA_Plot:
    def __init__(self):
        pass

    def set_style(self):    
        # np.seterr(over='ignore', divide='ignore')  # 忽略 overflow 和 divide by zero 的警告

        # import scienceplots
        # plt.rcdefaults()
        # plt.style.use(['science',])
        # # plt.rcParams['lines.linewidth'] = 1.5

        # plt.rcParams['font.size'] = 14
        # plt.rcParams['legend.fontsize'] = 'small'
        # plt.rcParams['legend.title_fontsize'] = 'small'

        # --- Config ---
        # plt.rcParams['font.size'] = 14
        # plt.rcParams['legend.fontsize'] = 12
        # plt.rcParams['legend.title_fontsize'] = 13
        # plt.rcParams['legend.frameon'] = False

        pass
    
    def plot_figure(
        self,
        *,
        sampled_points_on_contour: np.ndarray,
        A_V_max_list_on_contour: list[float],
        model_factory: Partial_A_V_Model,
        SED_table: QTable,
        cosmology: Cosmology,
        Labbe_spec_raw: QTable,
        PRIMA_sensitivity: dict[str, QTable],
        VB01_template_selected: QTable,
        opacity: OpacityData,
    ):

        self.set_style()

        fmt = partial(quantity_to_latex, p=4)  # 用于格式化 Quantity  
        cmap = plt.get_cmap('viridis')  # 使用默认的 colormap
        norm = plt.Normalize(vmin=A_V_max_list_on_contour.min(), vmax=A_V_max_list_on_contour.max())  #TEMP 这里的归一化范围需要调整

        fig, ax = plt.subplots(figsize=(6*1.618, 6), layout='constrained',)

        # ---------------------------------- 主 Panel --------------------------------- #

        # ax.set_title(rf"Maximizing $A_V$ for different $(\gamma, n_0)$ with $M_{{\rm dust}} = 10^5 \, M_\odot$")

        for (gamma, lg_n_0), A_V in zip(sampled_points_on_contour, A_V_max_list_on_contour):
            n_0 = 10**lg_n_0 * u.cm**-3
            model = model_factory(n_0=n_0, gamma=gamma, A_V=A_V)
            # TEMP 临时补丁
            # print(model.M_gas * dust_to_gas_mass_ratio)
            # if abs(model.M_gas * dust_to_gas_mass_ratio / (1e5 * u.Msun) - 1) > .2:
            #     continue
            color = cmap(norm(A_V))

            z = SED_table.meta['z']
            f_nu = f_nu_from_L_nu_rest(L_nu_rest=model.calc_L_nu(), cosmology=cosmology, z=z, magnification=SED_table.meta['magnification'])
            ax.loglog(model.opacity.wavelength * (1+z), f_nu.to(u.uJy), 
                    color=color, lw=2, alpha=1)

        _table = SED_table.copy()
        _table['f_nu'] = f_nu_from_L_nu_rest(L_nu_rest=_table['Lnu_rest'], cosmology=cosmology, z=_table.meta['z'], magnification=_table.meta['magnification'])
        _line_observed_sed, _handle_detection, _handle_nondetection = plot_observed_table(ax, x='wavelength_obs', y='f_nu', table=_table)
        _handle_detection.set_label('JWST NIRCam/MIRI')  # 设置图例，以便后续使用
        _handle_nondetection.set_label('Herschel/ALMA (non-detection)')

        ax.set_xlim(*[5e-1, 2e3] * u.um)
        ax.set_ylim(*[1e-2, 2e4] * u.uJy)

        set_xylabel_with_unit(ax, xlabel=r"Observed $\lambda$", ylabel=r"$f_\nu$")
        
        
        # TEMP
        # VB01 incident SED template
        [_handle_VB01] = ax.plot(VB01_template_selected['wavelength_rest'] * (1+SED_table.meta['z']), VB01_template_selected['f_nu'] * 2, 
                                 label='VB01 AGN template', color="#a7dbd8", lw=1.5)


        # incident SED
        Labbe_spec = Labbe_spec_raw  # 忘了这俩有啥区别了，暂时这么写
        wavelength_rest = Labbe_spec['wave'] / (1 + SED_table.meta['z'])
        A_rel = opacity.interp_A_rel(wavelength_rest.to(u.Hz, u.spectral()))
        A_V_deredden = 0.6
        [_handle_Labbe_spec_deredden] = ax.step(Labbe_spec['wave'],
                Labbe_spec['flux'] * 10**(0.4 * A_V_deredden * A_rel),
                where='pre',
                ls='--', color='k', lw=.8,
                label=rf'Incident spectrum (corrected with $A_V$ = {A_V_deredden})',
                )

        # for A_V in [0.5, 1]:
        #     de_reddened_SED = de_redden_SED(observed_SED=get_SED_detection(SED_table), A_V=A_V, opacity=opacity)
        #     ax.plot(de_reddened_SED.wavelength * (1+SED_table.meta['z']),
        #         f_nu_from_L_nu_rest(L_nu_rest=de_reddened_SED.L_nu, cosmology=cosmo, z=SED_table.meta['z'], magnification=SED_table.meta['magnification'])
        #     )


        # # PRIMA 的 sensitivity curve
        # _handle_PRIMA_sensitivity = []
        # for key, table in PRIMA_sensitivity.items():
        #     _line, = ax.plot('lambda', 'f_nu', '.-', data=table, label=f"{fmt(table.meta['resolution'])}, {fmt(table.meta['exposure_time'])}", lw=2,)
        #     _handle_PRIMA_sensitivity.append(_line)

        # arxiv 第一版中使用的 PRIMA 曲线，单独一条
        PRIMA_table = PRIMA_sensitivity['10arcmin^2_20h']
        _handle_PRIMA_sensitivity = ax.plot('lambda', 'f_nu', '.-', data=PRIMA_table,
                                 label=f"{fmt(PRIMA_table.meta['resolution'])}, {fmt(PRIMA_table.meta['exposure_time'])}",
                                 lw=3, color="#F48C4B", marker='D', markersize=6,)


        # # Ichikawa proposal 中的 PRIMA sensitivity，分为 shallow 和 deep
        # _handle_PRIMA_sensitivity = []
        # _line_shallow, = ax.plot('wavelength_obs', 'f_nu_shallow', data=PRIMA_sensitivity_Ichikawa,
        #         label='shallow', ls='-', marker='s', lw=2, color='orange')
        # _line_deep, = ax.plot('wavelength_obs', 'f_nu_deep', data=PRIMA_sensitivity_Ichikawa,
        #         label='deep', ls='-', marker='d', lw=2, color='orange')

        # _handle_PRIMA_sensitivity.append(_line_shallow)
        # _handle_PRIMA_sensitivity.append(_line_deep)


        # Labbe 观测光谱
        _handle_Labbe_spec, = ax.step('wave', 'flux', data=Labbe_spec_raw, 
                                      where='pre', label='JWST PRISM', color='k', alpha=1, lw=1)


        # 其他光谱（作为背景）
        if _plot_background_spectra := False:
            _handle_background_spectra = []
            for name, observed_flux in background_spectra.items():
                ref_table = Labbe_spec_raw
                z = observed_flux.meta['z']
                normalized_flux = observed_flux['flux'] / LogLogInterpolator(observed_flux['wave'] / (1+z), observed_flux['flux'])(5470*u.AA) * LogLinearInterpolator(ref_table['wave'] / (1+ref_table.meta['z']), ref_table['flux'])(5470*u.AA)  # 归一化到 5470 Å 的值
                _line, = ax.plot('wave', normalized_flux, data=observed_flux, label=name, alpha=.8, lw=1)
                ax.scatter(5470*u.AA * (1+z), LogLinearInterpolator(ref_table['wave'] / (1+ref_table.meta['z']), ref_table['flux'])(5470*u.AA), s=10**2, marker='*', zorder=100, color=_line.get_color(), edgecolor='k', )  # 在 5470 Å 处标记
                _handle_background_spectra.append(_line)

        # 图例
        _legend_observation = ax.legend(handles=[_handle_detection, _handle_nondetection, _handle_Labbe_spec, _handle_Labbe_spec_deredden, _handle_VB01], 
                                        # labels=['JWST NIRCam/MIRI', 'Herschel/ALMA (non-detection)', _handle_Labbe_spec.get_label(), _handle_Labbe_spec_deredden.get_label()],
                                        loc='upper left',
                                        title=rf'A2744-45924 ($z_{{\rm spec}} = {SED_table.meta["z"]}$)')
        _legend_observation.get_title().set_horizontalalignment('left')
        ax.add_artist(_legend_observation)

        ax.add_artist(
            ax.legend(handles=_handle_PRIMA_sensitivity, title='PRIMA sensitivity', 
                    loc='upper right',
                    #   loc=[.04, .62],
                    )
        )

        if _plot_background_spectra:
            ax.legend(handles=_handle_background_spectra, title='Background spectra', loc='upper right', framealpha=.5,)

        # ---------------------------------------------------------------------------- #

        # Colorbar
        if True:
            mappable = ScalarMappable(cmap=cmap, norm=norm)
            # mappable.set_array(A_V_list)

            # 分割主图区域，创建 colorbar 轴
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size='3%', pad='2%')
            cbar = fig.colorbar(mappable, cax=cax, label=r"$A_V$ [mag]", pad=0.01)

        # 底部标示各个仪器的波长范围
        import matplotlib.patches as patches

        ranges = [
            # (0.5, 2, 'HST', 'lightblue', 0.2, 0.3),
            (0.623, 4.981, 'JWST/NIRCam', 'lightgreen', 0.2,0.8),
            (5.054, 26.733, 'JWST/MIRI', "#ffe644", 0.2,0.8),
            # (4, 200, 'Spitzer', 'sandybrown', 0.2,0.3),
            (24, 235, 'PRIMA', "#ff9f68", 0.4, 0.8),
            (60, 670, 'Herschel', 'gray', 0.2, 0.8),
            # (400,1200, 'SCUBA2', 'tomato', 0.2,0.3),
            (300, 3300, 'ALMA', 'indianred', 0.2,0.8),  # 这里 ALMA 的范围上限是手动设置的，为了让 ALMA text 位置好看
        ]

        for start, end, label, color, y_pos, _height in ranges:
            y_pos = y_pos / 15.
            height = y_pos * .8
            rect = patches.Rectangle((start, y_pos), end-start, height=height, color=color, alpha=0.5, label=label)
            ax.add_patch(rect)

            # 获取矩形颜色并加深
            rect_color_rgba = rect.get_facecolor()
            darker_color_rgb = [c * 0.6 for c in rect_color_rgba[:3]] # 加深RGB分量
            ax.text(np.sqrt(start * end), np.sqrt(y_pos * (y_pos + height)), rf'\textbf{{{label}}}', ha='center', va='center', color=darker_color_rgb, fontsize=11, weight='bold')

        # fig.savefig('figures/PRIMA/M_dust=1e5_incident_1.pdf', bbox_inches='tight')
        # // ...existing code...
        
        self.fig = fig
        self.ax = ax