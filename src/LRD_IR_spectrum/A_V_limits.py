# calc_A_V_max 系列函数的定义
import logging
from functools import partial, lru_cache
from enum import IntEnum

import numpy as np
from scipy import optimize
import astropy.units as u
from astropy.units import Quantity

from .model_base import R_out_Error
from .A_V_model import A_V_Model, A_V_ModelFactory
from .incident_SED import SED


logger = logging.getLogger('A_V_max')
logger.setLevel(logging.DEBUG)

# CONSTRAINED_BY_NH_MAX = object()

class Constraint(IntEnum):
    RE_EMISSION = 1
    NH_MAX = 2
    M_DUST = 3

def model_constraint_diff(model: A_V_Model, constraint_SED: SED) -> tuple[int, float]:
    """计算 model 和 constraint_SED 之间的 log diff，返回 diff 最大点的 index 和 diff 值
    
    constraint_SED: 观测到的 SED（converted to rest frame），包括了 non-detection 数据（Herschel 和 ALMA）。用作 re-emission 限制
    """
    # 分别取 log 再 diff。因为 L 随 A_V 接近指数变化，取 log 能让 brentq 的收敛性好不少
    # diff = model.calc_nu_L_nu(constraint_SED.nu) - constraint_SED.nu_L_nu
    diff = np.log10(model.calc_nu_L_nu(constraint_SED.nu).to_value(u.Lsun)) - np.log10(constraint_SED.nu_L_nu.to_value(u.Lsun))
    index = np.argmax(diff)
    return index, diff[index]


# TODO 改名
def calc_A_V_max_for_model_factory(*, 
                 model_factory: A_V_ModelFactory,
                 constraint_SED: SED,
                 ):
    np.seterr(over='ignore', divide='ignore')  # 忽略 overflow 和 divide by zero 的警告
    xtol, rtol = 1e-6, 1e-3
    
    constrained_by = Constraint.RE_EMISSION
    xmin, xmax = [1e-3, 6]
    
    # @lru_cache
    def calc_index_diff(A_V: float):
        model = model_factory(A_V=A_V)
        return model_constraint_diff(model=model, constraint_SED=constraint_SED)
    
    # @lru_cache
    def calc_diff(A_V: float):
        try:
            return calc_index_diff(A_V)[1]
        except R_out_Error:  # 处理 gamma >= 1 时可能出现的 R_out_Error
            return np.inf
        
    
    # 对于会出现 R_out_Error 的情况，调整 xmin 和 xmax 让收敛更快
    # if calc_diff(xmin) == np.inf:  #TEMP 粗糙的尝试：A_V=0.1
    #     xmax_new = model_factory(
    #         n_0=n_0,
    #         gamma=gamma,
    #         A_V=xmin,
    #     ).A_V_max_from_NH_max.to_value(u.mag)
    #     xmax = min(xmax, xmax_new)
    #     xmin = 0.
    #     logger.info(f"meet R_out_Error, set {xmin = }, {xmax = }")
        
    #     constrained_by = Constraint.NH_MAX

    success = False
    while not success:
        try:
            res = optimize.root_scalar(
                calc_diff, 
                bracket=[xmin, xmax],  # 不取到 0，否则 log 的 diff 会给出 -np.inf，虽然 brentq 仍能处理，但预测能力下降，速度稍慢
                method='brentq', 
                xtol=xtol,
                rtol=rtol,  # 不用特别精确
            )
            success = True
        except ValueError as e:  # 处理 ValueError: f(a) and f(b) must have different signs
            # 这里用 .keywords 实际上假定了 model_factory 是一个 partial 对象。如果不是，可能会报错
            n_0 = model_factory.keywords['n_0']
            gamma = model_factory.keywords['gamma']
            logger.info(f"{e!r}. n_0 = {n_0}, gamma = {gamma}, A_V in {[xmin, xmax]}. {calc_diff(xmin) = }, {calc_diff(xmax) = }")
            while calc_diff(xmin) > 0:
                xmin /= 10
            while calc_diff(xmax) < 0:
                xmax *= 2
    A_V_max = res.root
    index, diff = calc_index_diff(A_V=A_V_max) 
    
    # if diff < 0 and abs(diff) > 2*(A_V_max*rtol + xtol):
    #     constrained_by = Constraint.NH_MAX
    try:
        # calc_index_diff(A_V=A_V_max + 1e-3)
        calc_index_diff(A_V=A_V_max + A_V_max*rtol + xtol)
        calc_index_diff(A_V=A_V_max + A_V_max*rtol)
        calc_index_diff(A_V=A_V_max + xtol)
        calc_index_diff(A_V=A_V_max)
    except R_out_Error:  # 说明是被 NH_max 限制住了
        constrained_by = Constraint.NH_MAX
    
    return A_V_max, res, constrained_by, index, diff, #, calc_index_diff.cache_info()  #FUTURE 目前直接以 tuple 返回，后续可以改成一个 dataclass 或 namedtuple 之类的



def calc_A_V_max_for_paras(*, gamma: float, n_0: Quantity,
                 model_factory: A_V_ModelFactory,
                 constraint_SED: SED,
                 ):
    """对于 paras survey 中的参数 (gamma, n_0), 计算其对应的 A_V_max。  
    
    实际上就是在 calc_A_V_max_for_model_factory 的基础上，覆盖其 gamma, n_0 参数。
    """
    _model_factory = partial(model_factory, n_0=n_0, gamma=gamma)  # 用 gamma, n_0 来覆写 model_factory 中的参数
    return calc_A_V_max_for_model_factory(model_factory=_model_factory, constraint_SED=constraint_SED)