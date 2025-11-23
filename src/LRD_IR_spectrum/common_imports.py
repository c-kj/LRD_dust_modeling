"""
用于便捷地在脚本和 Jupyter notebook 中一键导入常用的包。  
不包含我自己写的 LRD_IR_spectrum 模块中的任何部分。

使用方法：
```python
from LRD_IR_spectrum.common_imports import *
```
"""
# ruff: noqa: F401, E402  # 忽略未使用的导入和导入位置警告，因为这是一个专门用于导入的模块

from functools import partial
from itertools import product

import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import seaborn as sns

import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity
from astropy.table import Table, QTable
from astropy.visualization import quantity_support
quantity_support()  

from shapely import LineString

# 用于并行 paras survey
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

__doc__ += f"导入的名称有：\n{', '.join([name for name in dir() if not name.startswith('_')])}"