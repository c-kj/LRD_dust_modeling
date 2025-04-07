"""用于在 Jupyter Notebook 中重新加载模块，方便调试  
copy 自我的另一个项目 athena_post_processing

作者：陈可鉴
"""



import importlib
from types import ModuleType

def is_submodule(parent: ModuleType, child: ModuleType) -> bool:
    """检查 child 是否是 parent 的子模块
    方法是：检查 child 的包名是否以 parent 的包名开头
    """
    return isinstance(child, ModuleType) and child.__name__.startswith(parent.__name__)


def deep_reload(module: ModuleType, verbose=False) -> ModuleType:
    """
    递归地重新加载模块及其调用的子模块。
    其顺序是自底而上的，即先重新加载子模块，再重新加载模块本身，从而避免加载父模块时所调用的子模块还没有更新。
    
    deep_reload 只重新加载模块，不重新加载模块中的函数。所以在 `deep_reload(module)` 之后，还需要重复一遍 `from module import xxx` 的操作，这样才能把 import 进来的函数也都更新：
    
    ```python
    deep_reload(module)
    from module import function
    from module import *
    ```
    
    注意：
    1. 重新导入模块会「覆盖」相同变量名（函数名）的定义，但如果原来定义了一个变量（函数）而改动后取消了这个定义（注释掉了），那么重新导入并不会删除之。
    2. 如果子模块的名字和其内部 .py 文件的名字相同，这会造成名字的混乱，导致 reload 时无法把 from module import * 中的函数更新。因此，子模块的名字最好不要和其内部 .py 文件的名字相同。
    """
    # 递归地重新加载所有子模块
    # 遍历模块的所有属性（的名称），这包括了其中调用的所有内置模块、调用的子模块、定义的函数和变量等
    for attribute_name in module.__dir__():   # 这里不用 dir() 的原因是，dir() 是按字母序排列的，而 __dir__() 是按属性被定义的顺序排列的
        attribute_value = getattr(module, attribute_name)  # 获取该属性的值

        # 如果该属性是 module 的一个子模块，则递归地重新加载它
        if is_submodule(module, attribute_value):
            deep_reload(attribute_value, verbose=verbose)
            
    
    if verbose: # 如果 verbose 为 True，打印出正在重新加载的模块名
        print('reloading ' + module.__name__)
    # 重新加载模块本身，并返回重新加载后的模块
    return importlib.reload(module)
