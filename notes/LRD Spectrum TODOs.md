# LRD Spectrum TODOs





- [ ] 尝试不同的 $n_0, \gamma$ 组合下，给出的 $A_V$ 上限，尝试几个典型的组合
- [ ] 画出  $n_0, \gamma$ 参数空间图上的图
- [ ] 记录周二和 Kohei 聊的内容，关于项目的目标、讨论部分

## 科学问题

- [ ] **Feedback 的正确处理**

  - [ ] 尤其是在 A_V 小的时候（tau_UV < 1，不存在 tau_UV = 1 的点）
  - [ ] 总光深 < 1 怎么处理？
  - [ ] 光学厚的 thin layer，能否用单温度处理？
  - [ ] thin layer 模型是否有收敛？
  - [ ] 用 $\tau_{\rm rad} = 1$ 作为堆积位置合理吗？
    - [ ] 说是有模拟文献的依据
    - [ ] 可以自己简单推一下，辐射压 vs 引力 
  - [ ] **tau_UV 怎么取？**
    - [ ] 之前是取 截面最大值。这对于更广泛的 opacity law 并不合适。
    - [ ] 取 rad pressure cross 对 incident L_nu 平均的值作为截面，然后拿这个截面算 tau_UV
      - [x] Draine 2011 书上 23.10.1 节可以作为参考
- [ ] **换不同的 opacity law**
  - [x] 加入从政融那里拿到的 .opc 数据。但是波长覆盖范围都太小了，用不了。
  - [ ] 研究 sigma_H_V 的处理
    - [x] 确认 $n_0 \sigma_{H,V}$ 的简并性
    - [ ] 如何处理？能直接给一个差不多的 sigma_H_V 值吗？但这样 n_0 能互相比较吗？
      - [ ] 最好是直接拿到物理模型给出的 $\sigma_H$ 数据，比如从 CLOUDY 中
    - [ ] $\sigma_H$ 是一个合适的量吗？它是各种大小的颗粒的综合效应吗？
  - [ ] 找到波长覆盖范围合适的几个典型模型
  - [ ] 与政融给我的模型比较。尤其注意 FUV、UV、Optical、IR 多波段的比较
  - [ ] **比较不同的 opacity 下的光谱**
- [ ] 把 Far UV 的能量加进模型 （政融 working on it）

## Coding

- [x] Git 管理
  - [x] 使用 github desktop
  - [x] 存档之前的代码
  - [x] 更改文件层级
  - [x] 创建 github repo，与政融分享
  - [x] 把 project notes 和 TODOs 上传
- [x] 单位 Units
  - [x] 学习 astropy 的 units 文档
    - [x] 对数单位怎么用？
    - [x] equivalencies 怎么用？spectral_density 和 spectral 的区别？
    - [x] 类型标注，参数检查
    - [x] ……
  - [x] 如何处理积分等不支持单位的计算？
    - [x] 手动剥离单位，计算，再加上单位？
    - [x] 改造积分函数？
    - [x] 写一个辅助函数？
  - [x] 关掉全局的 equivalencies ？
  - [x] 统一把所有的值都尽量使用 `astropy.units`
  - [x] 用 `@u.quantity_input` 装饰器来检查
    - [ ] 让`@u.quantity_input` 只检查入口、出口，不检查中间计算的函数，避免太多开销
  - [x] IR_Flux 中的积分，其他方法还没适配单位
  - [x] brentq 方法求根，看看单位处理好没有
  - [x] OrionLRDModel，还遗留了很多 .cgs.value , .to(...) 之类的
  - [x] 测速，看看加单位的开销
    - [x] 对比「把 u.quantity_input 全部注释掉」
    - [x] 对比 main 分支上的代码版本
  - [x] 把 A_V 也用 mag 作为单位？
    - [x] `_repr_latex_` 方便吗？只需 `.to(u.mag)` 即可
    - [x] `u.mag` 作为单位怎么处理？
  - [x] 给「入口」函数加 equivalencies 自动转换
- [ ] B87 model
  - [x] 做了哪些假设、近似？写进 doc
  - [x] 重新推导：把 IR flux 的量纲弄对
  - [ ] 重新推导：让 beta 成为可变参数，把所有系数对 beta 的依赖搞明白
- [ ] 代码整理
  - [x] 弄清楚之前的 UV_flux IR_flux 的具体单位，写注释
  - [ ] 梳理继承关系 ？
  - [ ] 去掉一些不必要的继承？
  - [x] 重新写 repr 和 latex repr
  - [x] 把 A_V_model 改造成一个类
    - [x] 记录 A_V 属性作为实例属性
  - [x] 把 SED 改造为 dataclass？规范接口
  - [ ] *解决 warning*
  - [ ] 改名
    - [ ] 给 UV_Flux 和 IR_Flux 改名
    - [ ] 给 OrionLRDModel 改名
- [x] OpacityData
  - [x] 允许 sigma_abs 空着，默认取 ext 的值
  - [x] 处理 sigma_H_V 的指定
  - [x] 从 A / A_V 曲线转换为 opacity law：`from_extinction_data`、`from_extinction_model`

- [ ] 数值积分的处理
  - [ ] `trapz_log` 真的比 `trpz` 更好吗？？？
  - [ ] 对于很粗略的 SED，中间部分应该假设如何插值？
  - [ ] 原理问题
    - [ ] 搞清楚数值积分的「采样」和「rule」之间的区别与联系
    - [ ] `trapz_log` 的本质意义
      - [ ]  $\int y x \, {\rm d}\log(x)$ ，相当于在 $(\log(x), yx)$ 上的梯形法则？
      - [ ] 这种变量代换法，有没有什么相关参考（wiki 、书等？）问问 LLM
      - [ ] 收敛性、收敛阶数与普通的 trapz 一致吗？
      - [ ] 这样做对于 $\log(x)$ 接近均匀的 sample 来说，真的更好吗？
    - [ ] 直接梯形法对于分段线性函数是精确的；trapz_log 对于  $(\log(x), yx)$ 图上的分段直线函数是精确的，但这个要求很古怪。






- [x] 解决 nu 和 wavelength 升降序造成积分带负号的问题
- [ ] **nu_array 应该用 opacity 的还是 SED 的？**
- [x] 为什么我跑了一下，和政融的 T_out 不一致？
- [x] **能量守恒问题**
  - [x] 把 A_V 对应吸收的总能量、IR 发射的总能量都算出来，作为参考
  - [x] 写成函数 or 类的方法
  - [ ] **为什么从 extinction 算出来的功率 和 从 UV_Flux_with_feedback 算出来的不一样？？？**

    - [ ] $\tau$ 的处理，其实可以取原有的值，这样在薄层内也可以平滑过渡了
      - [ ] 但是平滑过渡是合理的吗？
      - [ ] 如果不平滑过渡，只取 $\tau=0$ ，是否会让能量不守恒？待尝试
    - [ ] **因为积分 $ \int (...) 4\pi r^2 {\rm d}r $ 的时候，没有把这个 $r^2$ 在 $r < r_{\rm ph}$ 的地方改为 $r_{\rm ph}^2$  ！**
- [ ] **写一个最朴素的 single layer model，用于对比**
- [ ] 潜在的问题：目前各种函数（n_profile, UV_Flux 等）都没有限定 r_in < r < r_out
- [ ] SED 有部分 L_nu 为 0 的话，能用 LogLog 插值吗？？





### 要跟政融讨论的事

- [ ] opacity law 的文件格式
  - [ ] 如果没有区分 ext 和 abs ，可以不用写两列？写也行。
- [ ] 如果有函数形式，可以直接给我，不用写成文件
- [ ] 找政融问问 CLOUDY 能不能给几条 sigma_H 的数据
  - [ ] 如果能给的话，还得区分 ext vs abs，反而还有点麻烦…
