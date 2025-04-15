# LRD Spectrum TODOs





- [ ] 尝试不同的 $n_0, \gamma$ 组合下，给出的 $A_V$ 上限，尝试几个典型的组合
- [ ] 画出  $n_0, \gamma$ 参数空间图上的图
- [ ] 记录周二和 Kohei 聊的内容，关于项目的目标、讨论部分



## Coding

- [x] Git 管理
  - [x] 使用 github desktop
  - [x] 存档之前的代码
  - [x] 更改文件层级
  - [x] 创建 github repo，与政融分享
  - [x] 把 project notes 和 TODOs 上传
- [ ] **单位 Units**
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
  - [x] **把 A_V 也用 mag 作为单位？**
    - [ ] `_repr_latex_` 方便吗？
    - [x] `u.mag` 作为单位怎么处理？
  - [ ] **给「入口」函数加 equivalencies 自动转换**
- [ ] B87 model
  - [x] 做了哪些假设、近似？写进 doc
  - [x] 重新推导：把 IR flux 的量纲弄对
  - [ ] 重新推导：让 beta 成为可变参数，把所有系数对 beta 的依赖搞明白
- [ ] 代码整理
  - [x] 弄清楚之前的 UV_flux IR_flux 的具体单位，写注释
  - [ ] 梳理继承关系 ？
  - [ ] 去掉一些不必要的继承？
  - [x] 重新写 repr 和 latex repr
  - [ ] 把 A_V_model 改造成一个类
    - [ ] 记录 A_V 属性作为实例属性
  - [ ] 把 SED 改造为 dataclass？规范接口
  - [ ] *解决 warning*
  - [ ] 改名
    - [ ] 给 UV_Flux 和 IR_Flux 改名
    - [ ] 给 OrionLRDModel 改名





- [ ] **解决 nu 和 wavelength 升降序造成积分带负号的问题**
- [ ] **为什么我跑了一下，和政融的 T_out 不一致？**
- [ ] **能量守恒问题**
  - [ ] 把 A_V 对应吸收的总能量、IR 发射的总能量都算出来，作为参考
- [ ] **feedback 的处理**
  - [ ] 总光深 < 1 怎么处理？
  - [ ] 光学厚的 thin layer，能否用单温度处理？
  - [ ] thin layer 模型是否有收敛？
- [ ] **写一个最朴素的 single layer model**
- [ ] 潜在的问题：目前各种函数（n_profile, UV_Flux 等）都没有限定 r_in < r < r_out
