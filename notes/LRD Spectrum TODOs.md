# LRD Spectrum TODOs





- [ ] 尝试不同的 $n_0, \gamma$ 组合下，给出的 $A_V$ 上限，尝试几个典型的组合
- [ ] 画出  $n_0, \gamma$ 参数空间图上的图
- [ ] 记录周二和 Kohei 聊的内容，关于项目的目标、讨论部分



## Coding

- [ ] Git 管理
  - [x] 使用 github desktop
  - [x] 存档之前的代码
  - [x] 更改文件层级
  - [ ] 创建 github repo，与政融分享
  - [ ] 把 project notes 和 TODOs 上传
- [ ] 单位 Units
  - [ ] 学习 astropy 的 units 文档
    - [ ] 对数单位怎么用？
    - [ ] equivalencies 怎么用？spectral_density 和 spectral 的区别？
    - [ ] 类型标注，参数检查
    - [ ] ……
  - [ ] 如何处理积分等不支持单位的计算？
    - [ ] 手动剥离单位，计算，再加上单位？
    - [ ] 改造积分函数？
    - [ ] 写一个辅助函数？
  - [ ] 关掉全局的 equivalencies ？
  - [ ] 统一把所有的值都尽量使用 `astropy.units`
- [ ] 代码整理
  - [ ] 弄清楚之前的 UV_flux IR_flux 的具体单位，写注释
  - [ ] 梳理继承关系 ？
- [ ] **解决 nu 和 wavelength 升降序造成积分带负号的问题**
- [ ] **为什么我跑了一下，和政融的 T_out 不一致？**
- [ ] **能量守恒问题**
  - [ ] 把 A_V 对应吸收的总能量、IR 发射的总能量都算出来，作为参考
- [ ] **feedback 的处理：怎样最好？**
- [ ] 