# LRD Spectrum

## 目前的模型

- 观测：X-ray 有 Chandra non-detection, Optical-UV 有 V-shape, IR 有 MIRI 的一个波段的数据，非常长波的有 ALMA (non-detecion?)
- Slim disk （我算的部分）+ Dust attenuation $A_V$ (Orion 模型，而非更常用的 xxx)，可以解释 V-shape。slope 吻合
- 红外 IR：本来如果有 dust，那么dust 的红外太强了，超过 MIRI 的观测；解释为 dust 的分布 $T\sim r^{-p}, \rho \sim r^{-\gamma}$ ，把 dust emission 按质量平均一下，就能把红外往更长波挪一挪
  - dust 吸收 optical-UV，辐射 IR
  - dust 的 column density 应该要和 optical-UV 所拟合得到的 $A_V$ 所要求的一致才行。
  - 温度的profile 用 $T\propto r^{-p}$,  目前取 $p=1/2$ ，来自于 $T^4 \propto L/r^2$。但这里没有考虑 shielding 的影响。对于密度分布很集中 ($\gamma \gtrsim 1$) 的情况，column density 主要由内区贡献，那么这个就应当考虑进来了…… 应该会使得 T(r) 更陡峭，也就是 p 更大。
  - 目前 $R_{\rm in},R_{\rm out}$ 没有仔细处理。取 $R_{\rm in} \sim 0.2 {\rm\ pc}$ , $R_{\rm out}$ 则根据 $T_{\rm out} \sim 200 {\rm K}$ 来取。同时不应该大于观测的 PSF。

参数空间的限制

- 柱密度

- ALMA non-detection

- PSF ~ rou < 600-800 pc (这是 F444W, 针对恒星的，而非 dust)

  

### 算法要点

- LHS(r) == RHS(T)
- RHS 的计算每次要做一个积分，LHS 只需要单个数字的计算
- 加速：先算一遍 RHS(T) 对 T 的表，存下来。然后对每个 r，根据 LHS(r) 对应查表确定区间
  - 如果表足够细，比如精确到 1K，那么插值就足够了
  - 如果表比较粗，可以再用 brentq 求根，找到更精细的 T



- [x] 区分 ext, abs
- [ ] 把 RHS 用 Orion * Temple 做
  - [ ] 把 tau 改成用 data

- [ ] B87 consistant
- [ ] 整理代码
  - [ ] 类的区分、继承
  - [ ] 单位处理


几种模型：

- power-law T profile
- B87：考虑遮挡。使用 $Q_\nu \propto \nu^\beta$
- Orion IR 出射
- Orion UV 入射 + IR 出射
