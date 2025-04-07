# LRD Spectrum

## 一开始的想法（2024-06）

- 观测：X-ray 有 Chandra non-detection, Optical-UV 有 V-shape, IR 有 MIRI 的一个波段的数据，非常长波的有 ALMA (non-detecion?)
- Slim disk （我算的部分）+ Dust attenuation $A_V$ (Orion 模型，而非更常用的 xxx)，可以解释 V-shape。slope 吻合
- 红外 IR：本来如果有 dust，那么dust 的红外太强了，超过 MIRI 的观测；解释为 dust 的分布 $T\sim r^{-p}, \rho \sim r^{-\gamma}$ ，把 dust emission 按质量平均一下，就能把红外往更长波挪一挪
  - dust 吸收 optical-UV，辐射 IR
  - dust 的 column density 应该要和 optical-UV 所拟合得到的 $A_V$ 所要求的一致才行。
  - 温度的profile 用 $T\propto r^{-p}$,  目前取 $p=1/2$ ，来自于 $T^4 \propto L/r^2$。但这里没有考虑 shielding 的影响。对于密度分布很集中 ($\gamma \gtrsim 1$) 的情况，column density 主要由内区贡献，那么这个就应当考虑进来了…… 应该会使得 T(r) 更陡峭，也就是 p 更大。
  - 目前 $R_{\rm in},R_{\rm out}$ 没有仔细处理。取 $R_{\rm in} \sim 0.2 {\rm\ pc}$ , $R_{\rm out}$ 则根据 $T_{\rm out} \sim 200 {\rm K}$ 来取。同时不应该大于观测的 PSF。

### 参数空间的限制

- 柱密度

- ALMA non-detection

- PSF ~ rou < 600-800 pc (这是 F444W, 针对恒星的，而非 dust)


## A_V constraint Project

## 算法描述

### 各个 Model

几种模型：

- power-law T profile
- B87：考虑遮挡。使用 $Q_\nu \propto \nu^\beta$
- Orion IR 出射
- Orion UV 入射 + IR 出射

#### 所有 Model 的共性

##### 假设 & 近似

TODO

##### 计算步骤

- 给定 gas density profile $n(r)$

- 计算 dust 的温度 profile $T_\mathrm{dust}(r)$

- 计算 IR 出射光谱：

$$
L_{\nu, \mathrm{dust}} = \Omega \int_{r_\mathrm{in}}^{r_\mathrm{out}}  \sigma_\nu^\mathrm{abs} \cdot B_\nu(T_\mathrm{dust}(r)) \cdot n(r) \cdot 4 \pi r^2 dr
$$



#### power-law T profile

直接假设 T profile 是个 power-law，忽略了遮挡的影响。但这里的 p 是自由参数。
$$
T_\mathrm{dust}(r) = T_0 \left( \frac{r}{r_0} \right)^{-p}
$$

#### B87 model
Barvainis 1987 文章中的模型。我做了一些拓展，让 $\beta$ 不限于 1.6。

基本假设：
- power-law $n(r)$
- 吸收截面 $\sigma_\nu$ 在 UV 是常数且为几何截面（$Q_\nu \approx 1$）, 在 IR 近似为幂律：$\sigma_\nu^\mathrm{abs} \propto \nu^{-\beta}$
  - 从而 UV 波段的吸收可以直接由 $L_\mathrm{UV}$ 给出，不需要知道具体光谱形状
  - IR 波段的辐射也可以解析积出来
- $T_\mathrm{dust}(r)$ 可以解析表达，从而 $r_\mathrm{in}$ 也可以。

#### Semi Orion model

IR 出射按照具体的 opacity 计算，而 UV 入射仍是直接给定 L_UV

#### Orion model (李政融 et al. 2025 paper)

依据是任意 $r$ 处的能量平衡方程
$$
L_{\mathrm{abs}}(r) = L_{\mathrm{rad}}(r)
$$

$$
\int F_{\nu, \mathrm{incident}} \; \sigma_\nu^{\rm abs} \mathrm{d}\nu = 4 \pi \int B_\nu[T_\mathrm{dust}(r)] \; \sigma_\nu^{\rm abs} \mathrm{d}\nu
$$

其中，
$$
F_{\nu, \mathrm{incident}} = L_{\nu, \mathrm{AGN}} \cdot \frac{\mathrm{e}^{-\tau_\nu (r)}}{4 \pi r^2} 
$$

$$
\tau_\nu (r) = \int_{r_\mathrm{in}}^{r} n(r') \cdot \sigma_\nu^{\rm ext} dr'
$$


### 能量平衡方程的加速算法：打表插值

- LHS(r) == RHS(T)
- RHS 的计算每次要做一个积分，LHS 只需要单个数字的计算
- 加速：先算一遍 RHS(T) 对 T 的表，存下来。然后对每个 r，根据 LHS(r) 对应查表确定区间
  - 如果表足够细，比如精确到 1K，那么插值就足够了
  - 如果表比较粗，可以再用 brentq 求根，找到更精细的 T
