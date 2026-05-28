# 关于本仓库

本仓库目标是构建 Rust 下，其中一种 DFT 格点积分 API 原型。

- API 设计原型、理念与简易实现 (简单矩阵乘法)。
- 可以实现 vxc, fxc, kxc 的计算。当前实现不保存所有格点，对内存压力较小。
- 矩阵乘法后端使用 Faer，没有 BLAS 依赖，编译方便。
- 简要性能测试下，在 C22H14 TPSS0/def2-TZVP (nao = 766) 分子，表现尚可 (单步 DFT 格点计算相对于 PyPI 发行的 PySCF 快 20% 左右)。

本仓库的代码尽管目前没有进一步抽象为 Trait 接口；但我个人希望未来的 Trait 能够与本文档中描述的 API 设计概念相匹配。

# DFT 格点积分 API 设计概念

该文档将讨论 DFT 格点积分 API 设计中的一些核心概念。希望这些概念将有助于未来 DFT 格点积分的可扩展性、性能优化、API 调用友好性等方面的设计。

该文档也希望以相对比较一致的文本，厘清 DFT 程序所需要的名词、概念、公式。本文的说法未必是最正统的，但这确实是我在设计 API 时一贯的思路。

## 1. 通论：自洽场关键 API 设计

### 1.1 讨论前提：能量可分及其与密度矩阵的关系

计算任务的根本变量是基函数；这些基函数可能是原子轨道、平面波、实空间格点，不过原子轨道最常见。在该项目中，习惯称基函数为 AO (atomic orbital)，但这并不意味着我们只能处理原子轨道。

**能量是可以分解的**。譬如，对于 wB97X-V 泛函，其能量可以拆分为下述部分：

$$
E[\mathbf{D}] = E_\text{nuc-repl} + E_\text{kin} + E_\text{nuc} + E_\text{J} + E_\text{K} + E_\text{srK} + E_\text{xc} + E_\text{VV10}
$$

其中有一些要点是，在假定原子核不作为变量的前提下，**能量及其分项是密度矩阵 $\mathbf{D}$ 的函数**。这句话本身很简单，但隐含了下述推论或不太平常的反例：

- 密度矩阵定义为基函数系数矩阵 $C_{\mu i}$ 的函数：

    $$
    D_{\mu\nu} = \sum_i C_{\mu i} C_{\nu i}
    $$

    因此，能量及其分项也可以是基函数系数矩阵 $\mathbf{C}$ 的函数。

- 能量不能仅是系数矩阵 $\mathbf{C}$ 的函数。它必须要能写为密度矩阵 $\mathbf{D}$ 的显函数。这里的反例是各种 orbital-optimized post-SCF 方法。

- 能量与密度的关系必须严格满足 Hartree-Fock-Roothaan 方程：

    $$
    \mathbf{V}[\mathbf{D}] \mathbf{C} = \mathbf{S} \mathbf{C} \mathbf{\epsilon}
    $$
    
    其中，Fock 矩阵 $\mathbf{V}$ 是能量对密度矩阵 $\mathbf{D}$ 的梯度：

    $$
    \mathbf{V}[\mathbf{D}] = \frac{\partial E[\mathbf{D}]}{\partial \mathbf{D}}
    $$

    这里的反例是 constraint DFT (cDFT) 或 density-corrected DFT (dcDFT) 等方法。cDFT 包含约束项；dcDFT 的非自洽特性使得其 Fock 矩阵不再是能量的梯度。尽管我们这里写程序时仍然可以利用当前的 API，但在程序设计理念上，为了简化讨论，我们暂时不考虑这些方法。

- 原子结构不在考虑范围；我们只考虑电子结构 (以基函数表示的电子密度)。因此，对于原子核排斥能 $E_\text{nuc-repl}$、以及类似于 DFT-D3 的方法，我们认为它的能量项是常数。

同时，请留意，**本文的 Fock 矩阵以 $\mathbf{V} [\mathbf{D}]$ 表示，而非通常的 $\mathbf{F} [\mathbf{D}]$**。在下一小节将明确本文档记号。

### 1.2 核心问题：能量对密度矩阵导数

涉及到自洽场方法的计算问题，很多核心的技术问题，都涉及到如何计算能量对密度矩阵的导数。

我们这里需要作下述定义：

| 导数阶 | 变量 | 变量全写 | 惯用变量记号 |
|--|--|--|--|
| 1 | $\mathbf{V}$ | $\mathbf{V} [\mathbf{D}]$ | `fock`, `v`, `veff` |
| 2 | $\mathbf{F}$ | $\mathbf{F} [\mathbf{D}, \mathbf{R}]$ | `resp`, `f` |
| 3 | $\mathbf{K}$ | $\mathbf{K} [\mathbf{D}, \mathbf{R}^1, \mathbf{R}^2]$ | `k` |

- 本文档的程序惯用变量记号是 `v`, `f`, `k`，分别对应上述的 $\mathbf{V}$, $\mathbf{F}$, $\mathbf{K}$。

- 一阶导数定义

    $$
    V_{\mu \nu} = \frac{\partial E}{\partial D_{\mu \nu}}
    $$

    一阶导数在其他文献中，通常记为 $F_{\mu \nu}$。我们这里为了区分一阶、二阶导数，特意使用了不同的记号。

- 二阶导数定义

    $$
    F_{\mu \nu} = \sum_{\kappa \lambda} \frac{\partial^2 E}{\partial D_{\mu \nu} \partial R_{\kappa \lambda}} R_{\kappa \lambda}
    $$

    其中 $R_{\kappa \lambda}$ 是扰动密度矩阵；它在 TDDFT 中 (Casida 方程) 经常是激发态密度矩阵，在梯度问题中 (CP-HF/KS 方程) 是导数密度矩阵。

    在其他文献中，二阶导数通常记为 $\sum_{\kappa \lambda} A_{\mu \nu, \kappa \lambda} R_{\kappa \lambda}$，或 $G_{\mu \nu} [\mathbf{R}]$。

- 三阶导数定义

    $$
    K_{\mu \nu} = \sum_{\kappa \lambda} \sum_{\xi \zeta} \frac{\partial^3 E}{\partial D_{\mu \nu} \partial R^1_{\kappa \lambda} \partial R^2_{\xi \zeta}} R^1_{\kappa \lambda} R^2_{\xi \zeta}
    $$

    其中 $R^1_{\kappa \lambda}$ 和 $R^2_{\xi \zeta}$ 是两个扰动密度矩阵。

上面的所有定义，对所有可用于自洽场计算的能量分项都适用；且可以线性加和。譬如说，对于 wB97X-V 泛函，其 Fock 矩阵可以依葫芦画瓢写为：

$$
\mathbf{V} [\mathbf{D}] = \mathbf{V}_\text{nuc-repl} + \mathbf{V}_\text{kin} + \mathbf{V}_\text{nuc} + \mathbf{V}_\text{J} + \mathbf{V}_\text{K} + \mathbf{V}_\text{srK} + \mathbf{V}_\text{xc} + \mathbf{V}_\text{VV10}
$$

因此，对于任何一个能量分项而言，它都可以定义下述的程序接口：

```python
class EnergyComponent:
    def energy(self, dm0: np.ndarray) -> float:
    def get_fock(self, dm0: np.ndarray) -> np.ndarray:
    def get_2nd_resp(self, dm0: np.ndarray, dm1: np.ndarray) -> np.ndarray:
    def get_3rd_resp(self, dm0: np.ndarray, dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
```

### 1.3 技术考虑：性能与可扩展性

上述的接口设计，相信是可以实现所有重要的计算化学需求。但实际使用中，我们还需要作如下的考量与适配：

- **闭壳层与开壳层的密度矩阵维度有所差异**。闭壳层维度是 `[nao, nao]`，而开壳层维度是 `[nao, nao, 2]` (col-major)。
- 密度矩阵，特别是扰动密度矩阵，**经常是多个**而非单个。因此，对于闭壳层，传入的 `dm1` 与 `dm2` 一般要允许是 3-dim 张量、或列表的 2-dim 矩阵。传入的 `dm0` 大多数情况下是单个矩阵。
- 密度矩阵通常是低秩的。它是由占据轨道系数构成的，而占据数 $n_\text{occ}$ 通常远小于基组数 $n_\text{AO}$。即使是 CP-KS 方程求解中所需要的扰动密度 (不同于自洽场密度)，它从构造上 (或者用 SVD 等方式分解上) 也可以分解为两个 $n_\text{AO} \times n_\text{occ}$ 的矩阵的乘积。因此**应该尽量利用密度矩阵的低秩结构**，以降低计算量。
    - 另一方面，对于特别大的分子，密度矩阵本身可能有足够的稀疏程度。密度矩阵零值的稀疏性与低秩结构不能 (至少难以) 同时利用；某些情况下，密度矩阵本身的稀疏性也可以利用。但我们关注的原子体系如果不超过 100 个原子，那么密度矩阵的低秩结构通常更重要。
    - 从 API 设计的角度而言，低秩性质的利用，等同于在计算函数中传入占据轨道系数矩阵：
        
        ```python
        def get_fock_by_occ(self, occ_coeff: np.ndarray) -> np.ndarray:
        ```

        具体的接口形式可能会有一些变化，但核心思想是，**API 设计应该允许直接传入占据轨道系数矩阵**，以利用密度矩阵的低秩结构。我们会在后面看到，在 DFT 格点积分中，我们是如何具体地利用这一点。
