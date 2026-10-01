# pyml 向量化内核：性能验证笔记

**目的**：pyml 的 C++ 部分（`pyml/cpp/`）已有自研 SIMD 内核，但此前没有任何性能证据。本文记录把它接到一个真实的、纯标量的系统发育软件（PAML/mcmctree）上做 A/B 测量的过程、结果，以及过程中发现的**三个会让向量化得不偿失的坑**。

**结论先行**：内核本身快 3.0–4.0×，端到端快 1.4–2.3×。但 `pyml/cpp/likelihood.hpp` 现在用的向量化写法是**错的**——实测比标量还慢，详见第 3 节。必须改成寄存器分块的写法。

---

## 1. 为什么选 mcmctree 做对照

- 它是 30 年历史的权威实现，似然计算是纯标量 C；
- 这个 abacus-gene 分支里**完全没有 BEAGLE、没有任何 SIMD**（已用 grep 确认），是个干净的基线；
- 它的 MCMC 能直接跑起来，不需要额外包装。

对照设计成三臂，同一个数据、同一个 seed、同一个 ctl（只差一行开关）：

| 臂 | 来源 | 编译选项 | 隔离出什么 |
|---|---|---|---|
| arm1 | 未打补丁的 PAML | `-O3`（PAML 自己的默认） | 基线 |
| arm2 | 未打补丁的 PAML | `-O3 -march=native` | **GCC 自动向量化的收益** |
| arm2b | 打了补丁，`vecLnL = 0` | `-O3 -march=native` | 与 arm3 同一个二进制，走标量路径 |
| arm3 | 打了补丁，`vecLnL = 1` | `-O3 -march=native` | 手写 SIMD |

arm2b 和 arm3 **是同一个二进制**，只差 ctl 里的 `vecLnL`，所以两者的差异只可能来自内核本身。

---

## 2. 关键约束：布局与向量化轴

mcmctree 的 `conP` 是 **pattern-major**：`conP[h*ncode + j]`。内层循环是

```c
for (j) { t = 0; for (k) t += PMat[j*n+k] * child[h*n+k]; parent[h*n+j] *= t; }
```

一个长度 `ncode` 的点积，归约在标量 `t` 上。**这里没有可向量化的轴**——pattern 轴 (`h`) 是唯一的长轴，但归约把它挡住了。

把节点块改成 **state-major** `vecConP[j*npatt + h]`，pattern 轴变连续，就有轴可用了。这正是 `likelihood.py` 注释里早就写明的决策，也是 `likelihood.hpp` 在用的布局。

补丁做法：新增 `ConditionalPNodeVec` + 独立的 `vecConPin` 双缓冲，**完全不动 PAML 原有的 `conP` 布局和算术**，由 ctl 的 `vecLnL` 开关切换，默认 0 时行为逐位不变。

---

## 3. 三个坑（这是本次最有价值的部分）

### 坑 1：AXPY 形式是错的，实测比标量慢 1.5×

最自然的写法是把 `sum_k P[j][k]*child[k][h]` 展开成 `ncode²` 次 AXPY，每个 (j,k) 一次：

```c
for (j) for (k) pyml_vec_muladd_f64(acc + j*npatt, child + k*npatt, P[j*n+k], npatt);
```

**`pyml/cpp/likelihood.hpp` 的 `compute_node` 用的正是这个写法。**

每个 AXPY 都要把整个 child 块读一遍、整个累加器写一遍，所以访存是 `ncode` 倍。ncode=4 时，SIMD 省下的指令数远不够抵消放大的访存。实测（独立微基准，1760 patterns）：

| 写法 | 相对标量 |
|---|---|
| AXPY（`pyml_vec_muladd`） | **×0.65**（慢 1.5 倍） |

改用寄存器分块：pattern 轴走一趟，每个 tile 加载一次 child，k 的归约在寄存器里做完，parent tile 只写一次——**访存和标量循环相同**。

### 坑 2：state 数必须是编译期常量，否则寄存器 tile 会掉到栈上

第一版分块 kernel 的 `nstates` 是运行时参数：

```c
double cv[64][4];            // 下标 k 是运行时变量
for (k = 0; k < nstates; ++k) cv[k] = LOAD(child + k*npatt + h);
for (...) acc = FMADD(cv[k], pv[k], acc);
```

编译器无法证明运行时下标落在范围内，只能把 `cv[]`/`pv[]` 放在**栈上**，于是每次 FMA 的操作数都要从栈读回。实测 ×0.65，和 AXPY 一样慢。

按 `nstates` 特化成模板参数后（`4`、`2`、`20` 三个特化 + 通用回退），`cv[]` 全部进寄存器：

| 写法 | 相对标量 |
|---|---|
| 运行时 nstates | ×0.65 |
| **编译期特化** | **×1.12 → ×1.50** |

### 坑 3：把 multiply 融合进 kernel，省掉一整趟父块扫描

节点有 ≥2 个子节点时，第二个及以后的贡献需要与已有的 parent 相乘。第一版写成「算到 scratch → 再乘进 parent」，那是**额外一整趟访存**，对双子节点的节点等于一半的工作量：

```c
pyml_vec_pmat_mul_f64(scratch, child, P, n, npatt);
pyml_vec_mul_f64(parent, parent, scratch, block);   // 多一整趟
```

融合成 `parent *= sum_k(...)` 后（`pyml_vec_pmat_mul_acc_f64`）：

| 写法 | 相对标量 |
|---|---|
| 分两趟 | ×1.12 → ×1.50 |
| **融合 accumulate** | **×1.6 → ×2.3** |

### 附：`-march=native` 一点用都没有

arm1 与 arm2 在所有规模上**都在噪声内相同**（6 taxa: 0.113 vs 0.119；16 taxa: 0.932 vs 0.963）——甚至略慢。GCC 无法自动向量化这个循环，结构上也不可能：归约挡住了 pattern 轴，而 state 轴只有 4 长。**手写 SIMD 是这里唯一有效的办法。**

（注：`-fno-tree-vectorize` 的对照实验也确认了这一点，标量耗时几乎不变 0.113 → 0.110 ms。）

---

## 4. 正确性验证

mcmc.txt 里 lnL 只印 3 位小数（`-9710.640`），靠比对轨迹最多只能分辨 1e-3，**不够**。所以加了一个自校验模式 `vecLnL = 2`：同一组参数下两条路径都算，逐 pattern 比对根似然向量的**全精度**差值。

| 数据 | patterns | 比对次数 | 最大绝对差 | 最大相对差 |
|---|---|---|---|---|
| 6 taxa × 1000，干净 | 592 | 392 | 8.3e-17 | **7.6e-16** |
| 6 taxa × 1000，12% 简并码 (R/Y/N) | 862 | 352 | 8.3e-17 | **8.3e-16** |
| 16 taxa × 2000，干净 | 1760 | 852 | 1.7e-16 | **1.8e-15** |

机器精度量级，且**每个 pattern 都比对**，不只是聚合值。简并码那条路径（指示块一格多个 1）是最容易出错的地方，单独做了数据（把 738 个碱基替换成 R/Y/N）来覆盖。

另外，`pyml/cpp/test_vectorize_c.c` 覆盖了全部 C ABI 内核，含 `pmat_mul` 的两种模式、`nstates ∈ {2,4}`、以及 **npatt 余数 0..3 的每个取值**（专门覆盖补零尾部）。

---

## 5. 结果

### 端到端（mcmctree 实际墙钟，`print=0`，5 次取最优）

| 数据 | arm1 | arm2 | arm2b | arm3 | 似然加速 | 端到端加速 |
|---|---|---|---|---|---|---|
| 6 taxa × 1000 | 0.113s | 0.119s | 0.122s | 0.054s | ×3.04 | **×2.08** |
| 16 taxa × 2000 | 0.932s | 0.963s | 0.970s | 0.427s | ×3.96 | **×2.27** |
| 32 taxa × 5000 | 5.377s | 5.483s | 5.483s | 2.829s | ×3.20 | **×1.88** |
| 64 taxa × 5000 | 7.647s | 7.799s | 7.892s | 5.855s | ×3.38 | **×1.39** |

「似然加速」是 mcmctree 内部对 `com.plfun(NULL,-1)` 单独计时的结果（同一条代码路径，两臂差异只可能来自内核）。

端到端加速随规模下降，是因为 mcmctree 的非似然开销（MCMC 提议、先验、枝长计算）增长得比似然快。64 taxa 时似然只占标量总时间的 ~53%。

### 独立 kernel 微基准（把 mcmctree 的全部开销剥掉）

| taxa | scalar | vec | 加速 |
|---|---|---|---|
| 4 | 0.030 ms | 0.025 ms | ×1.20 |
| 16 | 0.113 ms | 0.052 ms | ×2.14 |
| 64 | 0.540 ms | 0.240 ms | ×2.25 |
| 128 | 0.975 ms | 0.540 ms | ×1.81 |

注意这里比 mcmctree 里测出的 ×3–4 要低。原因是微基准的标量循环比 PAML 的 `ConditionalPNode` 更简单、对编译器更友好（PAML 那版有指针别名和分支）。**所以 ×3–4 是更贴近真实的值。**

---

## 6. 一个只有集成时才暴露的坑：双份 memcpy

第一次跑 64 taxa 时，arm3 比 arm2b **慢 3 倍**，而似然本身快 3.3 倍。查了很久才找到：

`UseLocus(copyconP=1)` 每个 MCMC 轮次都会 `memcpy` 一份完整的内部节点 conP 缓冲。我为了让影子缓冲区参与 PAML 的双缓冲，**照着又加了一次 memcpy**。64 taxa 下那是每次 ~10MB，10906 轮 ≈ **108GB 的额外内存拷贝**。

关键认识：**向量模式下 `conP` 根本不会被读**，那次拷贝纯属浪费。改成「只拷贝当前在跑的那条路径的缓冲」后：

| 64 taxa | 端到端 |
|---|---|
| 修之前 | ×0.68（慢） |
| 修之后 | **×1.39** |

`vecLnL = 2` 两条路径都跑，所以两份都要拷——这也是为什么修的时候必须按「哪条路径在用」判断，而不能简单按开关判断（第一版改错了，verify 模式当场崩掉，正好被自校验模式抓到）。

---

## 7. 关于「用 0 填满剩余位」

已按要求把 `PmatMulFixed*` 的标量尾部换成零填充：最后 `npatt%W` 个 pattern 拷进零初始化的 tile，走同一份向量代码，再拷回来。

**性能上没有改善，也不可能有。** 尾部最多 `W-1 = 3` 个 pattern，占 1760 的 0.17%，远低于噪声。用「余数 ≠ 0」和「余数 = 0」的 npatt 直接对比（同一二进制，600 次迭代）：

```
npatt  余数   vec ms/iter
1756   0     0.058
1757   1     0.073
1758   2     0.070
1759   3     0.068
1760   0     0.075      <- 余数 0 也没更快
1764   0     0.057
```

耗时与余数完全无关，全是噪声。

**但这个改动仍然值得保留**，理由不是速度：原来的标量尾部是**第二份算术实现**，在 FMA 下它与向量主体的舍入行为可能不同，得靠人工保持同步。零填充之后只剩一份实现，头部和尾部必然舍入一致。

---

## 8. 修复 `pyml/cpp/likelihood.hpp`

把第 3 节的结论应用回去。原来的 `compute_node` 有四处问题：

1. **AXPY 写法**（坑 1）——`left_sum`/`right_sum` 每个 (parent,child) 状态对跑一次 `MultiplyAccumulate`；
2. **每个内部节点每轮 248 次堆分配**——`PMatrix` 每次调用在 `transitionMatrix` 里分配两个 `Matrix`，再拷进一个 `std::vector` 返回（2 子节点 × 3 次 = 6），加上 `left_sum(sites)`、`right_sum(sites)` 两个临时向量；
3. **`node.likelihood.assign(...)` 每轮填充整块**，而随后的循环会覆写它；
4. **根归约是逐 site 的 4 元点积**，归约在标量累加器里，同样无法向量化。

**改法**：

- `SubstMatrix::PMatrix(t)` 返回 `std::vector` 的版本**删掉**，换成 `PMatrix(t, out, scratch)`，两个缓冲区都归调用方；`evd.hpp` 加了对应的非分配重载 `transitionMatrix(es, t, rate, out, scratch)`（旧的分配版保留为一行包装）。
- `LikelihoodCalculator` 持有工作区（`pmat_`、`pmat_scratch_`、`root_lik_`），构造时定尺寸，此后**每次评估零分配**。
- `compute_node` 改用 `PmatMulF64` / `PmatMulAccF64`：第一个子节点直接覆写父块，第二个融合乘入，**不需要任何临时累加器**。
- `resize` 替代 `assign`（kernel 会写满每个元素）。
- 根归约改成沿 site 轴的 AXPY，与 mcmctree 补丁里的 `VecRootLik()` 同一形式。

**分解测量**（32 taxa × 5000 sites，40 次；中间变体 = 新 kernel + 旧的每调用分配）：

| 版本 | 耗时 | 分配/次 | 归因 |
|---|---|---|---|
| old（AXPY + 每调用分配） | 2.070 ms | 248 | — |
| mid（PmatMul + 每调用分配） | 0.458 ms | 62 | **kernel 贡献 ×4.52** |
| new（PmatMul + 零分配） | 0.440 ms | 0 | 零分配再加 ×1.04 |

**结论：几乎全部收益来自 kernel 写法本身（×4.5），去掉分配只再加 4%。** 这和 mcmctree 侧的观察一致——写法是决定性的，分配是次要的。

**规模曲线**（只改 `likelihood.hpp`，其余完全相同）：

| case | before | after | 加速 |
|---|---|---|---|
| 8 taxa × 1000 | 0.038 ms | 0.013 ms | ×2.92 |
| 16 taxa × 2000 | 0.231 ms | 0.073 ms | ×3.16 |
| 32 taxa × 5000 | 1.278 ms | 0.445 ms | ×2.87 |
| 64 taxa × 5000 | 2.895 ms | 0.987 ms | ×2.93 |
| 128 taxa × 2000 | 2.146 ms | 0.826 ms | ×2.60 |

**正确性**：`bench_likelihood.cpp` 内置了一个独立锚点——4 taxa JC69 的似然可以手算（对两个内部节点状态 + 根状态穷举求和），得到 `-25.786608878551`。改动前后都精确复现该值（相对误差 1.4e-16），`float` 实例化给出 `-25.786613`。这个值同时也是本次会话开头用穷举法和 mcmctree C++ 路径交叉验证过的。

---

## 9. 交付物

**pyml 侧（新增为主，不动既有语义）**

| 文件 | 内容 |
|---|---|
| `pyml/cpp/vectorize.hpp` | 新增 `MultiplyInPlace`/`ScaleInPlace`、FMA 路径、`PmatMulF64/F32`、`PmatMulAccF64/F32`、宽度/ISA 查询声明 |
| `pyml/cpp/vectorize.cpp` | FMA 各 Ops 分支；两个原位算子；**分块 + state 数特化 + accumulate 融合的 PmatMul 内核**（含零填充尾部） |
| `pyml/cpp/vectorize_c.h` | **新增** — `extern "C"` 的 C ABI，无任何 STL 类型 |
| `pyml/cpp/vectorize_c.cpp` | **新增** — C ABI 实现 |
| `pyml/cpp/test_vectorize_c.c` | **新增** — 全部内核对照朴素实现的自检 |
| `pyml/cpp/likelihood.hpp` | **重写 `compute_node`**：融合 kernel、零分配工作区、根归约改 AXPY；`PMatrix` 改为非分配签名 |
| `pyml/cpp/evd.hpp` | 新增非分配重载 `transitionMatrix(es, t, rate, out, scratch)` |
| `pyml/cpp/bench_likelihood.cpp` | **新增** — 正确性锚点 + 计时 + 可选的分配计数 |

**第二部分（见下）另加：**

| 文件 | 内容 |
|---|---|
| `pyml/cpp/core/*.hpp` | **新增** — 无根树、状态、移动、先验、驱动（约 1300 行） |
| `pyml/cpp/bindings.cpp` | **新增** — `pyml._core`，pybind11 模块 |
| `pyml/cpp/build_core.py` | **新增** — 固定工具链的构建脚本 |
| `pyml/cpp/test_priors.cpp` | **新增** — 先验复现检验（Hastings 比的证伪手段） |
| `pyml/cpp/test_bindings.py` | **新增** — Python 侧绑定与自定义移动 |
| `pyml/cpp/run_trace.py` | **新增** — 长链运行，产出 Tracer 可读的 `.p` |

**PAML 侧（补丁，默认关闭）**

| 文件 | 内容 |
|---|---|
| `src/mcmctree.c` | `vecLnL` 开关 + `ConditionalPNodeVec` + 影子双缓冲 + 叶子指示块预计算 + `vecLnL=2` 自校验模式 + 似然分段计时 |
| `src/treesub.c` | `lfun()` 里的分支，用 `#ifdef MCMCTREE` 隔离（baseml/codeml 不受影响，已验证仍可编译） |

**基准环境**（`E:\Codes\PYML\paml_bench\`）

`run_bench.py`（四臂驱动）、`prep_data.py`（alisim 造数据）、`kernel_bench.c`（独立内核微基准）、`bin/`、`data/`、`out/`。

---

## 10. 复现

```bash
# 构建 pyml 内核
g++ -std=c++20 -O3 -march=native -c pyml/cpp/vectorize.cpp
g++ -std=c++20 -O3 -march=native -c pyml/cpp/vectorize_c.cpp

# 内核自检
gcc -O2 -c pyml/cpp/test_vectorize_c.c
g++ -O2 -mavx2 -mfma -o t test_vectorize_c.o vectorize.o vectorize_c.o -lm && ./t

# likelihood.hpp 正确性锚点 + 计时（-DPYML_COUNT_ALLOCS 附带分配计数）
g++ -std=c++20 -O3 -march=native -DPYML_COUNT_ALLOCS \
    -o bench pyml/cpp/bench_likelihood.cpp pyml/cpp/vectorize.cpp -lm
./bench 4 4 1          # 5-taxon 锚点，比对 -25.786608878551
./bench 32 5000 50     # 规模计时

# PAML 侧：把补丁打到一份干净的 PAML 检出上（abacus-gene/paml）
git clone https://github.com/abacus-gene/paml.git
cd paml && git apply <repo>/patches/paml-mcmctree-veclnl.patch

# PAML 三臂
gcc  -O3 -w -c mcmctree.c tools.c                       # arm1，需在未打补丁的源码上
gcc  -O3 -march=native -w -c mcmctree.c tools.c         # arm2
gcc  -O3 -march=native -w -I<pyml>/cpp -c mcmctree.c    # arm2b/arm3 共用
g++  -O3 -march=native -o mcmctree_vec mcmctree.o tools.o vectorize.o vectorize_c.o -lm

# 正确性（vecLnL = 2）
python paml_bench/run_bench.py --tag v --arms verify --seqfile sim16x2000.phy \
       --treefile sim16x2000.trees --burnin 20 --nsample 30

# 性能
python paml_bench/run_bench.py --seqfile sim16x2000.phy --treefile sim16x2000.trees \
       --arms arm1,arm2,arm2b,arm3 --repeat 5 --burnin 50 --nsample 500 --print 0
```

---

## 11. 建议

1. ~~修 `likelihood.hpp` 的 `compute_node`~~ —— 已完成，见第 8 节。
2. ~~`SubstMatrix::PMatrix` 按值返回 `std::vector`~~ —— 已改为非分配签名。
3. **`transpose2` / 叶子和根之间的中间节点仍会走完整 matmul。** 对只有 4 个 state 的问题，`PmatMulFixedDouble<4>` 的 16 次 FMA 里有 12 次乘的是 0/1 指示块（叶子）。要再进一步，得按 state 分组 pattern——代价是每个叶子一套不同的 pattern 序，会破坏共享的 site 轴。收益未知，先不动。
4. **离散 Γ 的 site-class 路径仍未验证**（`fx_r`/`conPSiteClass`）。本次明确限定 JC69 单速率。
5. pyml 侧还没有任何绑定。做绑定时注意 `LikelihoodCalculator::LogLikPruner()` 每次调用返回单个标量——若要暴露梯度或逐 site 似然，需要另行设计接口。

## 12. 未决（第一部分）

- 本轮所有测量都跑在一台会降频的笔记本上。除了 kernel 微基准外，端到端数字建议在稳定机器上复测。
- `vecLnL=1` 下 `nodes[].conP` 仍被分配（约 20MB @64taxa）却从不读取。可以省掉，但会让 `GetMem` 的 offset 数学与标量路径分叉，暂未动。
- `mcmctree.c` 的 `ConditionalPNodeVec` 与 `likelihood.hpp` 的 `compute_node` 现在是**两份独立实现**了同一件事。逻辑上应该收敛成一份（mcmctree 侧调用 pyml 的函数），但本次刻意没做——基准的价值恰恰在于两条路径互相独立、可以互相对拍。真要合并，应在这轮验证结论固化之后。

---

# 第二部分：MCMC 核心

第一部分把似然内核做快、验证对了。第二部分把它接上 MCMC，导出成 Python 可配置的模块，并和 MrBayes 正面比一次。**下一阶段的工作从这里继续。**

## 13. `pyml._core`：C++ 的 MCMC 与 pybind11 导出

### 可行性：MinGW 的模块能进 MSVC 的 Python

这是整件事的前提，实测通过：

```
$ /d/PyMOL/python.exe -c "import _core; print(_core.Multiplier)"
```

工具链组合只有一条路能走通，其余组合失败的方式都不指向病因，所以记下来：

| 部件 | 位置 | 备注 |
|---|---|---|
| 编译器 | `/d/msys64/mingw64/bin/g++` 15.2.0 | **必须**是 MSYS2 的 |
| pybind11 | `/d/msys64/mingw64/include/pybind11` 3.0.2 | 支持 Python ≥3.8 |
| Python 头 | `/d/PyMOL/include/Python.h` | |
| 导入库 | `/d/PyMOL/libs/python310.lib` | |
| 运行时 | `-static-libgcc -static-libstdc++` | 否则要发 libstdc++-6.dll |

三个坑：

1. **`which g++` 拿到的是 winlibs 那个**（`D:\Program Files\mingw64`，`mcf` 线程模型），不是 MSYS2 的。用错就链接失败或加载失败。
2. **必须把 MSYS2 的 bin 加到 PATH**。用绝对路径调用编译器不够——它要靠 PATH 找自己的 DLL，否则**静默退出、没有任何诊断信息**。
3. `_core*.pyd` 必须带正确的 `EXT_SUFFIX`（`.cp310-win_amd64.pyd`），否则导入时报找不到模块。

`pyml/cpp/build_core.py` 把上述全部固化，并把 GCC 临时目录指向 `PYML_BUILD_TMP`（见第 16 节）。

### 结构

| 文件 | 内容 |
|---|---|
| `core/tree.hpp` | 无根树（根在三叉内部节点）、Newick 读写、NNI 重连原语、**BranchLabel** 规范编号 |
| `core/state.hpp` | 状态 + 快照/回滚 + Q 重建（事务性 `TrySetFreqs`） |
| `core/moves.hpp` | `Multiplier`、`TLMultiplier`、`NNI`、`Dirichlet(Pi)`、`Slider(Pi)` |
| `core/priors.hpp` | `Dirichlet(Pi)`、`GammaDir(V)`、`Flat(Tau)` |
| `core/mcmc.hpp` | 驱动 + 自动调参 + trace |
| `bindings.cpp` | `pyml._core`，Move/Prior 带 trampoline |

`likelihood.hpp` 的 `Node` 从两个子节点推广到最多三个（无根树的根是三叉的），`connect(parent, left, right)` 签名保持兼容。这其实是**向参照实现靠拢**——PAML 的 `ConditionalPNode` 本来就是 `for (i = 0; i < nodes[inode].nson; i++)`。

### 快照/回滚

Python 端每轮迭代 `copy.deepcopy(phyloData)`，会把 pandas alignment、qmodel、整棵 Node 图、以及每个内部节点上 `(nstates × nsites)` 的 `lik_vector` 全部深拷一遍。

C++ 侧只拷**移动能改的东西**：拓扑链接（每节点 4 个 int）+ 每个分支一个 double + freqs，一百个 taxa 也就几 KB。**不拷各节点的似然向量**——那是 Python 深拷慢的主因，而且它不可能被移动改到。回滚是充分的，因为 `LogLikPruner()` 每次调用都重算所有内部节点（调用之间没有任何缓存），而叶子的似然只依赖数据。

### 自动调参

手动定的参数让频率类移动的接受率掉到 **0.012**（等于冻住）。加了 MrBayes 那套 Targetrate 调参后：

| 移动 | 调参前 | 调参后 | 目标 |
|---|---|---|---|
| `Multiplier(V)` | 0.318 | 0.254 | 0.25 |
| `TLMultiplier(V)` | 0.034 | 0.250 | 0.25 |
| `Dirichlet(Pi)` | **0.012** | 0.217 | 0.25 |
| `Slider(Pi)` | 0.035 | 0.234 | 0.25 |

**调参必须在 burn-in 后关掉。** 调参器每次窗口偏差就按固定倍率调整，所以它是**震荡而非收敛**；采样期还开着等于提议分布一直在变，这是真实的非平稳性，会表现为 split-half 检验失败（实测 z 从 0.5 涨到 3.7）。

### 验证：先验复现

把似然置为常数，链必须复现先验。这是对 Hastings 比**最直接的证伪**，而且能逐个移动独立检验。它当场抓出两个真 bug：

- **`Multiplier` 的符号写反了。** 推导时 Jacobian 用了 `1/x` 而非 `1/x'`。错了不报错，只会让链**塌缩到零枝长**（树长 0.200 → 0.00000）。
- **乘子指数**。文献里写法不一致，所以没靠推导——用 `GammaDir` 跑先验复现，总树长的边缘分布必须精确是 `Gamma(2,10)`：

  | 指数 | L 均值 | L 方差 |
  |---|---|---|
  | **= 枝数 k** | **0.20025** | **0.019960** |
  | 1 | 0.00000 | — |
  | 3 | 0.00061 | — |

  目标 0.20000 / 0.020000。**指数恰好等于枝数**，这是测出来的。

`test_priors.cpp` 里跑不同指数那个循环就是证据本身，不是注释。

## 14. 与 MrBayes 3.2 的对照

数据：`paml_bench/data/sim16x2000.phy`，16 taxa × 2000 sites（alisim 按 JC69 模拟）。
模型：`lset nucmodel=4by4 nst=1 rates=equal`；先验 `statefreqpr=dirichlet(1,1,1,1)`、`brlenspr=unconstrained:gammadir(1.0,0.1,1.0,1.0)`、`topologypr=uniform`。

### 先说一个踩过的坑：哪个二进制才是真的

目录里有四个 exe，**名字全都会骗人**：

| 二进制 | 实际链接 | 自我报告 |
|---|---|---|
| `src/mb.exe` | **MPI + BEAGLE** | `SSE AVX FMA Beagle MPI` |
| `src/mb-beagle.exe` | 仅 BEAGLE | `SSE AVX FMA Beagle` |
| `dist/mb.exe` | 仅 BEAGLE | `SSE AVX FMA Beagle` |
| `src/bayes.exe` | **两个都没有** | 无 features 行 |

判据是导入表，不是文件名：

```
objdump -p src/bayes.exe | grep "DLL Name"     # 只有 KERNEL32.dll
objdump -p src/mb.exe    | grep "DLL Name"     # 还有 msmpi / beagle
```

**用 `bayes.exe` 配 `mpiexec -n 8` 会启动 8 个互不相干的完整 MrBayes 进程**，各跑各的全量分析。我第一次就是这么测的，得出的"MPI 让每代贵 22 倍"完全是无中生有——那是 8 份冗余工作。

### 单链每代耗时

| 配置 | ms/代 |
|---|---|
| `dist/mb.exe` 单线程，BEAGLE off | **0.0773** |
| `dist/mb.exe` 单线程，BEAGLE on | 0.0929 |
| `pyml._core` | 0.13–0.20 |

### 8 链 × 20000 代（`nruns=2 nchains=4`）

| 配置 | wall time | 相对 |
|---|---|---|
| **`src/mb.exe` MPI `-n 8`，BEAGLE off** | **4.28 s** | 1.00 |
| `src/mb.exe` MPI `-n 1`，BEAGLE off | 12.40 s | 2.90× |
| `dist/mb.exe` 单进程，BEAGLE off | 12.17 s | 2.85× |
| `src/mb.exe` MPI `-n 8`，BEAGLE on | 13.12 s | 3.07× |
| `src/mb.exe` MPI `-n 8`，BEAGLE on，`beaglethreads=no` | 12.88 s | 3.01× |
| `src/mb.exe` MPI `-n 1`，BEAGLE on，`beaglethreads=no` | 17.89 s | 4.18× |
| `pyml._core`（1 链，外推到 8 链） | ~22 s | ~5.2× |

### 两条结论

**1. MPI 有效，8 进程 2.9× 加速。** 亚线性，因为 8 条链之间有温度交换的同步开销。

**2. BEAGLE 在这个规模是稳定的负收益。** 单进程慢 44%（12.40 → 17.89），8 进程慢 3 倍。而且 `beaglethreads=no` 与 auto 几乎一样（12.88 vs 13.12），所以**不是线程超订**——是 BEAGLE 自身的 machinery（partial buffer、scaling、资源管理）在 4 状态小树上收不回来。BEAGLE 的主场是密码子 61 状态和大树。

接受率可以对一下：MrBayes 的 `NNI` 是 4.9%，`pyml._core` 是 5.6%，量级一致。

## 15. 已知差距 —— 下一步在这里

按重要性排：

### 15.1 条件似然缓存（**最大的差距**）

MrBayes 每代跑 **10 个移动**（多出 `Nodeslider`、`ExtSPR`、`ExtTBR`、`ParsSPR`、`ParsTBR`），`pyml._core` 只有 5 个——**它做得更多却更快**。

差距的主因不是移动数量，而是：**MrBayes 只重算受影响的路径**。改一条枝长时，只有从那条枝到根的路径上的节点需要重算，其余的条件似然可以复用。pyml 每次 `LogLikPruner()` 都从叶子重算所有内部节点。

`Mcmc::Step()` 目前是「快照 → 提议 → 全量重算 → 接受/回滚」。要做缓存，需要：

- 每个节点带一个「有效」标记，提议时按移动类型标脏（`Multiplier` 只脏一条路径，`NNI` 脏局部，`TLMultiplier` 全脏）
- 接受时保留标记，拒绝时按快照恢复标记
- 这和 mcmctree 的 `com.oldconP` 是同一套机制（PAML 的 `ConditionalPNode` 里已经有：`if (nodes[ison].nson > 0 && !com.oldconP[ison])`）

注意这会把「快照 = 纯数据拷贝」变成「快照 = 数据 + 脏标记」，回滚的正确性论证要跟着更新。

### 15.2 多线程：加在哪条轴上（**结论：不是 pattern 轴**）

考虑过用 `std::thread` 加一个类似 BEAGLE 的线程池。实测下来这条路不成立。

**Amdahl 上限。** arm3（向量化后）似然 0.212 s / 总 0.455 s = **47%**。即使无限线程、零同步开销，端到端也只有 `1/(1-0.47)` = **1.89×**。

**并行区太短。** 单次似然 **24 µs**（16 taxa）/ 113 µs（64 taxa）。pattern 轴切分的话，8 线程每人只有 3 µs。

**同步比计算还贵。** 实测空线程体的 fork/join：

| n | µs/次 |
|---|---|
| 2 | 70 |
| 4 | 144 |
| 8 | 282 |
| 16 | 564 |

约 35 µs/线程，是 Windows 的线程创建成本。**线程创建本身就比整个似然贵 3–23 倍。** 常驻池能免掉创建成本，但 release+rendezvous 在 Windows 上仍是 ~10 µs 量级；套在 24 µs 的区域上理论上限掉到 ~1.3×，而 6 taxa 那档并行区只有 4–5 µs，完全没救。

**已经有实测的负面证据。** MrBayes 的 BEAGLE **就是** pattern 轴线程池。同一台机器、同一份数据：

| 8 链 × 20000 代 | wall time |
|---|---|
| MPI `-n 8`，BEAGLE **off** | **4278 ms** |
| MPI `-n 8`，BEAGLE **on** | 13115 ms |

**开了线程池反而慢 3 倍**，且 `beaglethreads=no` 与 auto 几乎一致（12875 vs 13115），所以不是线程超订。

**该加在哪：每个线程算一整份独立的似然，而不是切分一份似然。**

| 轴 | 粒度 | 现成条件 |
|---|---|---|
| 速率类别（`fx_r` 的 `ir` 循环） | 一整个类别的似然 | `com.conPSiteClass` 已给每类分了独立 conP 块 |
| 基因座（`lnpData` 的 locus 循环） | 一整个基因的似然 | 需要 `ndata > 1` |
| **链** | 一条完整的链（百万代） | 同步只发生在温度交换时 |

链级并行的粒度最好。

**PhyloSuite 已经这么做了，而且没改 mcmctree 一行。** `PhyloSuite/src/Lg_MCMCTree.py:2501-2520`：

```python
seeds = random.sample(range(2147483647), self.chains)
command_list = [self.prepare_for_run(fold_suffix=f"run{i+1}", seed=seeds[i])
                for i in range(self.chains)]
async_results = [self.pool.apply_async(run, args=(...)) for ... in command_list]
```

`multiprocessing.Pool(processes=self.chains)`，每个 worker `os.chdir` 进自己的目录（`run1/`、`run2/`…）后 `subprocess` 起一个**未经修改的 mcmctree**。这是链级并行，不是链内并行——**它提高吞吐量，不缩短单链的墙钟时间**，用 PAML 自带的 `--combine` 工作流手写 shell 循环也能得到同样效果。

**所以顺序是：先做 15.1 的缓存（减少工作量、无同步开销、单线程），剩下的再考虑按类别/基因座并行。** 若要把链级并行放进 `pyml._core`，`std::thread` 在这里是可行的——粒度是一条链，同步成本可忽略，而且比 PhyloSuite 的进程方案省内存（alignment 只读共享）。

### 15.3 Node scaling（**正确性缺口，不是性能**）

**`pyml` 的 C++ 完全没有 scaling。** 从根到叶是纯乘法，没有任何重标定。整个 C++ 侧搜不到 rescale 逻辑。

实测在 **500 taxa 处撞底**（长枝 + 随机序列 + caterpillar）：

| taxa | 实测 lnL | |
|---|---|---|
| 400 | -4514.50 | 正常 |
| **500** | **-5526.20** | ← 撞到 `1e-300` |
| 600 | -5526.20 | 一样 |
| 800 | -5526.20 | 一样 |

`-5526.2042 = 8 × log(1e-300)`，之后**完全不随树大小变化**。结果是错的，但不报错——`LogLikPruner()` 里那个 `std::max(site_likelihood, Precision{1e-300})` 把它静默截断了。

阈值取决于枝长和数据，不是固定数字。（顺带：**短枝不是触发条件**，触发条件是树的大小。短枝反而让匹配状态的 partial 保持在 1 附近而不衰减。）

对照：PAML 有 `NodeScale`、MrBayes 有 scaling、BEAGLE 有 scaling buffer。pyml 是这三个里唯一没有的。

### 15.4 移动集

还缺 MrBayes 默认集里的 5 个：

| 移动 | 可行性 |
|---|---|
| `Nodeslider(V)` | 需要节点高度；无根树上有定义，但要重新推导 |
| `ExtSPR` / `ExtTBR` | 扩展式 SPR/TBR 的 Hastings 比是文献里出了名难写对的，**只能近似**，需要单独数值检验量化近似程度 |
| `ParsSPR` / `ParsTBR` | 额外需要 Fitch 简约性打分（含每个回植点的分数），偏置项要进 Hastings 比 |

已验证正确的那 5 个的 Hastings 比都有闭式；上面这 5 个不是。

### 15.5 ARM NEON（**待补**）

`vectorize.cpp` 现在只有 x86 路径：`<immintrin.h>` 是无条件包含的，`DefaultOps` 按 `__AVX512F__` / `__AVX__` / `__SSE2__` 选择，SSE2 以下退到标量。**没有 NEON 分支，在 ARM 上编译不过**（`immintrin.h` 不存在）。

要补的东西：

- `NeonOps` 结构，接口与现有 `Ops` 一致（`LOADV`/`STOREV`/`MULTV`/`ADDV`/`SETV`/`FMADDV`/`WIDTH`）。ARMv8 有 FMA（`vmlaq_f64`/`vfmaq_f64`），走 `__ARM_NEON` 分支。
- `#include <immintrin.h>` 与 `<arm_neon.h>` 都要条件化。
- `DefaultOps` 的选择链加 `__ARM_NEON` 一支。
- `PmatMulFixedDouble<NS>` 的寄存器 tile 大小要重新核对：NEON 是 128 位（2 个 double），比 AVX 窄一半，`NS=4` 时 `cv[4]`+`pv[4]` 共 8 个向量寄存器，ARMv8 有 32 个，够用。
- `PYML_VEC_MAX_STATES` 在通用回退路径上按向量宽度分配栈数组，NEON 下会翻倍。
- 尾部的零填充逻辑与宽度无关，不用改。
- 编译期断言 `-march=native` 在 ARM 上等价于 `-march=armv8-a+simd` 之类，构建脚本要能识别。

验证手段现成：`test_vectorize_c.c` 和 `bench_likelihood.cpp` 都与平台无关，在 ARM 上重跑即可；`test_priors.cpp` 会顺带验证移动的 Hastings 比没有因 SIMD 宽度变化而漂移（浮点舍入差异）。

### 15.6 其它

- **多链与温度交换。** `MCMCMC` 和 `SS`（stepping stone）都还没做。`Mcmc` 已经暴露了 `log_posterior`/`current_*`，接口留好了。
- **`pyml` 包本身导入失败**（循环导入 + `phyloData.py` 类定义顺序），所以 `import pyml._core` 这条正常路径走不通，`test_bindings.py` 是直接 `import _core` 绕开的。修那三处很小，但它是「与 Python 端对拍」的前提。
- **`likelihood.hpp` 仍是 scalar `double` 单精度路径**，`PmatMulF32` 有但没接。

## 16. 环境注意事项

**C: 盘在本次工作中被写满到 100%**（327GB 剩 587MB）。表现是 GCC 报 `out of memory allocating 65536 bytes`，而机器还有 24GB 空闲——**报错完全是误导性的**，真因是临时目录写不下。

`build_core.py` 把 `TMPDIR`/`TEMP`/`TMP` 指向 `PYML_BUILD_TMP`（默认 `<repo>/../gcctmp`）。手工编译时也要记得设，否则会看到同样的假内存错误。

## 17. 复现

```bash
# 构建扩展模块（会自动处理 PATH 和临时目录）
/d/PyMOL/python.exe pyml/cpp/build_core.py

# C++ 侧的移动正确性（先验复现）—— 这一条最重要
g++ -std=c++20 -O3 -march=native -I pyml/cpp -o test_priors \
    pyml/cpp/test_priors.cpp pyml/cpp/vectorize.cpp
./test_priors

# Python 侧的绑定与 Python 自定义移动
/d/PyMOL/python.exe pyml/cpp/test_bindings.py

# 跑长链，产出 Tracer 可读的 .p
/d/PyMOL/python.exe pyml/cpp/run_trace.py <phylip> <generations> <prefix>
```

MrBayes 侧的对照（注意用对二进制）：

```bash
MB=/e/Codes/YR-MPE-Windows-x86_64-softwares/MrBayes-MPI-BEAGLE3/src/mb.exe
export PATH="/d/Program Files/Microsoft MPI/Bin:$PATH"
mpiexec -n 8 "$MB" cmd.txt     # cmd.txt 见第 14 节的模型与先验设置
```

## 18. 未决（第二部分）

- `kMinFreq = 1e-6` 是对单纯形的一处截断（`Dirichlet(1,1,1,1)` 在边界附近不排斥，而极端频率会让 Q 的特征分解不收敛）。截断概率约 4e-6，但它是**目标分布的一部分**，`DirichletPrior` 和频率类移动必须保持一致——改动其中一侧会静默改变后验。
- `Core::TrySetFreqs` 用异常做控制流（提议退化时抛了再捕获）。只在极端提议上触发，不在热路径，但值得知道。
- **逐枝长 trace 的列编号**用 `BranchLabel`（「不含 tip0 那一侧的最小 tip 号」），是拓扑无关的。MrBayes 用自己的规范编号，两者不可直接对照列号。
- trace 存的是 `std::vector<std::string>` 的 Newick（可选），10 万样本时内存不小；只在需要 `.t` 时才开。
