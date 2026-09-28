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

**PAML 侧（补丁，默认关闭）**

| 文件 | 内容 |
|---|---|
| `src/mcmctree.c` | `vecLnL` 开关 + `ConditionalPNodeVec` + 影子双缓冲 + 叶子指示块预计算 + `vecLnL=2` 自校验模式 + 似然分段计时 |
| `src/treesub.c` | `lfun()` 里的分支，用 `#ifdef MCMCTREE` 隔离（baseml/codeml 不受影响，已验证仍可编译） |

**基准环境**（`E:\Codes\PYML\paml_bench\`）

`run_bench.py`（四臂驱动）、`prep_data.py`（alisim 造数据）、`kernel_bench.c`（独立内核微基准）、`bin/`、`data/`、`out/`。

---

## 9. 复现

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

## 10. 建议

1. ~~修 `likelihood.hpp` 的 `compute_node`~~ —— 已完成，见第 8 节。
2. ~~`SubstMatrix::PMatrix` 按值返回 `std::vector`~~ —— 已改为非分配签名。
3. **`transpose2` / 叶子和根之间的中间节点仍会走完整 matmul。** 对只有 4 个 state 的问题，`PmatMulFixedDouble<4>` 的 16 次 FMA 里有 12 次乘的是 0/1 指示块（叶子）。要再进一步，得按 state 分组 pattern——代价是每个叶子一套不同的 pattern 序，会破坏共享的 site 轴。收益未知，先不动。
4. **离散 Γ 的 site-class 路径仍未验证**（`fx_r`/`conPSiteClass`）。本次明确限定 JC69 单速率。
5. pyml 侧还没有任何绑定。做绑定时注意 `LikelihoodCalculator::LogLikPruner()` 每次调用返回单个标量——若要暴露梯度或逐 site 似然，需要另行设计接口。

## 未决

- 本轮所有测量都跑在一台会降频的笔记本上。除了 kernel 微基准外，端到端数字建议在稳定机器上复测。
- `vecLnL=1` 下 `nodes[].conP` 仍被分配（约 20MB @64taxa）却从不读取。可以省掉，但会让 `GetMem` 的 offset 数学与标量路径分叉，暂未动。
- `mcmctree.c` 的 `ConditionalPNodeVec` 与 `likelihood.hpp` 的 `compute_node` 现在是**两份独立实现**了同一件事。逻辑上应该收敛成一份（mcmctree 侧调用 pyml 的函数），但本次刻意没做——基准的价值恰恰在于两条路径互相独立、可以互相对拍。真要合并，应在这轮验证结论固化之后。
