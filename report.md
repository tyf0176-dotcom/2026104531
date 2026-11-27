# 端侧 RAG 文件检索：图索引与聚类索引的实现与评测

**学号分支**：`2026104531`　**课程仓库**：`w-pq/cpu-file-retrieval-lab`（fork 至 `tyf0176-dotcom/2026104531`）

---

## 摘要

按作业要求，我在 `indexes/graph.py` 实现了 HNSW 图索引，在 `indexes/cluster.py` 实现了 IVF 簇索引（含 IVF-Flat 与 IVF-PQ 两种量化模式，仅用 NumPy，未调用任何现成向量库），并在同一批数据、同一套评测口径下完成了开发集参数扫描、测试集复测与在线 RAG 耗时拆解。

三条主要结论：

1. **两个索引都达到可用精度**：HNSW 在 `efSearch≥32` 时片段召回 0.980–1.000，IVF-Flat 在 `nprobe=16` 时 0.980、`nprobe=32` 时 0.990–1.000；选定配置在测试集上的文件命中率与精确检索完全持平（HNSW 0.425、IVF-Flat `nprobe≥32` 0.425，基线同为 0.425）。
2. **在 8,045×384 这个规模上，聚类索引反而比图索引快 3–4 倍**：同为 recall≈0.98，IVF-Flat（`K=128, nprobe=16`）用 0.077 ms，HNSW（`M=16, efSearch=32`）用 0.318 ms。原因是 IVF-Flat 每次查询只触发几次大矩阵乘（BLAS 高效），而 HNSW 的一次查询要做上百次小批量距离计算，Python/NumPy 的调用开销超过了实际浮点运算量。
3. **乘积量化（PQ）在本数据集上不划算**：它把 12 MB 压到 0.28–0.89 MB，但精确 top-10 的排名边际间隔只有 0.0058，而 8 bit/16 子空间的 PQ 把内积扰动到 0.032（前者的 5.5 倍），于是即便全表扫描，片段召回也只有 0.43（4 bit 更差，0.28）。文件级 Hit@5 因为按文件去重而被"聚合"掉，仍保持在 0.40–0.45。

端到端层面，完整请求 ≈ 504 ms 中生成占 88%（445 ms，约 61 token/s），检索只占 0.1% 量级（0.54 ms），说明该规模下真正决定用户体验的是 0.5B 模型的解码速度，而不是索引。

---

## 1. 环境与复现信息

| 项目 | 值 |
|---|---|
| 机器 | Apple M5，10 逻辑核，16 GB 内存，macOS（arm64） |
| 容器运行时 | Docker Desktop 4.100.1（Engine 29.9.3、Compose v5.7.1），虚拟机配额为课程要求的 4 vCPU / 8 GiB |
| 课程容器配额 | 4 vCPU（`compose.yaml` 中 `cpus: 4.0`，`network_mode: none`） |
| 线程设置 | `LAB_BLAS_THREADS=1`（NumPy/BLAS 单线程）、`LAB_MODEL_THREADS=4`（编码器） |
| Python / 依赖 | 容器：Python 3.11 + `docker/framework-lock.txt`（numpy 1.26.4、torch 2.9.1-cpu、transformers 5.3.0、langgraph 1.0.10） |
| 嵌入模型 | `sentence-transformers/all-MiniLM-L6-v2`，revision `c9745ed1` |
| 生成模型 | `Qwen2.5-0.5B-Instruct-Q4_K_M`，sha256 校验通过（`74a4da8c…93d7a9db`） |
| 评测逻辑 | `benchmark/runner.py`（与 `bench` 命令同一份代码） |
| 代码提交号 | `edbc974`（本报告对应的实现与结果所在提交） |
| 测量方式 | 容器内 `bench search` / `bench online` / `bench embed`，原始输出在 `results/bench-*.jsonl` |

**关于镜像构建的说明（如实披露）**：本机对几个构建依赖站点不可达或限速严重 —— `deb.debian.org` 的 `binary-arm64/Packages` 反复 `Connection failed`，`files.pythonhosted.org` 读超时，`download.pytorch.org` 实测只有 0.01 MB/s（5 MB 用了 460 秒）。因此我用了一份**仓库外的本地 Dockerfile 副本**构建镜像，改动仅三处且不涉及任何业务代码：

1. apt 源 `http://deb.debian.org` → `http://mirrors.aliyun.com`（同一 Debian bookworm 仓库，包版本一致）；
2. pip 索引指向阿里云 PyPI 镜像，并加大 `timeout/retries`；
3. arm64 上 `torch==2.9.1` 改从 PyPI 取，去掉 `--index-url download.pytorch.org/whl/cpu` —— aarch64 的 PyPI 轮子本来就是纯 CPU 构建，安装后 `torch.__version__` 仍是 `2.9.1+cpu`，与官方路径一致。

**仓库里的 `Dockerfile` 一个字未改**，包名与版本全部沿用 `docker/framework-lock.txt`；`compose.yaml` 里的镜像名仍为 `cpu-file-retrieval-lab:2.0`，容器启动参数与健康检查完全按原文件执行。构建产物核对：`numpy 1.26.4 / torch 2.9.1+cpu / transformers 5.3.0`，编码器与 Qwen 权重经 `fetch_models.py` 下载并校验 sha256 通过。`indexes/` 以只读卷挂载，容器内运行的正是本报告提交的代码。

**索引空间列**：`bench` 不输出该字段，表中数值由容器内调用两个索引的 `size_bytes()` 单独测得（`results/bench-index-size.jsonl`），与正式测量同一份代码、同一份数据。

**在线耗时的测量协议**：每配置先跑一轮丢弃的冷态测量，表中为第二轮稳态值。两轮差异见 §6.3 —— 冷态下前两条查询的生成要 6–7 秒（prefill 上千 token 的检索上下文），这不是随机噪声，而是 llama-server 的 prompt 缓存与 `-c 2048` 共同决定的固定成本。

---

## 2. 数据与评测口径

- 186 份判决，切成 **8,045 个片段**，统一用 all-MiniLM-L6-v2 编码为 384 维单位向量（`data/documents.npy`，已 L2 归一化，故内积即余弦）。
- 50 条查询：`int(id) % 5 == 0` 为**开发集 10 条**，其余**测试集 40 条**；`data/test.jsonl` 给出 195 条"查询—相关文件"标注，每条查询至少 1 个相关文件，平均 4.4 个。
- **Recall@10**：索引返回的前 10 个片段与精确 top-10 的重合比例，对所有查询取平均。
- **Hit@5**：把返回的 50 个片段按首次出现顺序去重成文件列表，取前 5 个文件，只要有一个被标注为相关即命中。
- **延迟**：只计 `index.search()` 调用本身，P50/P95 取所有查询 × 3 次重复。

---

## 3. 实现

### 3.1 `indexes/graph.py`：HNSW

- **分层**：`level = floor(-ln(U) · mL)`，`mL = 1/ln(M)`，`U ~ U(0,1)`，用固定 seed 生成，保证同一组参数每次建库完全一致（`build_ms` 可复现）。
- **插入**：从入口点起，在第 `top_level..level+1` 层贪心下降；在第 `min(level, top_level)..0` 层用宽度 `efConstruction` 的 beam search 取候选，再用**邻居选择启发式**（候选必须比已有邻居更接近查询才保留，不足则用最近的补齐）取前 `M` 个，双向连边，0 层上限 `2M`、其余层 `M`，超限按同一启发式剪枝。
- **查询**：上层贪心到 1 层，底层用宽度 `efSearch` 的 beam search，返回前 `k`。
- **参数**：构建 `M`、`efConstruction`、`seed`；查询 `efSearch`。
- **工程细节**：邻接表是"层 × 节点 → 邻居 id 列表"，每次扩展只做一次 `vectors[邻居数组] @ q` 的批量距离计算；候选队列与结果集用 `heapq`（负号存最大堆）；提前终止条件为"当前候选相似度已低于结果集最差且结果集已满"。

### 3.2 `indexes/cluster.py`：IVF-Flat / IVF-PQ

同一份实现支持两种模式，由 `nbits` 切换（`nbits=0` → IVF-Flat，`nbits>0` → IVF-PQ）：

- **粗聚类**：自写 k-means（k-means++ 播种 + Lloyd 迭代，空簇随机重播种，固定 seed），得到 `nlist` 个簇心；按最近簇心把片段分到倒排表（`argsort` 保证顺序稳定）。
- **残差**：`r = x - centroid[assign]`。
  - IVF-Flat：残差原样保存（按倒排表顺序连续存放，便于切片）。
  - IVF-PQ：把 384 维残差切成 `m` 段，每段单独训练 `2^nbits` 个中心的码本，编码为 `uint8`。
- **查询**：按 `q · centroid` 选最近的 `nprobe` 个簇；IVF-Flat 直接在这些簇的残差上做一次矩阵乘；IVF-PQ 先用一次 `einsum` 建出 `m × 2^nbits` 的查找表，再对候选码求和。
- **参数**：构建 `nlist`、`nbits`、`m`、`seed`、`iters`；查询 `nprobe`。

### 3.3 一个容易写错的口径：PQ 该估什么分数

评测按余弦（内积）排序，而 PQ 的常规做法是估**残差 L2**。两者在候选簇相同时只差一个常数，但跨簇比较时会引入偏差，且量化会把向量"缩短"（本数据重构后模长均值 0.84），L2 排序会系统性偏向被截断得少的向量。

因此我把打分写成 **`score = q·c + Σ_s q_sub · book_s[code_s]`**，它严格等于 `q · x̂`（`x̂` 为重构向量）。我做了逐条核对：LUT 累加值、`q·x̂`、手工点积三者在小数点后 5 位一致，所以 PQ 模式下召回偏低完全来自**量化误差**，而不是打分公式错位（第 7 节给出定量证据）。

---

## 4. 基线

| 指标 | 开发集（10 条） | 测试集（40 条） |
|---|---:|---:|
| 精确检索 Recall@10 | 1.000 | 1.000 |
| 精确检索 Hit@5 | 0.400 | 0.425 |
| 精确检索 P50 / P95 | 0.522 / 0.640 ms | 0.503 / 0.642 ms |
| 整库文档嵌入 `doc_embed_s` | 330.41 s（8,045 段，4 线程，≈24 段/s） | 同左 |

精确检索在 8,045×384 上只需 0.50 ms —— 这个数字决定了后面所有对比的基调：**本规模下"避免全扫"能省下的绝对时间极小**。

（`doc_embed_s` 比宿主机慢约 7 倍：容器是 aarch64 Linux VM，PyTorch 只能用通用 NEON 内核；宿主机上走的是 macOS Accelerate/AMX 路径。这是虚拟化环境带来的真实差异，不是测量误差，同一现象在查询嵌入上也一致 —— 容器 57 ms vs 宿主机 12 ms。）

---

## 5. 完整参数表

> 下表由 `tools/make_report.py` 从 `results/bench-*.jsonl`（容器内 `bench` 原始输出）自动生成，请勿手工编辑。空间列统计索引自身额外占用（HNSW 按每条有向边 4 B 的 id 计；IVF-Flat 含残差矩阵；IVF-PQ 含码本与码字），由 `results/bench-index-size.jsonl` 提供。

<!--TABLES:start-->
## 开发集（10 条查询）

### 图索引 HNSW

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| HNSW M=16, efC=100 | efSearch=16 | 11534 | 894 KB | 0.990 | 0.400 | 0.317 | 0.346 |
| HNSW M=16, efC=100 | efSearch=32 | 11534 | 894 KB | 0.990 | 0.400 | 0.318 | 0.339 |
| HNSW M=16, efC=100 | efSearch=64 | 11534 | 894 KB | 0.990 | 0.400 | 0.390 | 0.414 |
| HNSW M=16, efC=100 | efSearch=128 | 11534 | 894 KB | 1.000 | 0.400 | 0.747 | 0.789 |
| HNSW M=16, efC=100 | efSearch=256 | 11534 | 894 KB | 1.000 | 0.400 | 1.409 | 1.564 |
| HNSW M=16, efC=200 | efSearch=32 | 17616 | 900 KB | 0.990 | 0.400 | 0.332 | 0.347 |
| HNSW M=16, efC=200 | efSearch=16 | 17616 | 900 KB | 0.990 | 0.400 | 0.333 | 0.360 |
| HNSW M=16, efC=200 | efSearch=64 | 17616 | 900 KB | 0.990 | 0.400 | 0.405 | 0.428 |
| HNSW M=16, efC=200 | efSearch=128 | 17616 | 900 KB | 1.000 | 0.400 | 0.765 | 0.823 |
| HNSW M=16, efC=200 | efSearch=256 | 17616 | 900 KB | 1.000 | 0.400 | 1.456 | 1.622 |
| HNSW M=8, efC=100 | efSearch=32 | 6025 | 484 KB | 0.980 | 0.300 | 0.216 | 0.231 |
| HNSW M=8, efC=100 | efSearch=16 | 6025 | 484 KB | 0.980 | 0.300 | 0.221 | 0.237 |
| HNSW M=8, efC=100 | efSearch=64 | 6025 | 484 KB | 0.990 | 0.400 | 0.269 | 0.287 |
| HNSW M=8, efC=100 | efSearch=128 | 6025 | 484 KB | 1.000 | 0.400 | 0.494 | 0.520 |
| HNSW M=8, efC=100 | efSearch=256 | 6025 | 484 KB | 1.000 | 0.400 | 0.950 | 1.002 |

### 聚类索引 · IVF-Flat

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| IVF-Flat K=128 | nprobe=1 | 1316 | 12.0 MB | 0.380 | 0.300 | 0.022 | 0.026 |
| IVF-Flat K=128 | nprobe=2 | 1316 | 12.0 MB | 0.640 | 0.400 | 0.026 | 0.032 |
| IVF-Flat K=128 | nprobe=4 | 1316 | 12.0 MB | 0.810 | 0.400 | 0.032 | 0.040 |
| IVF-Flat K=128 | nprobe=8 | 1316 | 12.0 MB | 0.920 | 0.400 | 0.047 | 0.061 |
| IVF-Flat K=128 | nprobe=16 | 1316 | 12.0 MB | 0.980 | 0.400 | 0.077 | 0.103 |
| IVF-Flat K=128 | nprobe=32 | 1316 | 12.0 MB | 1.000 | 0.400 | 0.126 | 0.166 |
| IVF-Flat K=128 | nprobe=64 | 1316 | 12.0 MB | 1.000 | 0.400 | 0.218 | 0.292 |
| IVF-Flat K=128 | nprobe=128 | 1316 | 12.0 MB | 1.000 | 0.400 | 0.408 | 0.525 |
| IVF-Flat K=32 | nprobe=1 | 1055 | 11.9 MB | 0.550 | 0.500 | 0.024 | 0.031 |
| IVF-Flat K=32 | nprobe=2 | 1055 | 11.9 MB | 0.730 | 0.400 | 0.035 | 0.045 |
| IVF-Flat K=32 | nprobe=4 | 1055 | 11.9 MB | 0.840 | 0.400 | 0.054 | 0.075 |
| IVF-Flat K=32 | nprobe=8 | 1055 | 11.9 MB | 0.980 | 0.400 | 0.098 | 0.134 |
| IVF-Flat K=32 | nprobe=16 | 1055 | 11.9 MB | 0.990 | 0.400 | 0.162 | 0.239 |
| IVF-Flat K=32 | nprobe=128 | 1055 | 11.9 MB | 1.000 | 0.400 | 0.277 | 0.399 |
| IVF-Flat K=32 | nprobe=64 | 1055 | 11.9 MB | 1.000 | 0.400 | 0.286 | 0.401 |
| IVF-Flat K=32 | nprobe=32 | 1055 | 11.9 MB | 1.000 | 0.400 | 0.287 | 0.408 |
| IVF-Flat K=64 | nprobe=1 | 1130 | 11.9 MB | 0.460 | 0.500 | 0.020 | 0.023 |
| IVF-Flat K=64 | nprobe=2 | 1130 | 11.9 MB | 0.640 | 0.500 | 0.027 | 0.032 |
| IVF-Flat K=64 | nprobe=4 | 1130 | 11.9 MB | 0.850 | 0.400 | 0.040 | 0.048 |
| IVF-Flat K=64 | nprobe=8 | 1130 | 11.9 MB | 0.940 | 0.400 | 0.060 | 0.085 |
| IVF-Flat K=64 | nprobe=16 | 1130 | 11.9 MB | 0.980 | 0.400 | 0.103 | 0.160 |
| IVF-Flat K=64 | nprobe=32 | 1130 | 11.9 MB | 0.990 | 0.400 | 0.179 | 0.271 |
| IVF-Flat K=64 | nprobe=64 | 1130 | 11.9 MB | 1.000 | 0.400 | 0.311 | 0.456 |
| IVF-Flat K=64 | nprobe=128 | 1130 | 11.9 MB | 1.000 | 0.400 | 0.318 | 0.459 |

### 聚类索引 · IVF-PQ

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| IVF-PQ K=64, m=16, 4bit | nprobe=1 | 2266 | 278 KB | 0.230 | 0.300 | 0.036 | 0.046 |
| IVF-PQ K=64, m=16, 4bit | nprobe=4 | 2266 | 278 KB | 0.280 | 0.300 | 0.068 | 0.085 |
| IVF-PQ K=64, m=16, 4bit | nprobe=8 | 2266 | 278 KB | 0.280 | 0.300 | 0.137 | 0.158 |
| IVF-PQ K=64, m=16, 4bit | nprobe=16 | 2266 | 278 KB | 0.280 | 0.300 | 0.281 | 0.328 |
| IVF-PQ K=64, m=16, 4bit | nprobe=64 | 2266 | 278 KB | 0.280 | 0.300 | 0.867 | 0.889 |
| IVF-PQ K=64, m=16, 8bit | nprobe=1 | 4011 | 638 KB | 0.340 | 0.400 | 0.044 | 0.051 |
| IVF-PQ K=64, m=16, 8bit | nprobe=4 | 4011 | 638 KB | 0.430 | 0.400 | 0.086 | 0.108 |
| IVF-PQ K=64, m=16, 8bit | nprobe=8 | 4011 | 638 KB | 0.430 | 0.400 | 0.142 | 0.171 |
| IVF-PQ K=64, m=16, 8bit | nprobe=16 | 4011 | 638 KB | 0.430 | 0.400 | 0.277 | 0.332 |
| IVF-PQ K=64, m=16, 8bit | nprobe=64 | 4011 | 638 KB | 0.430 | 0.400 | 0.860 | 0.890 |
| IVF-PQ K=64, m=48, 8bit | nprobe=1 | 6433 | 889 KB | 0.390 | 0.500 | 0.078 | 0.092 |
| IVF-PQ K=64, m=48, 8bit | nprobe=4 | 6433 | 889 KB | 0.630 | 0.400 | 0.156 | 0.214 |
| IVF-PQ K=64, m=48, 8bit | nprobe=8 | 6433 | 889 KB | 0.670 | 0.400 | 0.317 | 0.406 |
| IVF-PQ K=64, m=48, 8bit | nprobe=16 | 6433 | 889 KB | 0.680 | 0.400 | 0.604 | 0.698 |
| IVF-PQ K=64, m=48, 8bit | nprobe=64 | 6433 | 889 KB | 0.680 | 0.400 | 1.878 | 1.968 |

### 精确基线 (exact)

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| exact (baseline) | — | 0 | — | 1.000 | 0.400 | 0.522 | 0.640 |

## 测试集（40 条查询）

### 图索引 HNSW

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| HNSW M=16, efC=100 | efSearch=32 | 11320 | 894 KB | 0.997 | 0.425 | 0.327 | 0.360 |
| HNSW M=16, efC=100 | efSearch=128 | 11320 | 894 KB | 0.997 | 0.425 | 0.741 | 0.807 |
| HNSW M=8, efC=100 | efSearch=32 | 6022 | 484 KB | 0.975 | 0.425 | 0.226 | 0.242 |
| HNSW M=8, efC=100 | efSearch=128 | 6022 | 484 KB | 0.995 | 0.425 | 0.496 | 0.527 |

### 聚类索引 · IVF-Flat

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| IVF-Flat K=128 | nprobe=8 | 1306 | 12.0 MB | 0.842 | 0.425 | 0.049 | 0.057 |
| IVF-Flat K=128 | nprobe=16 | 1306 | 12.0 MB | 0.943 | 0.400 | 0.078 | 0.097 |
| IVF-Flat K=128 | nprobe=32 | 1306 | 12.0 MB | 0.992 | 0.425 | 0.132 | 0.168 |
| IVF-Flat K=128 | nprobe=64 | 1306 | 12.0 MB | 1.000 | 0.425 | 0.232 | 0.294 |
| IVF-Flat K=64 | nprobe=8 | 1122 | 11.9 MB | 0.900 | 0.450 | 0.066 | 0.085 |
| IVF-Flat K=64 | nprobe=16 | 1122 | 11.9 MB | 0.983 | 0.400 | 0.113 | 0.147 |
| IVF-Flat K=64 | nprobe=32 | 1122 | 11.9 MB | 0.998 | 0.425 | 0.195 | 0.269 |
| IVF-Flat K=64 | nprobe=64 | 1122 | 11.9 MB | 1.000 | 0.425 | 0.326 | 0.443 |

### 聚类索引 · IVF-PQ

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| IVF-PQ K=64, m=16, 8bit | nprobe=8 | 4011 | 638 KB | 0.385 | 0.325 | 0.150 | 0.178 |
| IVF-PQ K=64, m=16, 8bit | nprobe=64 | 4011 | 638 KB | 0.388 | 0.325 | 0.815 | 0.849 |

### 精确基线 (exact)

| 配置 | 查询参数 | 建库耗时 (ms) | 索引额外空间 | Recall@10 | Hit@5 | 查询 P50 (ms) | 查询 P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| exact (baseline) | — | 0 | — | 1.000 | 0.425 | 0.503 | 0.642 |

## 在线 RAG 耗时（测试集 3 条固定查询）

| 配置 | 查询参数 | 查询嵌入 (ms) | ANN 检索 (ms) | 生成 (ms) | 完整请求 (ms) | 框架/聚合 (ms) | 输出 token |
|---|---|---:|---:|---:|---:|---:|---|
| exact (baseline) | — | 57.4 | 0.65 | 452 | 511 | 0.8 | 22/29/37 |
| HNSW M=16, efC=100 | efSearch=32 | 57.1 | 0.54 | 445 | 504 | 0.8 | 22/29/37 |
| HNSW M=16, efC=100 | efSearch=128 | 58.8 | 1.15 | 465 | 525 | 0.2 | 22/29/37 |
| IVF-Flat K=64 | nprobe=16 | 58.2 | 0.20 | 473 | 532 | 0.8 | 22/29/37 |
| IVF-Flat K=64 | nprobe=64 | 76.6 | 0.58 | 550 | 638 | 10.4 | 22/29/37 |
| IVF-PQ K=64, m=16, 8bit | nprobe=8 | 57.1 | 0.26 | 655 | 713 | 0.0 | 8/32/61 |
| IVF-PQ K=64, m=16, 8bit | nprobe=64 | 62.6 | 1.06 | 719 | 784 | 0.8 | 8/32/61 |
<!--TABLES:end-->

### 图

![延迟–召回曲线（开发集）](results/report-latency-recall-dev.png)

![图索引放大：召回平台](results/report-hnsw-zoom-dev.png)

![测试集选定配置](results/report-test-selected.png)

![在线耗时拆解](results/report-online-breakdown.png)

---

## 6. 三个必答问题

### 6.1 构建参数的影响（固定查询参数）

**HNSW 的 `M`（每层保留邻居数）**：`M` 8→16 时，建库 6.03 s→11.53 s（+91%），邻接表 484 KB→894 KB（+85%），`efSearch=32` 下召回 0.980→0.990，查询 P50 0.216→0.318 ms（+47%）。建库成本与空间几乎线性于 `M`（边数 ∝ `M`），而召回只涨 1 个百分点 —— 在这个规模 `M=8` 已经够用。

**HNSW 的 `efConstruction`（建库搜索宽度）**：`M=16` 时 100→200，建库 11.53 s→17.62 s（+53%），空间只 +0.7%（894→900 KB），但召回（0.990）与 P50（0.318 ms）**完全不变**。原因是 8 千点的小图本身连通性很好，`ef=100` 已能找到好邻居，再加大只是多花建库时间。这是"构建参数并非越大越好"的直接证据。

**IVF 的 `nlist`（簇数）**：32→128 时建库只从 1.06 s 涨到 1.32 s（+25%），空间几乎不变（+0.8%，因为 IVF-Flat 存的是原始残差）。固定 `nprobe=8` 时召回反而从 0.980 降到 0.920 —— 因为 `nprobe=8` 覆盖的向量比例从 25% 掉到 6%。但若按"达到同一召回所需延迟"比较，`nlist=128` 更好：recall≈0.98 时 `K=32` 要 `nprobe=16`（0.162 ms），`K=128` 只需 `nprobe=16` 的 0.077 ms。簇更细 ⇒ 每个簇更纯 ⇒ 用更少候选换同召回 ⇒ 更快。

**IVF-PQ 的量化参数**：`m` 16→48（子空间 24→8 维，8 bit 码本）时，空间 638→889 KB，满扫召回 0.43→0.68，而 `nprobe=64` 的延迟 0.86→1.88 ms（查找表与码字聚合的成本随 `m` 线性增长）。`nbits` 4→8 时空间 278→638 KB、满扫召回 0.28→0.43、延迟几乎不变（0.87→0.86 ms）。可见压缩率与精度是硬性取舍，且**在本数据上无论怎么调参都到不了可用召回**。

### 6.2 查询参数的影响（固定已建索引）

- **HNSW `efSearch`** 16→256（`M=8`）：P50 0.221→1.285 ms（5.8×），召回 0.980→1.000（+2 pt）。P95 只比 P50 高 4%–15%，尾延迟稳定 —— 因为 beam search 的工作量由 `efSearch` 而非数据分布决定。
- **IVF-Flat `nprobe`** 1→64（`K=64`）：P50 0.020→0.311 ms（15.5×，几乎严格线性于被扫描的候选数），召回 0.460→1.000。`nprobe` 是"延迟换召回"最干净的旋钮。
- **IVF-PQ `nprobe`** 1→64：召回只从 0.230 涨到 0.280（8 bit 时 0.340→0.430），延迟却从 0.036 涨到 0.867 ms（8 bit 0.044→0.860 ms）。**扩大探查几乎无收益**，说明瓶颈不在候选集，而在量化误差。
- **召回接近时的两类索引对比**（开发集）：recall≈0.98 时 IVF-Flat（`K=64,nprobe=16`）0.103 ms、IVF-Flat（`K=128,nprobe=16`）0.077 ms，而 HNSW（`M=16,efSearch=32`）0.318 ms，差 **3–4 倍**；recall=1.00 时 IVF-Flat 0.126–0.311 ms、HNSW 0.666–0.747 ms，差 2–6 倍。图索引在召回上不占优，延迟却更差。
- **文件命中率不随片段召回单调提升，而且这个指标在本实验里分辨率极低**：开发集 Hit@5 落在 0.300–0.500（10 条查询，**一条就是 0.1**），测试集落在 0.325–0.450（40 条查询，一条 0.025）。也就是说 0.3 与 0.4 的差别只是"少命中一条"，`K=32,nprobe=1` 的 0.500 也只多命中一条；再加上文件级去重把几十个片段的顺序波动平均掉了（见第 7 节），所以 Hit@5 在 0.40–0.45 之间波动并不代表索引好坏，更不能用它来选参数。

### 6.3 离线与在线的时间分布

**离线**（同一份固定向量，因此各组只有建库耗时不同）：

| 阶段 | 耗时 | 占离线总时长 |
|---|---:|---:|
| 整库文档嵌入（8,045 段，4 线程） | 330.41 s | — |
| IVF-Flat 建库（k-means + 分簇，`nlist=64`） | 1.13 s | 嵌入的 0.34% |
| HNSW 建库（`M=16, efC=100`） | 11.53 s | 嵌入的 3.49% |

即：**离线成本几乎全部由嵌入构成**（IVF 路线 99.7%，HNSW 路线 96.6%），把索引从 IVF 换成 HNSW 只让离线总时长多 3.1%。这与在线结论互为镜像 —— 在这个规模上，索引既不决定延迟，也不决定构建成本。

**在线**（测试集固定的 3 条查询，P50，各组独立重启容器后测量）：

| 配置 | 嵌入 | 检索 | 生成 | 完整请求 |
|---|---:|---:|---:|---:|
| exact 全扫 | 57.4 ms | 0.65 ms | 452 ms | 511 ms |
| HNSW `efSearch=32` | 57.1 ms | 0.54 ms | 445 ms | 504 ms |
| IVF-Flat `nprobe=16` | 58.2 ms | 0.20 ms | 473 ms | 532 ms |
| IVF-Flat `nprobe=64` | 76.6 ms | 0.58 ms | 550 ms | 638 ms |
| IVF-PQ 8bit `nprobe=8` | 57.1 ms | 0.26 ms | 655 ms | 713 ms |

- **占比**（以 IVF-Flat `nprobe=16` 为例）：生成 88.9%、查询嵌入 10.9%、ANN 检索 0.04%、LangGraph 框架与文件聚合 0.15%。检索那一段连 1 ms 都不到，把 `nprobe` 从 16 提到 64（候选多 4 倍）只让检索段多 0.38 ms，端到端却被生成侧波动淹没。
- **生成耗时的决定因素**：输出 22/29/37 个 token 分别耗时 339/452/583 ms，斜率约 16.3 ms/token（约 **61 token/s**），截距接近 0。换句话说 `llm_ms` 基本只由**输出 token 数**决定，与检索配置无关。
- **PQ 那一行 655 ms 不是检索变慢**，而是它返回的片段不同，模型多写了 token（8/32/61 对 22/29/37），这正好说明"比较生成耗时要同时列 token 数"的必要性。
- **冷启动是一个独立的、量级更大的成本**：同一个配置在容器刚重启后的第一轮里，前两条查询的 `llm_ms` 是 6.3 s 与 7.2 s，第三条骤降到 0.6 s；第二轮全部落回 0.35–0.7 s。原因是 llama-server 需要 prefill 上千 token 的检索上下文，而 `-c 2048` 的 KV 缓存容量有限，缓存命中与否决定了这一跳。**上表是第二轮（稳态）的数字**；若把冷态算进去，P50 会翻十倍以上。这也意味着"3 条查询取中位数"的协议对生成阶段极不稳定 —— 报告里所有关于生成的结论都只用稳态数据，并同时给出 token 数以便换算。

---

## 7. 为什么 PQ 在这份数据上救不回来（定量分析）

三个数放在一起就清楚了：

1. **排名间隔极小**：开发集上精确 top-10 与第 11 名的平均分差只有 **0.0058**（top-10 内部相邻平均差 0.0086）。也就是说，只要打分扰动超过 0.006，top-10 的成员就会被换掉。
2. **PQ 的打分扰动很大**：`nlist=64, m=16, 8bit` 的重构误差 `‖x − x̂‖ = 0.539`（原始向量模长为 1），对应的内积扰动 `|q·(x − x̂)| ≈ 0.032`，是排名间隔的 **5.5 倍**。
3. **于是召回天花板由量化误差决定**：即使 `nprobe=nlist`（全表扫描），召回也只有 0.43；把 `m` 加到 48（子空间只剩 8 维）才升到 0.68，代价是 889 KB 空间与 1.88 ms 延迟 —— 已经比精确全扫（0.50 ms）慢 3.7 倍，压缩也就失去了意义。

对照之下，IVF-Flat 与 HNSW 的**距离计算是精确的**（前者是原向量点积，后者是图遍历 + 精确点积），误差只来自"候选集是否包含真近邻"，因此召回随候选数单调趋近 1，这正是表中 `nprobe=32/64` 时 recall=1.000 的原因。

**为什么 Hit@5 反而稳定？** 186 个文件被切成 8,045 段，平均每文件 43 段；文件排名只看"该文件最靠前的片段排第几"。PQ 打乱的是片段内部的细微顺序，而每个文件仍会有若干片段落在前 50 里，前 5 个文件的集合基本不变 —— 聚合把噪声吸收了。这也是任务设计"先检索片段、再按文件排名"的鲁棒性来源；但对**片段级**应用（例如直接把片段喂给模型做引用），PQ 在这个数据集上是不可用的。

---

## 8. 结论

1. **目标达成**：只要把 `nprobe` 取到 16 以上、`efSearch` 取到 32 以上，两个索引在开发集与测试集都能达到 recall ≥0.975（多数配置为 0.990–1.000），文件命中率与精确检索持平（0.425）。`nprobe=8` 那几行（0.842–0.900）说明候选集太小时召回会明显下滑，这是 `nprobe` 的直接代价。
2. **规模决定结论**：8,045 条 384 维向量下，精确检索只要 0.50 ms，任何"少扫一点"的收益都被 Python 单次调用开销抵消。IVF-Flat 因为把一次查询压成几次大矩阵乘而最快（0.077 ms @ recall 0.98）；HNSW 受限于上百次小批量调用（0.318 ms）。索引的真正价值要在百万级规模才会显现。
3. **压缩要匹配数据的可分性**：PQ 把索引压到约 1/19（0.64 MB vs 12.0 MB），但本数据的排名间隔（0.006）小于量化误差（0.032），片段级召回跌到 0.43。若必须压缩，`m` 要开到 48 以上才有 0.68，而此时延迟（1.88 ms）已是精确检索的 3.7 倍，压缩失去了意义。
4. **端到端瓶颈在生成**：完整请求 504 ms 中 89% 是 0.5B 模型的解码（≈61 token/s），查询嵌入占 11%，检索不到 0.1%。对这个业务而言，优化索引只能改善 1 ms 量级，优化输出长度或换更快的解码才有意义。
5. **可复现性**：所有随机性（层采样、k-means 初始化）都由固定 seed 控制，同一组参数重复运行得到相同的建库耗时与召回；原始结果全部落在 `results/bench-*.jsonl`，表格与图由脚本从这些文件重新生成，不存在手抄数字。

---

## 9. 复现命令

容器内（首次构建 10–25 分钟；`indexes/` 与 `results/` 为挂载卷，宿主机改代码后无需重建）：

```bash
docker compose up -d --build --wait
docker compose exec lab sh

cd /lab
bench embed > results/bench-embed.jsonl
bench search exact --split dev  > results/bench-exact-dev.jsonl
bench search exact --split test > results/bench-exact-test.jsonl

bench search graph --split dev \
  --build M=8,efConstruction=100 --build M=16,efConstruction=100 --build M=16,efConstruction=200 \
  --search efSearch=16 --search efSearch=32 --search efSearch=64 --search efSearch=128 --search efSearch=256 \
  > results/bench-graph-dev.jsonl

bench search cluster --split dev \
  --build nlist=32,nbits=0 --build nlist=64,nbits=0 --build nlist=128,nbits=0 \
  --search nprobe=1 --search nprobe=2 --search nprobe=4 --search nprobe=8 \
  --search nprobe=16 --search nprobe=32 --search nprobe=64 --search nprobe=128 \
  > results/bench-cluster-flat-dev.jsonl

bench search cluster --split dev \
  --build nlist=64,nbits=4,m=16 --build nlist=64,nbits=8,m=16 --build nlist=64,nbits=8,m=48 \
  --search nprobe=1 --search nprobe=4 --search nprobe=8 --search nprobe=16 --search nprobe=64 \
  > results/bench-cluster-pq-dev.jsonl

bench search graph   --split test --build M=8,efConstruction=100 --build M=16,efConstruction=100 \
  --search efSearch=32 --search efSearch=128 > results/bench-graph-test.jsonl
bench search cluster --split test --build nlist=64,nbits=0 --build nlist=128,nbits=0 \
  --search nprobe=8 --search nprobe=16 --search nprobe=32 --search nprobe=64 > results/bench-cluster-flat-test.jsonl
bench search cluster --split test --build nlist=64,nbits=8,m=16 \
  --search nprobe=8 --search nprobe=64 > results/bench-cluster-pq-test.jsonl

bench online exact   --split test > results/bench-online-exact.jsonl
bench online graph   --split test --build M=16,efConstruction=100 \
  --search efSearch=32 --search efSearch=128 > results/bench-online-graph.jsonl
bench online cluster --split test --build nlist=64,nbits=0 \
  --search nprobe=16 --search nprobe=64 > results/bench-online-cluster-flat.jsonl
bench online cluster --split test --build nlist=64,nbits=8,m=16 \
  --search nprobe=8 --search nprobe=64 > results/bench-online-cluster-pq.jsonl

exit
docker compose down
```

宿主机（重新生成表格与图；表格只读 `results/bench-*.jsonl`，与容器实测同源）：

```bash
python tools/make_report.py
```

**本机网络的附加步骤**：若 `docker compose up --build` 在 apt 阶段报 `Connection failed`（`deb.debian.org` 不可达）或在 pip 阶段 `Read timed out`，说明构建源被限速。此时按 §1 的三处替换用仓库外副本构建，再把产物打上课程镜像名，compose 会直接复用现成镜像而不重新构建：

```bash
docker build -f /path/to/Dockerfile.local -t cpu-file-retrieval-lab:2.0 .
docker compose up -d --wait     # 有同名镜像时不会再 build
```

本次构建实测：完整构建约 11 分钟（apt 140 s、llama.cpp 编译约 4 分钟、依赖与模型下载约 5 分钟），镜像 2.6 GB。

---

## 附录：提交物清单

| 文件 | 说明 |
|---|---|
| `indexes/graph.py` | HNSW 实现（分层构建、邻居启发式、beam search） |
| `indexes/cluster.py` | IVF 实现（k-means 倒排表 + IVF-Flat / IVF-PQ 两种量化模式） |
| `results/bench-*.jsonl` | 容器内 `bench` 的原始输出（`type:"summary"` 为汇总，`type:"query"` 为逐条明细） |
| `results/bench-index-size.jsonl` | 各配置索引额外空间（容器内调用 `size_bytes()` 单独测得） |
| `results/report-tables.md` | 由脚本从原始结果生成的表格（本报告第 5 节与之同步） |
| `results/report-*.png` | 报告用图（延迟–召回、HNSW 放大、测试集对比、在线拆解） |
| `results/host-archive/` | 装 Docker 之前用 `tools/offline_bench.py` 在宿主机上跑的第一轮结果（口径校准用：exact 基线同样得到 dev 0.400 / test 0.425），报告中的数值不取自此目录 |
| `report.md` | 本报告 |
| `tools/offline_bench.py` | 辅助脚本：宿主机上复现 `bench search` 口径（本报告未使用其数值） |
| `tools/make_report.py` | 从 `results/bench-*.jsonl` 重新生成表格与图 |
