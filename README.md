# DOE Bayesian Optimisation Solver

基于贝叶斯优化的多因素工业过程实验设计（DOE）求解器。
以最少实验次数逼近最优参数组合。

## 算法原理

- **初始设计**：Latin Hypercube Sampling (LHS) 生成初始实验点
- **代理模型**：Gaussian Process with Matérn-5/2 核
- **采集函数**：Expected Improvement (EI)
- **策略**：每轮选择 EI 最大的参数组合进行实验，迭代逼近最优

## 模拟工业过程

内置 4 因素化学反应收率模型：

| 因素       | 范围          | 单位 |
|-----------|---------------|------|
| 温度       | 50 – 200      | °C   |
| 压力       | 1 – 10        | atm  |
| 反应时间    | 10 – 120      | min  |
| 催化剂浓度  | 0.1 – 5.0     | %    |

响应面为含交互效应的二阶多项式 + 高斯噪声 (σ = 1.5%)。

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 默认交互模式（手动输入实验结果）
python doe_optimizer.py

# 自定义参数
python doe_optimizer.py --budget 50 --seed 123

# 模拟模式（内置模拟器）
python doe_optimizer.py --simulate

# 导出历史到 CSV
python doe_optimizer.py --output history.csv
```

### Docker 运行

```bash
# 构建镜像
docker build -t doe-optimizer .

# 默认运行
docker run --rm doe-optimizer

# 自定义参数
docker run --rm doe-optimizer --budget 50 --seed 123

# 导出结果（挂载卷）
docker run --rm -v "$PWD/output:/app/output" doe-optimizer --output /app/output/history.csv
```

## 输出示例

```
============================================================
  DOE Bayesian Optimisation Solver
============================================================
  Budget      : 30 experiments
  Mode        : simulation
  Seed        : 42
  Factors     : temperature, pressure, time, catalyst
------------------------------------------------------------
  True optimum yield (noiseless): 87.42 %
------------------------------------------------------------
  [LHS] iter   1  |  temperature=168.75, pressure=4.12, ... |  yield=78.234 %
  [LHS] iter   2  |  temperature=95.30, pressure=8.67, ...  |  yield=81.056 %
  ...
  [EI ] iter  15  |  temperature=137.42, pressure=3.21, ... |  yield=87.103 %
============================================================
  OPTIMISATION COMPLETE
============================================================
  Best observed yield : 87.4512 %
  True optimum yield  : 87.4200 %
  Gap                 : +0.0312 %
  Total experiments   : 30

  Best parameters found:
     temperature:   137.420  (true:   137.400,  range: 50.0–200.0)
        pressure:     3.210  (true:     3.200,  range: 1.0–10.0)
            time:    89.560  (true:    89.500,  range: 10.0–120.0)
        catalyst:     1.870  (true:     1.860,  range: 0.1–5.0)
```

## 依赖

- Python ≥ 3.10
- numpy, scipy, scikit-learn, pandas, pyDOE2

## 文件结构

```
├── doe_optimizer.py      # CLI 入口
├── process_simulator.py  # 工业过程模拟器
├── bayesian_opt.py       # 贝叶斯优化引擎
├── requirements.txt      # Python 依赖
├── Dockerfile            # 容器化
└── README.md             # 说明文档
```
