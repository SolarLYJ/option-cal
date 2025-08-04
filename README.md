
# 欧式、美式期权 / 亚式期权 / 雪球票据定价与希腊字母批量计算

>可对欧式 / 美式香草期权（基于 Black-Scholes 模型和 CRR 二叉树模型）、亚式期权（基于控制变量蒙特卡洛）、自动赎回型（保本及非保本）雪球票据（基于对立路径的蒙特卡洛模拟）进行定价。通过main.py读取CSV 文件，并将价格及完整希腊字母集导出至results/*.csv文件中

## 目录结构

option-cal/
 ├─ pricers/              # 定价核心代码
 │  ├─ vanilla.py         # BlackScholes & CRR，欧式、美式期权定价
 │  ├─ asian.py           # 亚式期权定价期
 │  ├─ utils.py           # 小工具
 │  └─ snowball.py        # 雪球定价（MC）
 ├─ data/                 # 数据存放文件夹
 │  ├─ vanillas.csv
 │  ├─ asians.csv
 │  └─ snowballs.csv
 ├─ results/              # output (auto-created)
 ├─ main.py               # 批量定价入口
 └─ README.md

- pricers中存放定价器核心代码
- 运行后会生成results文件夹，存放生成结果csv

---

## 数学与算法速览

| 产品 | 方法 | 备注 |
|------|------|------|
| 欧式期权 | Black–Scholes closed-form / CRR binomial tree |  |
| 美式期权 | Cox-Ross-Rubinstein (CRR) binomial tree |  |
| 亚式期权 | Control-Variate GBM Monte-Carlo | Antithetic variance-reduction |
| 自动赎回型雪球票据 | GBM Monte-Carlo | Antithetic variance-reduction |

希腊字母均由 **bump-and-revalue** 有限差分计算  

------

## CSV 数据格式

`data/vanillas.csv`

| 列名                 | 说明                    |
| -------------------- | ----------------------- |
| id                   | 唯一标识                |
| S0, K, TTM, q, sigma | 市场参数                |
| opt_type             | `CALL` / `PUT`          |
| style                | `EUROPEAN` / `AMERICAN` |

`data/asians.csv`

| 列名                           | 说明                               |
| ------------------------------ | ---------------------------------- |
| id                             | 唯一标识                           |
| S0, K, TTM, q, sigma，opt_type，style  | 如上                               |
| n_obs                       | 观测次数 (包括起始点)       |

`data/snowballs.csv`

| 列名                           | 说明                               |
| ------------------------------ | ---------------------------------- |
| id                             | 唯一标识                           |
| note_type                      | `PROTECTED` / `UNPROTECTED`        |
| notional,S0,strike,q,sigma,TTM | 如上                               |
| KO_ratio                       | KO 障碍 (S ≥ KO → Auto-Call)       |
| KI_ratio                       | KI 障碍 (S ≤ KI → 敲入)            |
| coupon                         | 年化票息                           |
| obs_freq                       | KO 观察频率：月/季/日['M','Q','D'] |

`results/snowballs_pricing.csv`和`results/vanillas_pricing.csv`和`results/asians_pricing.csv`

| 列名    | 说明     |
| ------- | -------- |
| id      | 唯一标识 |
| price   | 定价     |
| delta   |          |
| gamma   |          |
| vega    |          |
| thetath |          |
| rho     |          |

------

## 一键定价
python main.py


输出示例

=== Vanilla Options ===
id price  delta  gamma  vega   theta   rho
v00001  10.45  0.562  0.013  0.343 -0.020  0.410
v00002     7.42 -0.438  0.013  0.343 -0.012 -0.410
v00003   10.45  0.562  0.013  0.343 -0.020  0.410
v00004    8.23 -0.421  0.014  0.344 -0.014 -0.415

=== Snowball Notes ===
id price  delta   gamma  vega  theta   rho
s00001   10119.47  0.368  0.0021 0.291 -0.0049 0.485
s00002  9884.27  0.426  0.0023 0.318 -0.0062 0.510

=== Asian Options ===
id price  delta   gamma  vega  theta   rho
a00001  7.59  0.61  0.048  0.21  0.012  0.71
a00002  3.40 -0.33  0.054  0.21  0.0015 -0.635


对应结果已写入

results/vanillas_pricing.csv
results/snowballs_pricing.csv
results/asians_pricing.csv

## 具体说明

pricers包含三个核心模块：

- vanilla.py：提供香草期权定价功能，包括 Black-Scholes 模型和 CRR 二叉树模型
- snowball.py：提供雪球结构化产品（Snowball Note）的蒙特卡洛定价功能
- asians.py：提供亚式期权的基于控制变量的蒙特卡洛定价功能
- utils.py：提供通用工具函数，如参数扰动、随机数生成等

### 数据结构定义

#### 期权相关枚举类

```python
# 期权类型
class OptType(Enum):
    CALL = "call"  # 看涨期权
    PUT = "put"    # 看跌期权

# 行权方式
class ExerciseStyle(Enum):
    EUROPEAN = "european"  # 欧式期权（到期行权）
    AMERICAN = "american"  # 美式期权（到期前任意时间行权）
```

#### 期权规格类

用于定义期权的基本参数，数据类（dataclass）结构如下：

```python
@dataclass
class OptionSpec:
    S0: float          # 标的资产当前价格
    K: float           # 行权价格
    TTM: float         # 到期时间（年）
    r: float           # 无风险利率
    q: float           # 股息率
    sigma: float       # 波动率
    opt_type: OptType  # 期权类型（看涨/看跌）
    style: ExerciseStyle  # 行权方式（欧式/美式）
```

### 雪球产品相关定义

```python
# 雪球产品类型
class SnowballType(Enum):
    PROTECTED = "protected"    # 保本型
    UNPROTECTED = "unprotected"  # 非保本型

# 雪球产品规格
@dataclass
class SnowballSpec:
    note_type: SnowballType       # 产品类型
    notional: float               # 名义本金
    S0: float                     # 标的初始价格
    strike: float                 # 行权价
    r: float                      # 无风险利率
    q: float                      # 股息率
    sigma: float                  # 波动率
    KO_ratio: float               # 敲出比例
    KI_ratio: float               # 敲入比例
    coupon: float                 # 年化票息率
    TTM: float                    # 期限（年）
    obs_freq: Literal['M','Q','D']  # 敲出观察频率（月/季/日）
```

### 定价模型使用说明

#### 1. Black-Scholes 模型（欧式期权）

##### 功能特点

- 仅适用于欧式期权
- 提供解析解计算价格及希腊字母
- 计算速度快，精度高

##### 使用示例

```python
from vanilla import OptionSpec, OptType, ExerciseStyle, BlackScholes

# 定义期权参数
spec = OptionSpec(
    S0=100,          # 标的价格
    K=100,           # 行权价
    TTM=1,           # 1年到期
    r=0.03,          # 无风险利率3%
    q=0.02,          # 股息率2%
    sigma=0.25,      # 波动率25%
    opt_type=OptType.CALL,  # 看涨期权
    style=ExerciseStyle.EUROPEAN  # 欧式
)

# 创建定价器实例
bs_pricer = BlackScholes(spec)

# 计算价格
price = bs_pricer.price()

# 计算所有希腊字母
greeks = bs_pricer.all_greeks()
print("价格:", greeks["price"])
print("Delta:", greeks["delta"])
print("Gamma:", greeks["gamma"])
print("Vega:", greeks["vega"])
print("Theta:", greeks["theta"])
print("Rho:", greeks["rho"])
```

#### 2. CRR 二叉树模型（欧式 / 美式期权）

##### 功能特点

- 支持欧式和美式期权定价
- 通过二叉树数值方法计算
- 可通过调整步数（steps）平衡精度和速度

##### 使用示例

```python
from vanilla import CRRBinomial

# 使用之前定义的OptionSpec
# 对于美式期权，只需将style设为ExerciseStyle.AMERICAN

spec_american = OptionSpec(
    S0=100,
    K=100,
    TTM=1,
    r=0.03,
    q=0.02,
    sigma=0.25,
    opt_type=OptType.PUT,
    style=ExerciseStyle.AMERICAN
)

# 创建二叉树定价器，指定步数为200
crr_pricer = CRRBinomial(spec_american, steps=200)

# 计算价格和希腊字母
price = crr_pricer.price()
greeks = crr_pricer.all_greeks()
```

#### 3. 雪球产品定价（蒙特卡洛模拟）

##### 功能特点

- 支持保本型和非保本型雪球产品
- 支持不同敲出观察频率（月 / 季 / 日）
- 使用蒙特卡洛模拟（GBM 模型）定价
- 内置对立路径（Antithetic）降低方差
- 观察方式：每日连续监控

##### 现金流结算逻辑

| 场景                       | 结算规则                                                |
| -------------------------- | ------------------------------------------------------- |
| 存续期内敲出（KO）         | 本金 + 票息 × 持有时间（从生效到敲出的实际时间）        |
| 未敲出且未敲入             | 到期返还本金 + 票息 × 产品期限（全额票息）              |
| 未敲出但已敲入（保本型）   | 到期返还全额本金（无论标的最终价格）                    |
| 未敲出但已敲入（非保本型） | 到期按标的最终价格结算：本金 × (ST /strike)（可能亏损） |

##### 使用示例

```python
from snowball import SnowballSpec, SnowballType, SnowballMC

# 定义雪球产品参数
spec = SnowballSpec(
    note_type=SnowballType.PROTECTED,  # 保本型
    notional=100000,                   # 名义本金10万元
    S0=100,                            # 标的初始价格
    strike=100,                        # 行权价
    KO_ratio=1.05,                     # 敲出比例105%
    KI_ratio=0.70,                     # 敲入比例70%
    coupon=0.12,                       # 年化票息12%
    TTM=1.0,                           # 期限1年
    r=0.03,                            # 无风险利率3%
    q=0.02,                            # 股息率2%
    sigma=0.25,                        # 波动率25%
    obs_freq="M",                      # 月度观察敲出
)

# 创建蒙特卡洛定价器，指定模拟路径数
pricer = SnowballMC(spec, n_paths=131072)

# 计算价格和希腊字母
res = pricer.all_greeks()
for k, v in res.items():
    print(f"{k}: {v:.4f}")
```

#### 4. 亚式期权定价（基于控制变量的蒙特卡洛模拟）

##### 功能特点

- 实现了基于控制变量蒙特卡洛（Control-Variate Monte-Carlo）方法的算术平均亚式期权定价器
- 支持欧式到期行权方式，并能计算期权的价格及主要希腊字母（delta、gamma、vega、theta、rho）
- 控制变量法通过引入几何平均亚式期权（具有解析解）作为控制变量，有效降低了蒙特卡洛模拟的误差。
##### 核心公式
定价核心公式如下：
V_A = V_A_MC + β*(V_G_analytic - V_G_MC)

其中：

- V_A：算术平均亚式期权的定价结果
- V_A_MC：算术平均亚式期权的蒙特卡洛模拟结果
- V_G_analytic：几何平均亚式期权的解析解（Kemna & Vorst 1990）
- V_G_MC：几何平均亚式期权的蒙特卡洛模拟结果
- β：协方差系数，β = Cov(A,G)/Var(G)

##### 实现细节

1. **随机路径生成**：
   - 使用正态分布生成随机增量
   - 支持对偶变量法（antithetic variates）以降低方差
   - 路径生成基于几何布朗运动模型
2. **均值计算**：
   - 算术平均（A）：路径价格的简单平均值
   - 几何平均（G）：路径价格的几何平均值
3. **控制变量调整**：
   - 计算算术平均和几何平均收益的协方差
   - 利用几何平均的解析解对算术平均结果进行调整
4. **希腊字母计算**：
   - 采用中心有限差分法
   - 价格扰动为 1 个基点（1e-4）
   - Theta 计算基于 1 个交易日（1/252 年）的变化

##### 几何平均解析解

几何平均亚式期权的解析解基于 Kemna & Vorst（1990）的研究，核心参数计算如下：

```plaintext
sigma_g = sigma * sqrt((2 * N + 1) / (6 * N))
mu_g = 0.5 * (r - q - 0.5 * sigma²) + 0.5 * sigma_g²
```

其中 N 为观测次数，在此基础上使用类似 Black-Scholes 的公式计算价格。

##### 使用示例

```python
# 1. 定义期权参数
spec = AsianSpec(
    S0=100.0,
    K=100.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    TTM=1.0,
    opt_type=OptType.CALL,
    n_obs=252  # 每日观测（1年）
)

# 2. 创建定价器
pricer = AsianMC(spec=spec, n_paths=100000, seed=1234)

# 3. 计算价格和希腊字母
price = pricer.price()
delta = pricer.delta()
all_values = pricer.all_greeks()

print(f"价格: {price:.4f}")
print(f"Delta: {delta:.4f}")
print("所有指标:", {k: f"{v:.4f}" for k, v in all_values.items()})
```

### 希腊字母说明

所有模型均提供以下风险指标（希腊字母）：

| 希腊字母 | 含义                                 | 单位          |
| -------- | ------------------------------------ | ------------- |
| price    | 衍生品价格                           | 货币单位      |
| delta    | 标的价格变动对衍生品价格的影响       | 无单位        |
| gamma    | 标的价格变动对 delta 的影响          | 1 / 价格      |
| vega     | 波动率变动 1% 对衍生品价格的影响     | 货币单位      |
| theta    | 每日时间流逝对衍生品价格的影响       | 货币单位 / 日 |
| rho      | 无风险利率变动 1% 对衍生品价格的影响 | 货币单位      |

### 工具函数说明

utils.py提供两个主要工具函数：

1. `bump_parameter(obj: dataclass, field: str, amount: float) -> dataclass`
   - 功能：生成参数扰动后的新数据类对象
   - 参数：
     - `obj`：原始数据类对象
     - `field`：要扰动的字段名
     - `amount`：扰动幅度
   - 用途：用于通过有限差分法计算希腊字母
2. `generate_normal_matrix(n_paths: int, dim: int, seed) -> np.ndarray`
   - 功能：生成指定维度的标准正态分布随机矩阵
   - 参数：
     - `n_paths`：路径数量
     - `dim`：维度（步数）
     - `seed`：随机数种子
   - 用途：为蒙特卡洛模拟提供随机数

### 注意事项

1. Black-Scholes 模型仅适用于欧式期权，对美式期权会抛出异常
2. 在此代码中欧式期权可以使用CRR二叉树模型也可以使用Black-Scholes 模型，但需要在main.py中调整
3. 雪球产品的定价结果包含模拟误差，可通过返回的`stderr`查看标准误差
4. 亚式期权的定价结果包含模拟误差，可通过返回的`stderr`查看标准误差
