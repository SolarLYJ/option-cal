# -*- coding: utf-8 -*-
# @author: Li Yijia
# @date: 2025/08/04
# @filename: vanilla.py

"""
==================================================================
Vanilla期权定价器(欧式 & 美式)
------------------------------------------------------------------
支持欧式和美式期权
采取BS + CRR两种定价方法, 其中BS仅限于欧式, CRR支持欧式和美式
输出：
    price, delta, gamma, vega, theta, rho
==================================================================
"""
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum
from math import exp, sqrt, log
from .utils import bump_parameter


# ---------- 数据结构 -------------------------------------------------------
class OptType(Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(Enum):
    EUROPEAN =  "european"
    AMERICAN = "american"

@dataclass
class OptionSpec:
    S0: float # spot price
    K:  float # strike price
    TTM:  float # time to maturity (in years)
    r:  float # risk-free interest rate
    q:  float # dividend yield
    sigma: float # volatility
    opt_type: OptType # option type
    style: ExerciseStyle # exercise style

# ---------- Black-Scholes --------------------------------------------------
class BlackScholes:
    """
    欧式期权 Black-Scholes 闭式定价器
    ------------------------------------------------------------
    输出单位约定
        price  : 货币单位
        delta  : 无单位
        gamma  : 1/价格
        vega   : 每 1% 波动率变动导致的价格变化
        theta  : 每日时间流逝导致的价格变化
        rho    : 每 1% 无风险利率变动导致的价格变化
    """

    def __init__(self, spec: "OptionSpec"):
        # 查期权是否为欧式，若非则报错
        if spec.style != ExerciseStyle.EUROPEAN:
            raise ValueError(f"Black-Scholes模型仅适用于欧式期权，当前期权类型为{spec.style.value}")
        self.p = spec
        self._precompute()      # 预先把 d1 d2 等常用量算好

    # -------------------------------------------------------- #
    # 内部
    # -------------------------------------------------------- #
    def _precompute(self):
        """预计算 d1, d2、贴现因子等，后续 Greek 调用"""
        p = self.p
        self._sqrtT = sqrt(p.TTM)

        self.d1 = (log(p.S0 / p.K) +
                   (p.r - p.q + 0.5 * p.sigma ** 2) * p.TTM) / (p.sigma * self._sqrtT)
        self.d2 = self.d1 - p.sigma * self._sqrtT

        # φ(d1)  ϕ(d1) PDF
        self._pdf_d1 = norm.pdf(self.d1)
        # Φ(d1)   CDF
        self._cdf_d1 = norm.cdf(self.d1)
        self._cdf_d2 = norm.cdf(self.d2)
        self._cdf_nd1 = norm.cdf(-self.d1)
        self._cdf_nd2 = norm.cdf(-self.d2)

        # 折现因子
        self._disc_r = exp(-p.r * p.TTM)
        self._disc_q = exp(-p.q * p.TTM)

    # -------------------------------------------------------- #
    # 价格
    # -------------------------------------------------------- #
    def price(self) -> float:
        p = self.p
        if p.opt_type == OptType.CALL:          # 看涨
            return (p.S0 * self._disc_q * self._cdf_d1 -
                    p.K  * self._disc_r * self._cdf_d2)
        else:                                   # 看跌
            return (p.K  * self._disc_r * self._cdf_nd2 -
                    p.S0 * self._disc_q * self._cdf_nd1)

    # -------------------------------------------------------- #
    # Greeks
    # -------------------------------------------------------- #
    def delta(self) -> float:
        """∂V / ∂S"""
        p = self.p
        sign = 1 if p.opt_type == OptType.CALL else -1    # CALL:+ , PUT:-
        return sign * self._disc_q * norm.cdf(sign * self.d1)

    def gamma(self) -> float:
        """∂²V / ∂S²"""
        p = self.p
        return (self._disc_q * self._pdf_d1) / (p.S0 * p.sigma * self._sqrtT)

    def vega(self) -> float:
        """∂V / ∂σ   —— 结果已按 1% 波动率变动（除以 100）"""
        p = self.p
        return p.S0 * self._disc_q * self._pdf_d1 * self._sqrtT / 100

    def theta(self) -> float:
        """∂V / ∂t   —— 按“每日”给出"""
        p = self.p
        # 第一项：时间衰减
        term1 = - (p.S0 * self._disc_q * self._pdf_d1 * p.sigma) / (2 * self._sqrtT)
        # 利息 & 分红项
        if p.opt_type == OptType.CALL:
            term2 =  p.q * p.S0 * self._disc_q * self._cdf_d1
            term3 = -p.r * p.K  * self._disc_r * self._cdf_d2
        else:
            term2 = -p.q * p.S0 * self._disc_q * self._cdf_nd1
            term3 =  p.r * p.K  * self._disc_r * self._cdf_nd2
        return (term1 + term2 + term3) / 365.0     # 每日 theta

    def rho(self) -> float:
        """∂V / ∂r   —— 按 1% 利率变动（除以 100）"""
        p = self.p
        if p.opt_type == OptType.CALL:
            return  p.K * p.TTM * self._disc_r * self._cdf_d2  / 100
        else:
            return -p.K * p.TTM * self._disc_r * self._cdf_nd2 / 100

    # -------------------------------------------------------- #
    # 打包输出
    # -------------------------------------------------------- #
    def all_greeks(self):
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
        }

# ---------- CRR 二叉树 ----------------------------------------------
"""
CRR 二叉树定价器（欧式 + 美式）
----------------------------------------------------------------
• price   价格
• delta   ∂V/∂S
• gamma   ∂²V/∂S²
• theta   ∂V/∂t      （日）
• vega    ∂V/∂σ      （对 1% 波动率变动）
• rho     ∂V/∂r      （对 1% 无风险利率变动）
----------------------------------------------------------------
"""

@dataclass
class CRRBinomial:
    spec : OptionSpec
    steps: int = 200          # 网格步数，越大精度越高

    # --------------------- 主 -----------------------------
    def price(self) -> float:
        self._build_lattice()
        return self._V0

    def delta(self) -> float:
        if not hasattr(self, "_built"):
            self._build_lattice()
        V_up, V_dn = self._V1_up, self._V1_dn
        S_up, S_dn = self._S1_up, self._S1_dn
        # Delta = 期权价值变化 / 标的价格变化
        return (V_up - V_dn) / (S_up - S_dn)

    def gamma(self) -> float:
        if not hasattr(self, "_built"):
            self._build_lattice()

        # 二阶差分：使用第二层 3 个节点的期权价值
        V_uu, V_ud, V_dd = self._V2_uu, self._V2_ud, self._V2_dd
        S_uu, S_ud, S_dd = self._S2_uu, self._S2_ud, self._S2_dd

        delta_up   = (V_uu - V_ud) / (S_uu - S_ud)
        delta_dn   = (V_ud - V_dd) / (S_ud - S_dd)
        # Gamma = Delta变化 / 标的价格中点变化
        return (delta_up - delta_dn) / ((S_uu - S_dd) / 2)

    def theta(self) -> float:
        if not hasattr(self, "_built"):
            self._build_lattice()
        V_now  = self._V0
        V_next = (self._prob * self._V1_up + (1 - self._prob) *
                  self._V1_dn) * self._disc
        dt = self.spec.TTM / self.steps
        return (V_next - V_now) / dt / 365     # 按每日
    
    def vega(self) -> float:
        """有限差分：σ 上移 1bp (=0.0001) 后重算价格，再除以 1%"""
        bump = 1e-4
        spec_up = bump_parameter(self.spec,"sigma", +bump)
        price_up = CRRBinomial(spec_up, self.steps).price()
        return (price_up - self.price()) / bump / 100  # 对 1%

    def rho(self) -> float:
        """有限差分：r 上移 1bp"""
        bump = 1e-4
        spec_up = bump_parameter(self.spec,"sigma", +bump)
        spec_up = self.spec.__class__(**{**self.spec.__dict__, "r": self.spec.r + bump})
        price_up = CRRBinomial(spec_up, self.steps).price()
        return (price_up - self.price()) / bump / 100

    # -------------------- 内部 ----------------------------

    def _build_lattice(self):
        p = self.spec
        N = self.steps
        dt = p.TTM / N
        u  = exp(p.sigma * sqrt(dt))
        d  = 1 / u
        self._prob = (exp((p.r - p.q) * dt) - d) / (u - d)   # 风险中性概率
        self._disc = exp(-p.r * dt) # 单步折现因子（按无风险利率折现）

        # -------- 末端资产价格 & 期权价值 ------------------
        # S_N(j) = S0 * u^{N-j} d^{j}, j=0..N
        j = np.arange(N + 1)
        S = p.S0 * u ** (N - j) * d ** j
        if p.opt_type is OptType.CALL:
            V = np.maximum(S - p.K, 0.0)
        else:
            V = np.maximum(p.K - S, 0.0)

        # -------- 回溯 ------------------
        for step in range(N, 0, -1):
            # 向上一层折现
            V = self._disc * (self._prob * V[:-1] + (1 - self._prob) * V[1:])

            # 若美式则比较立即行权价值
            if p.style is ExerciseStyle.AMERICAN:
                S = S[:-1] / u                      # 先得到当前层资产价格
                if p.opt_type is OptType.CALL:
                    V = np.maximum(V, S - p.K)
                else:
                    V = np.maximum(V, p.K - S)
            else:
                S = S[:-1] / u

            # 记录第 1、2 层节点用于 Greeks
            if step == 1:          # 根节点
                self._V0 = V[0]
            elif step == 2:
                # 获取1步后上涨/下跌的期权价值和标的价格
                self._V1_up, self._V1_dn = V[0], V[1]
                self._S1_up, self._S1_dn = S[0], S[1]
            elif step == 3:
                # 获取2步后三个节点的期权价值和标的价格
                self._V2_uu, self._V2_ud, self._V2_dd = V[0], V[1], V[2]
                self._S2_uu, self._S2_ud, self._S2_dd = S[0], S[1], S[2]

        self._built = True

    # -------------------- 汇总输出 ------------------------
    def all_greeks(self):
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
        }
    
# ----------------------------------------------------------------
# 3. DEMO
# ----------------------------------------------------------------
if __name__ == "__main__":
    spec = OptionSpec(
        S0=100, 
        K=100, 
        TTM=1, 
        r=0.03,
        q=0.02, 
        sigma=0.25,
        opt_type = OptType.CALL,
        style = ExerciseStyle.EUROPEAN)
    bs = BlackScholes(spec)
    res = bs.all_greeks()
    print("看涨欧式期权价格及希腊字母计算结果：")
    for key, value in res.items():
        print(f"{key}: {value:.4f}")

    spec = OptionSpec(
        S0=100, 
        K=100, 
        TTM=1, 
        r=0.03,
        q=0.02, 
        sigma=0.25,
        opt_type=OptType.CALL,
        style=ExerciseStyle.AMERICAN)
    crr = CRRBinomial(spec, steps=200)
    res = crr.all_greeks()
    print("看跌美式期权价格及希腊字母计算结果：")
    for key, value in res.items():
        print(f"{key}: {value:.4f}")