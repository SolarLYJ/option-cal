# -*- coding: utf-8 -*-
# @author: Li Yijia
# @date: 2025/08/04
# @filename: snowball.py

"""
==================================================================
Autocallable Snowball Note 票据定价器  (欧系结构化)
------------------------------------------------------------------
支持保本 (Protected) / 非保本 (Unprotected)
Auto-Call Knock-Out 观测频率：月(M)、季(Q)、日(D)
连续 Knock-In 监控（日频近似）
蒙特卡洛 (GBM)  + Antithetic
输出：
    price, delta, gamma, vega, theta, rho
==================================================================
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Literal, Tuple

from .utils import bump_parameter, generate_normal_matrix

# ================================================================
# 1. 基础数据结构
# ================================================================
class SnowballType(Enum):
    PROTECTED   = "protected"     # 保本雪球   – KI 后到期至少返还本金
    UNPROTECTED = "unprotected"     # 非保本雪球 – KI 后按标的表现结算

@dataclass
class SnowballSpec:
    note_type : SnowballType                # 产品类型
    notional  : float                       # 名义本金
    # ---- 标的 & 市场假设 ---------------------------------------
    S0        : float        
    strike    : float        
    r         : float                       # 无风险利率
    q         : float                       # 分红率
    sigma     : float                       # 年化波动率
    # ---- 结构参数 ---------------------------------------------
    KO_ratio  : float                       # KO 障碍 (S ≥ KO → Auto-Call)
    KI_ratio  : float                       # KI 障碍 (S ≤ KI → 敲入)
    coupon    : float                       # 年化票息
    TTM         : float                     # 期限 (年)
    obs_freq  : Literal['M','Q','D']        # KO 观察频率：月/季/日

# ================================================================
# 2. Monte-Carlo 定价类
# ================================================================
@dataclass
class SnowballMC:
    spec: SnowballSpec
    n_paths: int
    seed: int = 666
    antithetic: bool = True  # 是否使用对立路径（降低方差）
    # 缓存（避免重复计算）
    _cache: dict[str, float] = None
    def __post_init__(self):
        """初始化后执行的方法，确保_cache"""
        if self._cache is None:
            self._cache = {}  # 确保_cache_cache初始字典为空字典而不是None

    # ================================================================
    # 主接口：价格 + 希腊字母
    # ================================================================
    def price(self) -> float:
        if "price" not in self._cache:
            self._cache["price"], self._cache["stderr"] = self._price_core(self.spec)
        return self._cache["price"]

    def delta(self) -> float:
        if "delta" not in self._cache:
            bump = self.spec.S0 * 1e-4                     # 1 bp spot
            price_up = self._price_core(bump_parameter(self.spec,"S0", +bump))[0]
            price_dn = self._price_core(bump_parameter(self.spec,"S0", -bump))[0]
            self._cache["delta"] = (price_up - price_dn) / (2 * bump)
        return self._cache["delta"]

    def gamma(self) -> float:
        if "gamma" not in self._cache:
            bump = self.spec.S0 * 1e-4
            price_up = self._price_core(bump_parameter(self.spec,"S0", +bump))[0]
            price_dn = self._price_core(bump_parameter(self.spec,"S0", -bump))[0]
            price0   = self.price()
            self._cache["gamma"] = (price_up - 2 * price0 + price_dn) / (bump ** 2)
        return self._cache["gamma"]

    def vega(self) -> float:
        if "vega" not in self._cache:
            bump = 1e-4                                    # 1 bp vol
            price_up = self._price_core(bump_parameter(self.spec,"sigma", +bump))[0]
            price_dn = self._price_core(bump_parameter(self.spec,"sigma", -bump))[0]
            vega_bp  = (price_up - price_dn) / (2 * bump)
            self._cache["vega"] = vega_bp / 100            # 换算到 1 %
        return self._cache["vega"]

    def theta(self) -> float:
        if "theta" not in self._cache:
            dT = 1 / 252                                   # 1 个交易日
            if self.spec.TTM <= dT:
                raise ValueError("TTM 太短无法计算 Theta")
            price_dn = self._price_core(bump_parameter(self.spec,"TTM", -dT))[0]    # 时间向前
            theta_yr = (price_dn - self.price()) / (-dT)            # ∂V/∂T
            self._cache["theta"] = theta_yr / 365                    # 每日
        return self._cache["theta"]

    def rho(self) -> float:
        if "rho" not in self._cache:
            bump = 1e-4
            price_up = self._price_core(bump_parameter(self.spec,"r", +bump))[0]
            price_dn = self._price_core(bump_parameter(self.spec,"r", -bump))[0]
            rho_bp   = (price_up - price_dn) / (2 * bump)
            self._cache["rho"] = rho_bp / 100
        return self._cache["rho"]

    def all_greeks(self):
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega" : self.vega(),
            "theta": self.theta(),
            "rho"  : self.rho(),
        }

    # ================================================================
    # 内部：Monte-Carlo 主流程
    # ================================================================
    def _price_core(self, spec: SnowballSpec) -> Tuple[float, float]:
        """
        返回 (price, stderr)  ;   其余希腊字母统一由外层 bump-and-revalue 获得
        """
        # ----------- 时间网格 ------------------------------------
        dt_sim = 1 / 252 # 模拟时间步长（1个交易日，单位：年）
        n_steps = ceil(spec.TTM / dt_sim) # 总模拟步数

        # KO 观察索引
        if spec.obs_freq.upper() == "M":
            ko_times = 12
            dt_ko = spec.TTM / ko_times # 每次观察间隔（年）
        elif spec.obs_freq.upper() == "Q":
            ko_times = 4
            dt_ko = spec.TTM / ko_times
        else:                                       # 'D'
            dt_ko = dt_sim
        # 计算敲出观察点在模拟步数中的索引
        ko_indices = np.arange(
            ceil(dt_ko / dt_sim), n_steps + 1, ceil(dt_ko / dt_sim)
        )

        # ----------- 随机数：Sobol + Antithetic ------------------
        # 生成标准正态分布随机矩阵（基础路径×步数）
        z = generate_normal_matrix(n_paths=self.n_paths // (2 if self.antithetic else 1), dim=n_steps,seed = self.seed) #若使用对立路径，总路径数会翻倍
        if self.antithetic:
            # 对立路径：将随机数取负，总路径数翻倍，降低方差
            z = np.vstack((z, -z))

        # ----------- 生成 GBM 路径 -------------------------------
        # GBM（几何布朗运动）漂移项（单位时间）
        drift = (spec.r - spec.q - 0.5 * spec.sigma ** 2) * dt_sim
        # GBM波动率项（单位时间）
        vol_dt = spec.sigma * np.sqrt(dt_sim)
        # 计算对数价格路径（累积求和得到各时间点的对数收益）
        log_paths = np.cumsum(drift + vol_dt * z, axis=1)
        # 转换为实际价格路径（初始价格为S0，加上初始时刻的0）
        S_paths = spec.S0 * np.exp(np.hstack([np.zeros((z.shape[0], 1)), log_paths]))

        # ----------- KO / KI 监控 -------------------------------
        KO_bar = spec.KO_ratio * spec.strike # 敲出障碍价格
        KI_bar = spec.KI_ratio * spec.strike # 敲入障碍价格

        # 检查各观察点是否敲出（每行是一条路径，每列是一个观察点）
        ko_matrix = S_paths[:, ko_indices] >= KO_bar
        ko_flag = ko_matrix.any(axis=1) # 是否有路径敲出
        # 找到第一次敲出的索引（每行的第一个 True）
        first_ko = ko_matrix.argmax(axis=1)
        # 检查是否有路径敲入（KI）
        ki_flag = (S_paths[:, 1:] <= KI_bar).any(axis=1)

        # ----------- 现金流 -------------------------------------
        N = spec.notional # 名义本金
        c = spec.coupon # 年化票息
        payoffs = np.zeros(S_paths.shape[0]) # 存储每条路径的现金流

        # 1) KO 路径
        if ko_flag.any():
            ko_ts = ko_indices[first_ko[ko_flag]]
            ko_time = ko_ts * dt_sim
            # 现金流 = 本金×(1 + 票息率×持有时间)
            payoffs[ko_flag] = N * (1 + c * ko_time)

        # 2) 未 KO
        not_ko = ~ko_flag # 未敲出的路径标记
        if not_ko.any():
            # 2a 未敲入且未敲出（到期获得全额票息）
            cond = not_ko & (~ki_flag)
            payoffs[cond] = N * (1 + c * spec.TTM)

            # 2b 已敲入且未敲出（到期按规则结算）
            cond = not_ko & ki_flag
            ST = S_paths[cond, -1]
            if spec.note_type is SnowballType.PROTECTED:
                # 保本型：无论标的价格如何，返还本金
                payoffs[cond] = N
            else:
                # 非保本型：按到期价格与行权价的比例结算
                payoffs[cond] = N * (ST / spec.strike)

        # ----------- 折现 & 统计 -------------------------------
        # 将未来现金流按无风险利率折现到现在
        disc_pay = payoffs * np.exp(-spec.r * spec.TTM)
        price = disc_pay.mean()
        stderr = disc_pay.std(ddof=1) / np.sqrt(disc_pay.size)
        return price, stderr


# ----------------------------------------------------------------
# 3. DEMO
# ----------------------------------------------------------------
if __name__ == "__main__":
    spec = SnowballSpec(
        note_type=SnowballType.PROTECTED,
        notional=100000,
        S0=100,
        strike=100,
        KO_ratio=1.05,
        KI_ratio=0.70,
        coupon=0.12,
        TTM=1.0,
        r=0.03,
        q = 0.02,
        sigma=0.25,
        obs_freq="M",
    )

    pricer = SnowballMC(spec, n_paths=131_072)
    res = pricer.all_greeks()
    for k, v in res.items():
        print(f"{k:<6}: {v:10.4f}")