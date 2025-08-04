# -*- coding: utf-8 -*-
# @author: Li Yijia
# @date: 2025/08/04
# @filename: asian.py

"""
==================================================================
算术平均亚式期权定价器
------------------------------------------------------------------
• 欧式到期行权 (European-style)
• Control-Variate Monte-Carlo:
      V_A  = V_A_MC              (arithmetic)
            + β*(V_G_analytic - V_G_MC)         (geometric)
  其中 β =  Cov(A,G)/Var(G)
• 输出 price, delta, gamma, vega, theta, rho  (有限差分)
==================================================================
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from math import log, sqrt
from typing import Tuple

from .utils import bump_parameter, generate_normal_matrix

# ---------------------------------------------------------------
# 1. 基础枚举
# ---------------------------------------------------------------
class OptType(Enum):
    CALL = "CALL"
    PUT  = "PUT"

# ---------------------------------------------------------------
# 2. 期权规格
# ---------------------------------------------------------------
@dataclass
class AsianSpec:
    S0      : float
    K       : float
    r       : float
    q       : float
    sigma   : float
    TTM       : float
    opt_type: OptType
    n_obs   : int # 观测次数 (包括起始点)


# ---------------------------------------------------------------
# 3. 主定价类
# ---------------------------------------------------------------
@dataclass
class AsianMC:
    """
    Arithmetic-average Asian option pricer with control-variate.
    """
    spec: AsianSpec
    n_paths: int
    seed: int = 666
    antithetic: bool = True
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

    # 每个 Greek 都是中心差分 1 bp
    def delta(self) -> float:
        if "delta" not in self._cache:
            bump = self.spec.S0 * 1e-4
            up = self._price_core(bump_parameter(self.spec, "S0", +bump))[0]
            dn = self._price_core(bump_parameter(self.spec, "S0", -bump))[0]
            self._cache["delta"] = (up - dn) / (2 * bump)
        return self._cache["delta"]

    def gamma(self) -> float:
        if "gamma" not in self._cache:
            bump = self.spec.S0 * 1e-4
            up = self._price_core(bump_parameter(self.spec, "S0", +bump))[0]
            dn = self._price_core(bump_parameter(self.spec, "S0", -bump))[0]
            mid = self.price()
            self._cache["gamma"] = (up - 2 * mid + dn) / bump**2
        return self._cache["gamma"]

    def vega(self) -> float:
        if "vega" not in self._cache:
            bump = 1e-4
            up = self._price_core(bump_parameter(self.spec, "sigma", +bump))[0]
            dn = self._price_core(bump_parameter(self.spec, "sigma", -bump))[0]
            self._cache["vega"] = (up - dn) / (2 * bump) / 100  # → 1%
        return self._cache["vega"]

    def theta(self) -> float:
        if "theta" not in self._cache:
            dT = 1 / 252
            if self.spec.TTM <= dT:
                raise ValueError("TTM too small for theta.")
            dn = self._price_core(bump_parameter(self.spec, "TTM", -dT))[0]
            self._cache["theta"] = (dn - self.price()) / (-dT) / 365
        return self._cache["theta"]

    def rho(self) -> float:
        if "rho" not in self._cache:
            bump = 1e-4
            up = self._price_core(bump_parameter(self.spec, "r", +bump))[0]
            dn = self._price_core(bump_parameter(self.spec, "r", -bump))[0]
            self._cache["rho"] = (up - dn) / (2 * bump) / 100
        return self._cache["rho"]

    def all_greeks(self):
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
        }
    # ================================================================
    # 核心定价
    # ================================================================
    def _price_core(self, sp: AsianSpec) -> Tuple[float, float]:
        """
        Control-variate Monte-Carlo pricing.
        Return (price, stderr)
        """
        N = sp.n_obs
        dt = sp.TTM / (N - 1)

        # ----------- 随机路径 ------------------
        z = generate_normal_matrix(n_paths=self.n_paths // (2 if self.antithetic else 1), dim = N - 1,seed = self.seed)
        if self.antithetic:
            z = np.vstack((z, -z))

        drift = (sp.r - sp.q - 0.5 * sp.sigma**2) * dt
        vol_dt = sp.sigma * sqrt(dt)
        log_incr = drift + vol_dt * z
        log_paths = np.cumsum(log_incr, axis=1)
        S_paths = sp.S0 * np.exp(np.hstack([np.zeros((log_paths.shape[0], 1)), log_paths]))

        # ----------- 计算 arithmetic & geometric 均价 ----------
        A = S_paths.mean(axis=1)
        G = np.exp(np.log(S_paths).mean(axis=1))  # 几何平均

        # ----------- Payoff ----------------------------------
        if sp.opt_type is OptType.CALL:
            pay_A = np.maximum(A - sp.K, 0)
            pay_G = np.maximum(G - sp.K, 0)
        else:
            pay_A = np.maximum(sp.K - A, 0)
            pay_G = np.maximum(sp.K - G, 0)

        # ----------- Control-Variate --------------------------
        # 几何平均的解析现值
        geo_price = self._geom_closed_form(sp)

        # β = Cov(A,G)/Var(G)
        cov = np.cov(pay_A, pay_G, ddof=1)
        beta = cov[0, 1] / cov[1, 1]

        pay_cv = pay_A + beta * (geo_price - pay_G)
        disc_pay = np.exp(-sp.r * sp.TTM) * pay_cv

        price = disc_pay.mean()
        stderr = disc_pay.std(ddof=1) / np.sqrt(disc_pay.size)
        return price, stderr

    # -------------------- 几何平均解析解 -----------------------
    def _geom_closed_form(self, sp: AsianSpec) -> float:
        """
        Closed-form geometric Asian option (Kemna & Vorst 1990).
        """
        N = sp.n_obs
        sigma_g = sp.sigma * sqrt((2 * N + 1) / (6 * N))
        mu_g = 0.5 * (sp.r - sp.q - 0.5 * sp.sigma**2) + 0.5 * sigma_g**2

        d1 = (log(sp.S0 / sp.K) + (mu_g + 0.5 * sigma_g**2) * sp.TTM) / (sigma_g * sqrt(sp.TTM))
        d2 = d1 - sigma_g * sqrt(sp.TTM)

        from scipy.stats import norm

        if sp.opt_type is OptType.CALL:
            price = (
                np.exp(-sp.q * sp.TTM) * sp.S0 * np.exp(mu_g * sp.TTM) * norm.cdf(d1)
                - sp.K * np.exp(-sp.r * sp.TTM) * norm.cdf(d2)
            )
        else:
            price = (
                sp.K * np.exp(-sp.r * sp.TTM) * norm.cdf(-d2)
                - np.exp(-sp.q * sp.TTM) * sp.S0 * np.exp(mu_g * sp.TTM) * norm.cdf(-d1)
            )
        return price