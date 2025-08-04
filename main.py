# -*- coding: utf-8 -*-
# @author: Li Yijia
# @date: 2025/08/04
# @filename: main.py

"""
==================================================================
批量定价脚本
----------------------------------------
读取 data 目录下的 csv，分别定价 Vanilla 期权 & 雪球票据，
结果打印到屏幕并保存到 results/*.csv
==================================================================
"""

import pandas as pd
from pathlib import Path
import yaml

# ---------- pricer 模块 ------------------------------
from pricers.vanilla import OptionSpec, OptType, ExerciseStyle, BlackScholes, CRRBinomial
from pricers.snowball import SnowballMC, SnowballSpec, SnowballType
from pricers.asian import AsianSpec, AsianMC, OptType

DATA_DIR   = Path(__file__).parent / "data"
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

# -------- 读取配置 --------
cfg = yaml.safe_load(open(Path(__file__).with_name('config')/'settings.yaml',encoding ='utf-8'))

# ====================================================
# 一、Vanilla 期权
# ====================================================
opt_df = pd.read_csv(DATA_DIR / "vanillas.csv")

opt_results = []
for _, row in opt_df.iterrows():
    spec = OptionSpec(
        S0      = row.S0,
        K       = row.K,
        TTM       = row.TTM,
        r       = cfg['risk_free_rate'],
        q       = row.q,
        sigma   = row.sigma,
        opt_type= OptType[row.opt_type.strip().upper()],
        style   = ExerciseStyle[row.style.strip().upper()],
    )

    if spec.style is ExerciseStyle.EUROPEAN:
        pricer = BlackScholes(spec)                # 解析解
    else:
        pricer = CRRBinomial(spec, steps=cfg['steps'])      # 二叉树

    greeks = pricer.all_greeks()
    greeks["id"] = row.id.strip()
    opt_results.append(greeks)

opt_out = pd.DataFrame(opt_results).set_index("id")
opt_out.to_csv(RESULT_DIR / "vanillas_pricing.csv")
print("=== Vanilla Options ===")
print(opt_out, "\n")


# ====================================================
# 二、Snowball 票据
# ====================================================
sb_df = pd.read_csv(DATA_DIR / "snowballs.csv")

sb_results = []
for _, row in sb_df.iterrows():
    spec = SnowballSpec(
        note_type = SnowballType[row.note_type.strip().upper()],
        notional  = row.notional,
        S0        = row.S0,
        strike    = row.strike,
        r         = cfg['risk_free_rate'],
        q         = row.q,
        sigma     = row.sigma,
        KO_ratio  = row.KO_ratio,
        KI_ratio  = row.KI_ratio,
        coupon    = row.coupon,
        TTM         = row.TTM,
        obs_freq  = row.obs_freq.strip().upper(),
    )

    pricer = SnowballMC(spec, n_paths=cfg['n_path'],seed=cfg['seed'],antithetic = cfg['antithetic'])
    greeks = pricer.all_greeks()
    greeks["id"] = row.id.strip()
    sb_results.append(greeks)

sb_out = pd.DataFrame(sb_results).set_index("id")
sb_out.to_csv(RESULT_DIR / "snowballs_pricing.csv")
print("=== Snowball Notes ===")
print(sb_out)

# ====================================================
# 三、Asian Options
# ====================================================

asian_df = pd.read_csv(DATA_DIR / "asians.csv")

asian_res = []
for _, row in asian_df.iterrows():
    spec = AsianSpec(
        S0      = row.S0,
        K       = row.K,
        TTM       = row.TTM,
        r       = row.r,
        q       = row.q,
        sigma   = row.sigma,
        opt_type= OptType[row.opt_type.strip().upper()],
        n_obs   = cfg['n_obs']  # 使用配置文件中的观测次数,
    )
    pricer = AsianMC(spec, n_paths=cfg['n_path'],seed=cfg['seed'],antithetic = cfg['antithetic'])
    gk = pricer.all_greeks()
    gk["id"] = row["id"].strip()
    asian_res.append(gk)

asian_out = pd.DataFrame(asian_res).set_index("id")
asian_out.to_csv(RESULT_DIR / "asians_pricing.csv")
print("\n=== Asian Options ===")
print(asian_out)