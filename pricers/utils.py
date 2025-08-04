from dataclasses import dataclass
import numpy as np

def bump_parameter(obj: dataclass, field: str, amount: float) -> dataclass:
    """
    生成参数扰动后的新对象
    
    参数:
        obj: 原始数据类对象
        field: 要扰动的字段名
        amount: 扰动幅度
        
    返回:
        新的、参数被扰动的数据类对象
    """
    # 将原对象属性转换为字典，更新指定字段的值
    obj_dict = {**obj.__dict__, field: getattr(obj, field) + amount}
    # 创建并返回新的对象
    return obj.__class__(** obj_dict)

def generate_normal_matrix(n_paths: int, dim: int,seed) -> np.ndarray:
    """
    返回 n_paths × dim 的 N(0,1) 矩阵。
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, dim))
    return z