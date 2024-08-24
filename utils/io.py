import numpy as np

import os
import json
from typing import Literal, Dict


def create_dir(path: str):
    """
        创建文件夹
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"create {path} success!")


def read_json(path: str):
    """
        读取 json 文件
    """
    with open(path) as f:
        content = json.loads(f.read())
    return content

def save_json(path: str, content: Dict):
    with open(path, mode='w') as f:
        json.dump(content, f, ensure_ascii=False)

def read_dict_npy(path: str):
    return np.load(path, allow_pickle=True).item()

def read_npy(path: str):
    return np.load(path, allow_pickle=True)