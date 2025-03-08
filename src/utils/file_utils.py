# src\utils\file_utils.py

import os
import json
import csv
from typing import Any, Dict, List

class FileUtils:
    """文件操作工具类"""

    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """读取 JSON 文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any]) -> None:
        """写入 JSON 文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def read_csv(file_path: str) -> List[Dict[str, Any]]:
        """读取 CSV 文件"""
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def ensure_dir(directory: str) -> None:
        """确保目录存在"""
        if not os.path.exists(directory):
            os.makedirs(directory)

# 示例用法
if __name__ == "__main__":
    FileUtils.ensure_dir("test_data")
    data = {"key": "value"}
    FileUtils.write_json("test_data/sample.json", data)
    print(FileUtils.read_json("test_data/sample.json"))