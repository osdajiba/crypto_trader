#!/bin/bash
# 脚本名称: scripts/run.sh
# 描述: 运行 src/bin/main.py 文件

# 检查 Python 是否已安装
if ! command -v python3 &> /dev/null; then
    echo "Python 3 未安装。请安装 Python 3 并将其添加到 PATH 环境变量。"
    exit 1
fi

# 设置要运行的 Python 脚本路径
SCRIPT_PATH="src/bin/main.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "脚本文件 $SCRIPT_PATH 不存在。"
    exit 1
fi

# 运行 Python 脚本
echo "正在运行脚本: $SCRIPT_PATH"
python3 "$SCRIPT_PATH"

# 检查脚本是否成功运行
if [ $? -eq 0 ]; then
    echo "脚本运行成功。"
else
    echo "脚本运行失败，错误代码: $?"
fi