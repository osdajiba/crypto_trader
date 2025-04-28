#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def debug_python_script():
    # 配置参数 (根据您的 launch.json 调整)
    config = {
        "python_interpreter": "/home/linuxuser/miniconda3/bin/python",
        "just_my_code": True,
        "console_type": "integratedTerminal",
        "debug_port": 5678  # 默认调试端口
    }

    # 获取要调试的文件 (模拟VSCode的${file}变量)
    target_file = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if not target_file or not Path(target_file).exists():
        print("Error: Please specify a valid Python file to debug")
        print(f"Usage: {sys.argv[0]} <python_script.py>")
        return False

    # 构建调试命令
    cmd = [
        config["python_interpreter"],
        "-m",
        "debugpy",
        "--listen",
        str(config["debug_port"]),
        "--wait-for-client",
        target_file
    ]

    # 添加justMyCode选项
    if config["just_my_code"]:
        cmd.insert(3, "--configure")
        cmd.insert(4, "justMyCode=true")

    print("\nStarting Python debug session...")
    print(f"  Python: {config['python_interpreter']}")
    print(f"  Target: {target_file}")
    print(f"  Debug port: {config['debug_port']}")
    print(f"  Just my code: {config['just_my_code']}\n")

    # 运行调试命令
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"Debugger started (PID: {process.pid})")
        print("Waiting for debug client to connect...\n")

        # 实时输出处理
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            if output:
                print(output.strip())
            if error:
                print(error.strip(), file=sys.stderr)
            
            # 检查进程是否结束
            if process.poll() is not None:
                break

        return process.returncode == 0

    except Exception as e:
        print(f"Debugger failed to start: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if debug_python_script():
        sys.exit(0)
    else:
        sys.exit(1)