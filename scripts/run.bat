@echo off
REM 脚本名称: scripts\run.bat
REM 描述: 运行 src\bin\main.py 文件

REM 设置 Python 解释器的路径（如果未设置 PATH 环境变量）
REM 如果需要，可以手动指定 Python 解释器的路径，例如：
REM set PYTHON_PATH=C:\Python39\python.exe

REM 检查 Python 是否已安装
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 未安装或未添加到系统 PATH 环境变量。
    echo 请安装 Python 并将其添加到 PATH 环境变量。
    pause
    exit /b 1
)

REM 设置要运行的 Python 脚本路径
set SCRIPT_PATH=src\bin\main.py

REM 检查脚本是否存在
if not exist "%SCRIPT_PATH%" (
    echo 脚本文件 %SCRIPT_PATH% 不存在。
    pause
    exit /b 1
)

REM 运行 Python 脚本
echo 正在运行脚本: %SCRIPT_PATH%
python "%SCRIPT_PATH%"

REM 检查脚本是否成功运行
if %errorlevel% equ 0 (
    echo 脚本运行成功。
) else (
    echo 脚本运行失败，错误代码: %errorlevel%
)

REM 暂停，以便查看输出结果
pause