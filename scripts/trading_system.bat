@echo off
REM trading_system.bat - Main launcher script for Windows
setlocal EnableDelayedExpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Get the project root directory (one level up from scripts)
for %%i in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fi"

REM Check environment directly (not as a function to avoid infinite loop)
echo.
echo [*] Checking environment and required files...
python "%PROJECT_ROOT%\src\common\check_env.py"
if %ERRORLEVEL% NEQ 0 (
    echo [!] Environment check reported issues that need to be resolved.
    echo Press any key to continue or Ctrl+C to abort...
    pause > nul
    echo [WARNING] Continuing despite environment check issues.
)

REM Change to project root directory
cd "%PROJECT_ROOT%"

echo Trading System Launcher
echo ==========================
echo [1] Start in GUI mode
echo [2] Start in CLI mode
echo [3] Exit
echo ==========================

REM Clear any previous input and get user option
set "option="
set /p option="Enter option (1-3): "

REM Ensure proper handling of options
if "%option%"=="1" goto gui_mode
if "%option%"=="2" goto cli_mode_setup
if "%option%"=="3" goto exit_script

echo Invalid option selected! Please enter 1, 2, or 3.
goto exit_script

:gui_mode
echo.
echo [*] Launching trading system
echo [*] Launching in --gui mode
python src\main\main.py --gui
goto after_run

:cli_mode_setup
echo.
echo [*] Launching in CLI mode...
echo.
echo CLI Mode Configuration
echo ==========================
echo Select trading mode
echo [1] Backtest - Historical data simulation
echo [2] Paper - Real-time simulation with virtual orders
echo [3] Live - Real order execution (USE WITH CAUTION)
echo ==========================

set /p trade_mode="Enter trading mode (1-3): "

set mode_param=backtest
if "%trade_mode%"=="1" set mode_param=backtest
if "%trade_mode%"=="2" set mode_param=paper
if "%trade_mode%"=="3" set mode_param=live

echo.
echo Selected trading mode: %mode_param%
echo.

echo Select strategy
echo [1] dual_ma (Dual Moving Average)
echo [2] neural_network
echo [3] Custom (specify name)

set /p strat_choice="Enter strategy choice (1-3): "

set strategy_param=dual_ma
if "%strat_choice%"=="1" set strategy_param=dual_ma
if "%strat_choice%"=="2" set strategy_param=neural_network
if "%strat_choice%"=="3" (
    set /p strategy_param="Enter custom strategy name: "
)

echo.
echo Selected strategy: %strategy_param%
echo.

set /p symbol_param="Enter trading symbols (comma-separated, default=BTC/USDT): "
if "%symbol_param%"=="" set symbol_param=BTC/USDT

echo.
echo Select timeframe
echo [1] 1m  [2] 5m  [3] 15m  [4] 30m  [5] 1h  [6] 4h  [7] 1d  [8] 1w

set /p tf_choice="Enter timeframe choice (1-8): "

set timeframe_param=1h
if "%tf_choice%"=="1" set timeframe_param=1m
if "%tf_choice%"=="2" set timeframe_param=5m
if "%tf_choice%"=="3" set timeframe_param=15m
if "%tf_choice%"=="4" set timeframe_param=30m
if "%tf_choice%"=="5" set timeframe_param=1h
if "%tf_choice%"=="6" set timeframe_param=4h
if "%tf_choice%"=="7" set timeframe_param=1d
if "%tf_choice%"=="8" set timeframe_param=1w

echo.
echo Selected timeframe: %timeframe_param%
echo.

if "%mode_param%"=="backtest" (
    set /p start_date="Enter backtest start date (YYYY-MM-DD, default=2023-01-01): "
    if "!start_date!"=="" set start_date=2023-01-01
    
    set /p end_date="Enter backtest end date (YYYY-MM-DD, default=2023-12-31): "
    if "!end_date!"=="" set end_date=2023-12-31
) else (
    set start_date=
    set end_date=
)

echo.
set /p debug_mode="Enable debug mode (y/n, default=n): "
set debug_param=
if /i "%debug_mode%"=="y" set debug_param=--debug

echo.
set /p additional="Enter any additional parameters (or press Enter to skip): "

echo.
echo [*] Launching trading system
echo [*] Launching in CLI mode with the following configuration
echo     - Mode: %mode_param%
echo     - Strategy: %strategy_param%
echo     - Symbols: %symbol_param%
echo     - Timeframe: %timeframe_param%
if defined start_date echo     - Start date: %start_date%
if defined end_date echo     - End date: %end_date%
if defined debug_param echo     - Debug mode: Enabled
if defined additional echo     - Additional parameters: %additional%
echo.

set cmd=python src\main\main.py --cli --mode %mode_param% --strategy %strategy_param% --symbol "%symbol_param%" --timeframe %timeframe_param%

if defined start_date set cmd=%cmd% --start-date %start_date%
if defined end_date set cmd=%cmd% --end-date %end_date%
if defined debug_param set cmd=%cmd% %debug_param%
if defined additional set cmd=%cmd% %additional%

echo Command: %cmd%
echo.
%cmd%
goto after_run

:after_run
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] System exited with error code: %ERRORLEVEL%
) else (
    echo [SUCCESS] System completed successfully
)
pause
goto exit_script

:exit_script
echo Exiting...
exit /b 0