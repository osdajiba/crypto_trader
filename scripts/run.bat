@echo off
setlocal EnableDelayedExpansion

REM Script Name: scripts\run.bat
REM Description: Enhanced trading system launcher with CLI visualization

REM --------------------------
REM Environment Configuration
REM --------------------------
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%~dp0.."
set "MAIN_SCRIPT=%PROJECT_ROOT%\src\bin\main.py"
set "LOG_DIR=%PROJECT_ROOT%\logs"

REM Create logs directory if it doesn't exist
if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
)

REM Create a proper date-time string for log filename (Windows-safe)
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set LOGDATE=%%c-%%a-%%b)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set LOGTIME=%%a%%b)
set "LOG_FILE=%LOG_DIR%\app_%LOGDATE%_%LOGTIME%.log"

REM --------------------------
REM Simple ASCII Art Header
REM --------------------------
echo.
echo  ===============================================
echo    _______             _ _                
echo   ^|__   __^|           ^| (_)               
echo      ^| ^|_ __ __ _  __^| ^|_ _ __   __ _ 
echo      ^| ^| '__/ _` ^|/ _` ^| ^| '_ \ / _` ^|
echo      ^| ^| ^| ^| (_^| ^| (_^| ^| ^| ^| ^| ^| ^| (_^| ^|
echo      ^|_^|_^|  \__,_^|\__,_^|_^|_^|_^| ^|_^|\__, ^|
echo                                     __/ ^|
echo                                    ^|___/ 
echo    _____           _                 
echo   / ____^|         ^| ^|                
echo  ^| (___  _   _ ___^| ^|_ ___ _ __ ___  
echo   \___ \^| ^| ^| / __^| __/ _ \ '_ ` _ \ 
echo   ____) ^| ^|_^| \__ \ ^|^|  __/ ^| ^| ^| ^| ^|
echo  ^|_____/ \__, ^|___/\__\___^|_^| ^|_^| ^|_^|
echo           __/ ^|                      
echo          ^|___/                       
echo  ===============================================
echo.

REM --------------------------
REM Dependency Checks
REM --------------------------
echo [*] Checking dependencies...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Verify Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo [*] Python version: %PYTHON_VERSION%

REM Check if version is 3.8 or higher
set "MIN_VERSION=3.8"
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo [ERROR] Python 3.8+ required
    pause
    exit /b 1
)
if %MAJOR% EQU 3 (
    if %MINOR% LSS 8 (
        echo [ERROR] Python 3.8+ required
        pause
        exit /b 1
    )
)

REM --------------------------
REM Path Validation
REM --------------------------
if not exist "%MAIN_SCRIPT%" (
    echo [ERROR] Main script not found at %MAIN_SCRIPT%
    pause
    exit /b 1
)

REM --------------------------
REM Options Processing
REM --------------------------
set GUI_MODE=
set TRADING_MODE=
set CONFIG_FILE=
set STRATEGY=
set SYMBOL=
set TIMEFRAME=
set START_DATE=
set END_DATE=
set DEBUG=
set VERBOSE=

REM Process arguments
:arg_loop
if "%~1"=="" goto arg_loop_end

if /i "%~1"=="--gui" (
    set GUI_MODE=--gui
    shift
    goto arg_loop
)

if /i "%~1"=="--mode" (
    set TRADING_MODE=--mode %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--config" (
    set CONFIG_FILE=--config %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--strategy" (
    set STRATEGY=--strategy %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--symbol" (
    set SYMBOL=--symbol %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--timeframe" (
    set TIMEFRAME=--timeframe %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--start-date" (
    set START_DATE=--start-date %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--end-date" (
    set END_DATE=--end-date %~2
    shift & shift
    goto arg_loop
)

if /i "%~1"=="--debug" (
    set DEBUG=--debug
    shift
    goto arg_loop
)

if /i "%~1"=="--verbose" (
    set VERBOSE=-v
    shift
    goto arg_loop
)

if /i "%~1"=="-v" (
    set VERBOSE=-v
    shift
    goto arg_loop
)

REM Handle unknown arguments
echo [WARNING] Unknown argument: %~1
shift
goto arg_loop

:arg_loop_end

REM If no mode specified and not GUI, prompt for mode
if "%GUI_MODE%"=="" if "%TRADING_MODE%"=="" (
    call :prompt_mode_selection
)

REM --------------------------
REM Change to Project Root
REM --------------------------
cd /d "%PROJECT_ROOT%"

REM --------------------------
REM Command Construction
REM --------------------------
set "COMMAND=python "%MAIN_SCRIPT%" %GUI_MODE% %TRADING_MODE% %CONFIG_FILE% %STRATEGY% %SYMBOL% %TIMEFRAME% %START_DATE% %END_DATE% %DEBUG% %VERBOSE%"

REM --------------------------
REM Main Execution
REM --------------------------
echo.
echo [*] Launching trading system...
echo [*] Command: %COMMAND%
echo [*] Log file: %LOG_FILE%
echo [*] Working directory: %CD%
echo.
echo ===============================================

REM Execute with output to console and log file
%COMMAND% 2>&1 | findstr /V "^$" | tee "%LOG_FILE%"
if %ERRORLEVEL% neq 0 (
    %COMMAND%
)

set EXIT_CODE=%ERRORLEVEL%

echo ===============================================
echo.

REM Handle exit status
if %EXIT_CODE% equ 0 (
    echo [SUCCESS] System exited successfully
) else (
    echo [ERROR] System exited with error code: %EXIT_CODE%
)

pause
exit /b %EXIT_CODE%

REM --------------------------
REM Functions
REM --------------------------
:prompt_mode_selection
echo.
echo Trading Mode Selection
echo ===============================================
echo  1. Backtest         Historical data simulation
echo  2. Paper            Real-time simulation with mock execution
echo  3. Live             Real order execution (CAUTION)
echo ===============================================
echo Warning: Live trading requires proper configuration

set /p MODE_CHOICE="Select trading mode (1-3): "

if "%MODE_CHOICE%"=="1" (
    set TRADING_MODE=--mode backtest
) else if "%MODE_CHOICE%"=="2" (
    set TRADING_MODE=--mode paper
) else if "%MODE_CHOICE%"=="3" (
    set TRADING_MODE=--mode live
) else (
    echo Invalid selection. Defaulting to backtest mode.
    set TRADING_MODE=--mode backtest
)

echo You selected: %TRADING_MODE%
echo.
exit /b 0