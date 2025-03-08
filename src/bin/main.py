# src/bin/main.py

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.common.async_executor import AsyncExecutor
from src.core.core import TradingCore
from src.mode.trading_mode_factory import TradingModeFactory

BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = BASE_DIR / "conf/bt_config.yaml"


async def async_main() -> Dict[str, Any]:
    """异步主函数入口"""
    # 检查配置文件是否存在
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
        
    # 1. 初始化配置并验证
    config = ConfigManager(config_path=CONFIG_PATH)
    log_dir = BASE_DIR / config.get("log_paths", "error_log", default="logs/error.log")

    # 2. 初始化日志系统
    LogManager(base_dir=log_dir)  # 创建日志管理器单例
    log_level = config.get("logging", "level", default="INFO")
    logger = LogManager.get_logger(name="trading_system", level=log_level)
    logger.info("Trading system initializing...")

    # 3. 初始化交易核心
    core = TradingCore(config)
    
    # 4. 运行交易流程
    logger.info("Starting trading pipeline...")
    try:
        result = await core.run_pipeline()
        if "error" in result:
            logger.error(f"Trading pipeline failed: {result['error']}")
        else:
            logger.info("Trading pipeline completed successfully")
        return result
    finally:
        # 确保资源被正确释放
        logger.info("Shutting down trading system...")
        await core.shutdown()


def main() -> Optional[Dict[str, Any]]:
    """同步主函数入口"""
    try:
        # 使用AsyncExecutor运行异步主函数
        executor = AsyncExecutor()
        return executor.run(async_main())

    except FileNotFoundError as e:
        print(f"Configuration error: {str(e)}")
        return {"error": f"Configuration error: {str(e)}"}

    except Exception as e:
        print(f"Critical error: {str(e)}")
        return {"error": f"Critical error: {str(e)}"}


if __name__ == "__main__":
    try:
        # 运行主函数并获取结果
        report = main()

        # 显示结果摘要
        if report:
            if isinstance(report, dict) and "error" not in report:
                print(f"\nTrading Results Summary:")
                print(f"-------------------------")
                for key, value in report.items():
                    if key not in ["trades", "equity_curve"] and not key.startswith("_"):
                        print(f"{key}: {value}")
            elif isinstance(report, dict) and "error" in report:
                print(f"\nError occurred: {report['error']}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)