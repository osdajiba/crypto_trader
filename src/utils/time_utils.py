# utils/time_utils.py

import pytz
from datetime import datetime, timedelta, timezone
from typing import Union, Optional

from src.common.log_manager import LogManager

logger = LogManager.get_logger("trading_system")


class TimeUtils:
    """时间处理工具类"""
    
    @staticmethod
    def parse_timestamp(timestamp: Union[str, datetime, int, float, None], 
                       default_days_ago: int = 30) -> Optional[datetime]:
        """
        Parse various timestamp formats into a datetime object
        
        Args:
            timestamp: Timestamp in various formats
            default_days_ago: Days ago to use if timestamp is None
                
        Returns:
            Datetime object with timezone (UTC)
        """
        # Return default if None
        if timestamp is None:
            return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
        
        # If already a datetime
        if isinstance(timestamp, datetime):
            # Add timezone if missing
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp
        
        # Parse integers (timestamps)
        if isinstance(timestamp, (int, float)):
            # Check if milliseconds or seconds
            if timestamp > 1e11:  # Milliseconds have more digits
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            else:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Parse strings with common formats
        if isinstance(timestamp, str):
            # Try ISO format first
            try:
                # Handle 'Z' suffix for UTC time
                clean_ts = timestamp.replace('Z', '+00:00')
                # Handle ISO format
                if 'T' in clean_ts or '+' in clean_ts or '-' in clean_ts and 'T' in clean_ts:
                    dt = datetime.fromisoformat(clean_ts)
                    # Add UTC timezone if missing
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
            except ValueError:
                pass
            
            # Try common date formats
            date_formats = [
                '%Y-%m-%d',       # 2023-01-31
                '%Y/%m/%d',       # 2023/01/31
                '%m/%d/%Y',       # 01/31/2023
                '%d-%m-%Y',       # 31-01-2023
                '%Y-%m-%d %H:%M:%S',  # 2023-01-31 14:30:00
                '%Y-%m-%dT%H:%M:%S',  # 2023-01-31T14:30:00
                '%Y%m%d'          # 20230131
            ]
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    # Add UTC timezone
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # If all formats fail, log the issue
            logger.warning(f"Couldn't parse timestamp: {timestamp}")
            # Return default
            return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
        
        # For any other type, return default
        logger.warning(f"Unsupported timestamp type: {type(timestamp)}")
        return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
    
    @staticmethod
    def to_timestamp(dt: datetime, milliseconds: bool = True) -> int:
        """
        将 datetime 转换为时间戳。
        
        Args:
            dt (datetime): 输入时间。
            milliseconds (bool): 是否返回毫秒（否则返回秒）。
        
        Returns:
            int: 时间戳。
        """
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        factor = 1000 if milliseconds else 1
        return int(dt.timestamp() * factor)

# 示例用法
if __name__ == "__main__":
    # 测试不同输入
    print(TimeUtils.parse_timestamp("2023-01-01"))  # ECT
    print(TimeUtils.parse_timestamp(1698777600000))  # 毫秒时间戳
    print(TimeUtils.parse_timestamp(datetime.now()))
    print(TimeUtils.parse_timestamp(None, timedelta(days=1)))

    # 测试时间戳转换
    dt = TimeUtils.parse_timestamp("2023-01-01")
    print(TimeUtils.to_timestamp(dt, milliseconds=True))

