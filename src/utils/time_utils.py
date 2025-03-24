# utils/time_utils.py

import pandas as pd
import pytz
from datetime import datetime, timedelta, timezone
from typing import Union, Optional, Tuple, List

from src.common.log_manager import LogManager

logger = LogManager.get_logger("trading_system")


class TimeUtils:
    """Enhanced time processing utility class"""
    
    @staticmethod
    def parse_timestamp(timestamp: Union[str, datetime, int, float, None], 
                    default_days_ago: int = 30) -> Optional[datetime]:
        """Parse various timestamp formats into a datetime object"""
        # 定义默认返回值函数
        def default_time():
            return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
        
        # 处理None
        if timestamp is None:
            return default_time()
        
        # 处理datetime对象
        if isinstance(timestamp, datetime):
            return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        
        # 处理数字时间戳
        if isinstance(timestamp, (int, float)):
            # 使用三元表达式检查是否为毫秒
            divisor = 1000 if timestamp > 1e11 else 1
            return datetime.fromtimestamp(timestamp / divisor, tz=timezone.utc)
        
        # 处理字符串时间戳
        if isinstance(timestamp, str):
            # 清理常见的时区指示符
            clean_ts = timestamp.replace('Z', '+00:00')
            
            # 尝试ISO格式
            has_iso_indicators = ('T' in clean_ts or 
                                ('+' in clean_ts or '-' in clean_ts and 'T' in clean_ts))
            if has_iso_indicators:
                try:
                    dt = datetime.fromisoformat(clean_ts)
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            
            # 尝试常见日期格式
            date_formats = {
                '%Y-%m-%d': None,
                '%Y/%m/%d': None,
                '%m/%d/%Y': None,
                '%d-%m-%Y': None,
                '%Y-%m-%d %H:%M:%S': None,
                '%Y-%m-%dT%H:%M:%S': None,
                '%Y%m%d': None
            }
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        
        # 对于所有其他情况记录警告并返回默认值
        logger.warning(f"Couldn't parse timestamp: {timestamp}")
        return default_time()
    
    @staticmethod
    def to_timestamp(dt: datetime, milliseconds: bool = True) -> int:
        """Convert datetime to a timestamp"""
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        factor = 1000 if milliseconds else 1
        return int(dt.timestamp() * factor)
    
    @staticmethod
    def from_timestamp(timestamp: Union[int, float], milliseconds: bool = False) -> datetime:
        """Convert timestamp to datetime object"""
        if milliseconds:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    @staticmethod
    def ensure_tz_aware(dt: Union[datetime, pd.Timestamp]) -> Union[datetime, pd.Timestamp]:
        """Ensure a datetime object has timezone information (UTC)"""
        if dt is None:
            return datetime.now(timezone.utc)
            
        # 处理pandas Timestamp
        if isinstance(dt, pd.Timestamp):
            return dt if dt.tzinfo else dt.tz_localize('UTC')
            
        # 处理Python datetime
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def is_in_date_range(dt: datetime, start_dt: datetime, end_dt: datetime) -> bool:
        """Check if a datetime is within a date range"""
        # Ensure all datetimes are timezone-aware
        dt = TimeUtils.ensure_tz_aware(dt)
        start_dt = TimeUtils.ensure_tz_aware(start_dt)
        end_dt = TimeUtils.ensure_tz_aware(end_dt)
        
        return start_dt <= dt <= end_dt
    
    @staticmethod
    def is_timestamp_in_range(ts: int, start_ts: int, end_ts: int) -> bool:
        """Check if a timestamp is within a timestamp range"""
        return start_ts <= ts <= end_ts
    
    @staticmethod
    def round_datetime(dt: datetime, interval_minutes: int = 5) -> datetime:
        """Round a datetime to the nearest interval"""
        minutes = dt.minute
        rounded_minutes = ((minutes + interval_minutes // 2) // interval_minutes) * interval_minutes
        
        # Create new datetime with rounded minutes
        rounded_dt = dt.replace(minute=rounded_minutes % 60, second=0, microsecond=0)
        
        # Add an hour if needed
        if rounded_minutes >= 60:
            rounded_dt = rounded_dt + timedelta(hours=rounded_minutes // 60)
            
        return rounded_dt
    
    @staticmethod
    def get_time_intervals(start_dt: datetime, end_dt: datetime, 
                        interval: Union[str, timedelta]) -> List[datetime]:
        """Generate a list of datetime points at regular intervals"""
        # 解析字符串间隔为timedelta
        if isinstance(interval, str):
            unit_map = {'d': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}
            unit = interval[-1].lower()
            value = int(interval[:-1])
            
            if unit not in unit_map:
                raise ValueError(f"Unsupported interval unit: {unit}")
                
            delta = timedelta(**{unit_map[unit]: value})
        else:
            delta = interval
        
        # 使用列表推导式生成时间点
        result = []
        current = start_dt
        while current <= end_dt:
            result.append(current)
            current += delta
            
        return result
    
    @staticmethod
    def get_start_end_of_day(dt: datetime) -> Tuple[datetime, datetime]:
        """Get start and end of day for a given datetime"""
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return start, end
    
    @staticmethod
    def get_start_end_of_month(dt: datetime) -> Tuple[datetime, datetime]:
        """Get start and end of month for a given datetime"""
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Find the last day of the month
        if dt.month == 12:
            end_month = dt.replace(year=dt.year+1, month=1, day=1)
        else:
            end_month = dt.replace(month=dt.month+1, day=1)
            
        end = end_month - timedelta(microseconds=1)
        return start, end
    
    @staticmethod
    def get_current_timestamp(milliseconds: bool = True) -> int:
        """Get the current UTC timestamp"""
        dt = datetime.now(timezone.utc)
        return TimeUtils.to_timestamp(dt, milliseconds)
    
    @staticmethod
    def format_datetime(dt: datetime, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format a datetime object to string"""
        return dt.strftime(fmt)
    
    @staticmethod
    def parse_timeframe(timeframe: str) -> timedelta:
        """Parse a timeframe string to timedelta"""
        unit = timeframe[-1].lower()
        
        try:
            value = int(timeframe[:-1])
        except ValueError:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 's':
            return timedelta(seconds=value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

