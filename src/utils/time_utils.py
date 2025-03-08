# utils/time_utils.py

import pytz
from datetime import datetime, timedelta
from typing import Union, Optional

class TimeUtils:
    """时间处理工具类"""

    @staticmethod
    def parse_timestamp(
        time_val: Union[str, datetime, int, float, None],
        default_delta: Optional[timedelta] = None,
        tz: str = 'Asia/Shanghai'
    ) -> datetime:
        """
        统一解析时间，支持多种输入格式并转换为指定时区的时间。
        
        Args:
            time_val (Union[str, datetime, int, float, None]): 输入时间值。
            default_delta (Optional[timedelta]): 默认时间偏移量（当输入为 None 时）。
            tz (str): 目标时区，默认 'Asia/Shanghai'。
        
        Returns:
            datetime: 解析后的时间（带时区）。
        
        Raises:
            ValueError: 时间格式不支持或解析失败。
        """
        target_tz = pytz.timezone(tz)
        try:
            if time_val is None:
                base_time = datetime.now(pytz.utc)
                return (base_time - default_delta).astimezone(target_tz) if default_delta else base_time.astimezone(target_tz)

            if isinstance(time_val, str):
                formats = [
                    '%Y-%m-%d %H:%M:%S',  # 完整日期时间
                    '%Y-%m-%d',           # 仅日期
                    '%H:%M:%S',           # 仅时间（当天）
                ]
                for fmt in formats:
                    try:
                        dt = datetime.strptime(time_val, fmt)
                        if fmt == '%H:%M:%S':
                            dt = datetime.combine(datetime.now().date(), dt.time())
                        return pytz.utc.localize(dt).astimezone(target_tz)
                    except ValueError:
                        continue
                raise ValueError(f"无法解析的时间格式: {time_val}")

            elif isinstance(time_val, datetime):
                if time_val.tzinfo is None:
                    return pytz.utc.localize(time_val).astimezone(target_tz)
                return time_val.astimezone(target_tz)

            elif isinstance(time_val, (int, float)):
                divisor = 1000 if time_val > 1e12 else 1  # 判断毫秒或秒
                dt = datetime.fromtimestamp(time_val / divisor, pytz.utc)
                return dt.astimezone(target_tz)

            raise ValueError(f"不支持的时间类型: {type(time_val)}")
        except Exception as e:
            raise ValueError(f"时间解析失败: {str(e)}")

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

