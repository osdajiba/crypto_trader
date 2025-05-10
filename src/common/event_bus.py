#!/usr/bin/env python3
# src/common/event_bus.py

import asyncio
import time
from typing import Dict, Set, Callable, Any, List, Optional
import uuid
import logging

from src.common.log_manager import LogManager

logger = LogManager.get_logger("common.event_bus")

class EventBus:
    """
    轻量级事件总线，用于组件间通信
    
    特点:
    - 支持异步事件处理
    - 支持事件过滤
    - 支持事件优先级
    - 事件回放功能（可选）
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取事件总线单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._subscribers = {}  # 类型: Dict[str, Set[Callable]]
        self._filters = {}      # 类型: Dict[str, List[Callable]]
        self._event_history = {}  # 类型: Dict[str, List[Dict]]
        self._max_history = 100  # 每种事件最多保留的历史记录数
        self._record_history = False
        
    def set_history_recording(self, enabled: bool, max_items: int = 100):
        """设置是否记录事件历史"""
        self._record_history = enabled
        self._max_history = max_items
        
    def clear_history(self, event_type: Optional[str] = None):
        """清除事件历史"""
        if event_type:
            self._event_history.pop(event_type, None)
        else:
            self._event_history.clear()
    
    def subscribe(self, event_type: str, callback: Callable, 
                  priority: int = 0) -> str:
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数，接收事件数据
            priority: 优先级 (0最高)
            
        Returns:
            str: 订阅ID，用于取消订阅
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
            
        # 创建唯一的订阅ID
        subscription_id = str(uuid.uuid4())
        
        # 存储回调和元数据
        self._subscribers[event_type].add({
            'id': subscription_id,
            'callback': callback,
            'priority': priority,
            'created_at': time.time()
        })
        
        logger.debug(f"Subscribed to {event_type} with ID {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, event_type: str, subscription_id: str) -> bool:
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            subscription_id: 订阅ID
            
        Returns:
            bool: 是否成功取消
        """
        if event_type not in self._subscribers:
            return False
            
        for sub in list(self._subscribers[event_type]):
            if sub['id'] == subscription_id:
                self._subscribers[event_type].remove(sub)
                logger.debug(f"Unsubscribed from {event_type} with ID {subscription_id}")
                return True
                
        return False
    
    def add_filter(self, event_type: str, filter_func: Callable) -> None:
        """
        添加事件过滤器
        
        Args:
            event_type: 事件类型
            filter_func: 过滤函数，返回True表示允许事件传递
        """
        if event_type not in self._filters:
            self._filters[event_type] = []
        self._filters[event_type].append(filter_func)
    
    async def publish(self, event_type: str, data: Any = None) -> int:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            int: 接收事件的订阅者数量
        """
        if event_type not in self._subscribers:
            return 0
            
        # 记录事件历史
        if self._record_history:
            if event_type not in self._event_history:
                self._event_history[event_type] = []
                
            event_record = {
                'timestamp': time.time(),
                'data': data
            }
            
            self._event_history[event_type].append(event_record)
            
            # 限制历史记录数量
            if len(self._event_history[event_type]) > self._max_history:
                self._event_history[event_type] = self._event_history[event_type][-self._max_history:]
        
        # 应用过滤器
        if event_type in self._filters:
            for filter_func in self._filters[event_type]:
                if not filter_func(data):
                    logger.debug(f"Event {event_type} filtered out")
                    return 0
        
        # 按优先级排序订阅者
        subscribers = sorted(
            self._subscribers[event_type], 
            key=lambda x: x['priority']
        )
        
        # 创建通知任务
        tasks = []
        for sub in subscribers:
            callback = sub['callback']
            
            if asyncio.iscoroutinefunction(callback):
                task = asyncio.create_task(callback(data))
            else:
                # 包装同步函数为异步
                task = asyncio.create_task(asyncio.to_thread(callback, data))
                
            tasks.append(task)
        
        # 等待所有通知完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        return len(tasks)
    
    def get_history(self, event_type: str, count: int = None) -> List[Dict]:
        """
        获取事件历史
        
        Args:
            event_type: 事件类型
            count: 最大返回数量
            
        Returns:
            List[Dict]: 事件历史记录
        """
        if not self._record_history or event_type not in self._event_history:
            return []
            
        history = self._event_history[event_type]
        
        if count is not None:
            history = history[-count:]
            
        return history
    
    def get_subscriber_count(self, event_type: str = None) -> Dict[str, int]:
        """
        获取订阅者数量
        
        Args:
            event_type: 可选的事件类型过滤
            
        Returns:
            Dict[str, int]: 各事件类型的订阅者数量
        """
        if event_type:
            return {event_type: len(self._subscribers.get(event_type, []))}
            
        return {et: len(subs) for et, subs in self._subscribers.items()}