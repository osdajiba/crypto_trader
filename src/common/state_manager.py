#!/usr/bin/env python3
# src/common/state_manager.py

import os
import json
import asyncio
import time
from typing import Dict, Any, Optional, Set
from datetime import datetime
import logging

from src.common.log_manager import LogManager

logger = LogManager.get_logger("common.state_manager")

class StateManager:
    """
    状态管理器，用于系统状态保存和恢复
    
    特点:
    - 自动周期性保存
    - 支持增量更新
    - 支持状态版本管理
    - 支持状态回滚
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, state_dir: str = './state'):
        """获取状态管理器单例"""
        if cls._instance is None:
            cls._instance = cls(state_dir)
        return cls._instance
    
    def __init__(self, state_dir: str = './state'):
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, 'system_state.json')
        self.backup_dir = os.path.join(state_dir, 'backups')
        
        # 确保目录存在
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # 状态数据
        self.state = {}
        self._dirty_keys = set()  # 修改过的键
        self._backup_count = 5    # 保留备份数量
        self._save_interval = 60.0  # 保存间隔(秒)
        self._last_save_time = 0
        self._save_task = None
        self._lock = asyncio.Lock()
        self._closed = False
        
    async def start(self):
        """启动状态管理器并加载状态"""
        await self.load()
        self._save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("State manager started")
        
    async def _auto_save_loop(self):
        """自动保存循环"""
        try:
            while not self._closed:
                await asyncio.sleep(self._save_interval)
                if self._dirty_keys or time.time() - self._last_save_time > 300:
                    await self.save()
        except asyncio.CancelledError:
            logger.debug("Auto save task cancelled")
        except Exception as e:
            logger.error(f"Error in auto save loop: {e}")
    
    async def load(self) -> bool:
        """
        加载状态
        
        Returns:
            bool: 是否成功加载状态
        """
        async with self._lock:
            try:
                if os.path.exists(self.state_file):
                    with open(self.state_file, 'r') as f:
                        self.state = json.load(f)
                    self._last_save_time = time.time()
                    logger.info(f"State loaded from {self.state_file}")
                    return True
                else:
                    logger.info("No state file found, starting with empty state")
                    self.state = {}
                    return False
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                self.state = {}
                return False
    
    async def save(self, force: bool = False) -> bool:
        """
        保存状态
        
        Args:
            force: 是否强制保存
            
        Returns:
            bool: 是否成功保存状态
        """
        async with self._lock:
            # 如果没有修改且非强制保存，跳过
            if not self._dirty_keys and not force:
                return True
                
            try:
                # 创建备份
                if os.path.exists(self.state_file):
                    await self._create_backup()
                
                # 保存状态
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=2)
                    
                self._last_save_time = time.time()
                self._dirty_keys.clear()
                logger.debug(f"State saved to {self.state_file}")
                return True
            except Exception as e:
                logger.error(f"Error saving state: {e}")
                return False
    
    async def _create_backup(self):
        """创建状态备份"""
        try:
            # 生成备份文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(self.backup_dir, f"state_{timestamp}.json")
            
            # 复制当前状态文件
            with open(self.state_file, 'r') as src:
                with open(backup_file, 'w') as dst:
                    dst.write(src.read())
                    
            # 清理过期备份
            await self._cleanup_backups()
            
            return backup_file
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    async def _cleanup_backups(self):
        """清理过期备份，只保留最新的几个"""
        try:
            backups = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("state_") and filename.endswith(".json"):
                    backups.append(os.path.join(self.backup_dir, filename))
                    
            # 按修改时间排序
            backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 删除多余的备份
            for old_backup in backups[self._backup_count:]:
                os.remove(old_backup)
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    def set(self, key: str, value: Any) -> None:
        """
        设置状态值
        
        Args:
            key: 键
            value: 值
        """
        # 深层键支持 (dot notation)
        if '.' in key:
            parts = key.split('.')
            curr = self.state
            
            # 遍历路径
            for part in parts[:-1]:
                if part not in curr:
                    curr[part] = {}
                elif not isinstance(curr[part], dict):
                    curr[part] = {}
                curr = curr[part]
                
            # 设置最终值
            curr[parts[-1]] = value
        else:
            self.state[key] = value
            
        self._dirty_keys.add(key.split('.')[0])
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取状态值
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            Any: 状态值
        """
        # 深层键支持 (dot notation)
        if '.' in key:
            parts = key.split('.')
            curr = self.state
            
            # 遍历路径
            for part in parts:
                if part not in curr:
                    return default
                curr = curr[part]
                
            return curr
        else:
            return self.state.get(key, default)
    
    def delete(self, key: str) -> bool:
        """
        删除状态值
        
        Args:
            key: 键
            
        Returns:
            bool: 是否删除成功
        """
        # 深层键支持 (dot notation)
        if '.' in key:
            parts = key.split('.')
            curr = self.state
            
            # 遍历路径
            for part in parts[:-1]:
                if part not in curr:
                    return False
                curr = curr[part]
                
            # 删除最终值
            if parts[-1] in curr:
                del curr[parts[-1]]
                self._dirty_keys.add(parts[0])
                return True
            return False
        else:
            if key in self.state:
                del self.state[key]
                self._dirty_keys.add(key)
                return True
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有状态
        
        Returns:
            Dict[str, Any]: 完整状态
        """
        return self.state.copy()
    
    def set_many(self, values: Dict[str, Any]) -> None:
        """
        设置多个状态值
        
        Args:
            values: 键值对字典
        """
        for key, value in values.items():
            self.set(key, value)
    
    def clear(self, save: bool = True) -> None:
        """
        清空状态
        
        Args:
            save: 是否保存清空后的状态
        """
        self.state.clear()
        for key in list(self._dirty_keys):
            self._dirty_keys.add(key)
            
        if save:
            asyncio.create_task(self.save(force=True))
    
    async def restore_backup(self, backup_file: Optional[str] = None) -> bool:
        """
        从备份恢复
        
        Args:
            backup_file: 备份文件路径，默认使用最新备份
            
        Returns:
            bool: 是否成功恢复
        """
        async with self._lock:
            # 如果没有指定备份文件，使用最新的
            if backup_file is None:
                backups = []
                for filename in os.listdir(self.backup_dir):
                    if filename.startswith("state_") and filename.endswith(".json"):
                        backups.append(os.path.join(self.backup_dir, filename))
                        
                if not backups:
                    logger.error("No backup files found")
                    return False
                    
                backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                backup_file = backups[0]
            
            # 验证备份文件
            if not os.path.exists(backup_file):
                logger.error(f"Backup file not found: {backup_file}")
                return False
                
            try:
                # 加载备份
                with open(backup_file, 'r') as f:
                    self.state = json.load(f)
                    
                # 立即保存
                await self.save(force=True)
                logger.info(f"State restored from backup: {backup_file}")
                return True
            except Exception as e:
                logger.error(f"Error restoring from backup: {e}")
                return False
    
    async def close(self) -> None:
        """关闭状态管理器"""
        if self._closed:
            return
            
        self._closed = True
        
        # 取消自动保存任务
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            
        # 保存最终状态
        await self.save(force=True)
        
        logger.info("State manager closed")