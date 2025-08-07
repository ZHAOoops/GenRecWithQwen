import pickle
import os
import time
import shutil
from typing import Any, Optional, Iterable

class SimpleKVStore:
    """轻量级KV存储系统，用于缓存和索引管理"""
    def __init__(self, storage_dir: str = 'kv_store'):
        self.storage_dir = storage_dir
        self.data = {}
        self.index = {}  # 内存中维护索引，减少IO操作
        self._init_storage()
        self._load()
        
    def _init_storage(self) -> None:
        """初始化存储目录和索引文件"""
        try:
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir, exist_ok=True)
            
            index_path = os.path.join(self.storage_dir, 'index.pkl')
            if not os.path.exists(index_path):
                with open(index_path, 'wb') as f:
                    pickle.dump({}, f)
        except OSError as e:
            raise Exception(f"初始化存储目录失败: {str(e)}")
    
    def _load(self) -> None:
        """加载存储的数据和索引，过滤过期数据"""
        try:
            index_path = os.path.join(self.storage_dir, 'index.pkl')
            if os.path.exists(index_path):
                with open(index_path, 'rb') as f:
                    self.index = pickle.load(f)
                
                # 清理并加载有效数据
                current_time = time.time()
                expired_keys = []
                
                for key, info in self.index.items():
                    # 检查是否过期
                    if info['expiry'] is not None and current_time >= info['expiry']:
                        expired_keys.append(key)
                        continue
                    
                    # 加载未过期的数据
                    data_path = os.path.join(self.storage_dir, f"{key}.pkl")
                    if os.path.exists(data_path):
                        try:
                            with open(data_path, 'rb') as f:
                                self.data[key] = pickle.load(f)
                        except (pickle.UnpicklingError, EOFError):
                            print(f"数据文件损坏，跳过: {key}")
                            expired_keys.append(key)
                
                # 清理过期键
                for key in expired_keys:
                    self.delete(key, update_index=False)
                
                # 保存更新后的索引
                self._save_index()
                
        except Exception as e:
            print(f"加载KV存储失败: {str(e)}")
    
    def _save_index(self) -> None:
        """保存索引到文件"""
        try:
            index_path = os.path.join(self.storage_dir, 'index.pkl')
            with open(index_path, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            print(f"保存索引失败: {str(e)}")
    
    def set(self, key: str, value: Any, expiry: Optional[float] = None) -> None:
        """
        设置键值对
        :param key: 键（字符串类型）
        :param value: 值（可序列化的对象）
        :param expiry: 过期时间（时间戳），None表示永不过期
        """
        if not isinstance(key, str):
            raise ValueError("键必须是字符串类型")
            
        try:
            # 内存中更新数据
            self.data[key] = value
            
            # 保存数据到文件
            data_path = os.path.join(self.storage_dir, f"{key}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(value, f)
                
            # 更新内存索引
            self.index[key] = {
                'timestamp': time.time(),
                'expiry': expiry
            }
            
            # 保存索引
            self._save_index()
            
        except Exception as e:
            print(f"设置键值对失败 ({key}): {str(e)}")
            # 回滚内存数据
            if key in self.data:
                del self.data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取键值对
        :param key: 键
        :param default: 默认值
        :return: 存储的值或默认值
        """
        # 先检查内存中的数据
        if key in self.data:
            # 检查是否过期
            if key in self.index:
                info = self.index[key]
                if info['expiry'] is not None and time.time() >= info['expiry']:
                    self.delete(key)
                    return default
            return self.data[key]
        
        # 内存中没有则检查文件
        data_path = os.path.join(self.storage_dir, f"{key}.pkl")
        if os.path.exists(data_path) and key in self.index:
            info = self.index[key]
            # 检查过期
            if info['expiry'] is not None and time.time() >= info['expiry']:
                self.delete(key)
                return default
                
            # 从文件加载
            try:
                with open(data_path, 'rb') as f:
                    value = pickle.load(f)
                self.data[key] = value  # 加载到内存
                return value
            except (pickle.UnpicklingError, EOFError):
                print(f"数据文件损坏，删除: {key}")
                self.delete(key)
                return default
        
        return default
    
    def delete(self, key: str, update_index: bool = True) -> None:
        """
        删除键值对
        :param key: 键
        :param update_index: 是否更新索引文件
        """
        # 从内存删除
        if key in self.data:
            del self.data[key]
        
        # 从索引删除
        if key in self.index:
            del self.index[key]
            if update_index:
                self._save_index()
        
        # 删除数据文件
        data_path = os.path.join(self.storage_dir, f"{key}.pkl")
        if os.path.exists(data_path):
            try:
                os.remove(data_path)
            except OSError as e:
                print(f"删除数据文件失败 ({key}): {str(e)}")
    
    def keys(self) -> Iterable[str]:
        """返回所有有效键"""
        self.clear_expired()  # 先清理过期键
        return self.data.keys()
    
    def clear_expired(self) -> int:
        """
        清理过期数据
        :return: 清理的过期键数量
        """
        current_time = time.time()
        expired_keys = [
            key for key, info in self.index.items()
            if info['expiry'] is not None and current_time >= info['expiry']
        ]
        
        for key in expired_keys:
            self.delete(key, update_index=False)
        
        # 批量更新索引
        if expired_keys:
            self._save_index()
            
        return len(expired_keys)
    
    def clear_all(self) -> None:
        """清空所有数据"""
        try:
            # 清空内存数据
            self.data.clear()
            self.index.clear()
            
            # 删除所有文件
            if os.path.exists(self.storage_dir):
                for filename in os.listdir(self.storage_dir):
                    file_path = os.path.join(self.storage_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"删除文件失败 ({file_path}): {str(e)}")
            
            # 重建空索引
            self._save_index()
            print("已清空所有KV存储数据")
        except Exception as e:
            print(f"清空存储失败: {str(e)}")


class CacheManager:
    """缓存管理工具，基于KV存储实现"""
    def __init__(self, kv_store: Optional[SimpleKVStore] = None):
        self.kv_store = kv_store if kv_store else SimpleKVStore()
    
    def cache_user_preferences(self, user_id: int, preferences: Any, ttl: int = 86400) -> Any:
        """
        缓存用户偏好
        :param user_id: 用户ID
        :param preferences: 用户偏好数据
        :param ttl: 生存时间（秒），默认1天
        :return: 缓存的偏好数据
        """
        if not isinstance(user_id, int):
            raise ValueError("user_id必须是整数")
            
        expiry = time.time() + ttl
        self.kv_store.set(f"user_prefs:{user_id}", preferences, expiry)
        return preferences
    
    def get_cached_preferences(self, user_id: int) -> Optional[Any]:
        """
        获取缓存的用户偏好
        :param user_id: 用户ID
        :return: 缓存的偏好数据或None
        """
        if not isinstance(user_id, int):
            raise ValueError("user_id必须是整数")
            
        return self.kv_store.get(f"user_prefs:{user_id}")
    
    def cache_recommendations(self, user_id: int, recommendations: Any, ttl: int = 3600) -> Any:
        """
        缓存推荐结果
        :param user_id: 用户ID
        :param recommendations: 推荐结果数据
        :param ttl: 生存时间（秒），默认1小时
        :return: 缓存的推荐结果
        """
        if not isinstance(user_id, int):
            raise ValueError("user_id必须是整数")
            
        expiry = time.time() + ttl
        self.kv_store.set(f"recs:{user_id}", recommendations, expiry)
        return recommendations
    
    def get_cached_recommendations(self, user_id: int) -> Optional[Any]:
        """
        获取缓存的推荐结果
        :param user_id: 用户ID
        :return: 缓存的推荐结果或None
        """
        if not isinstance(user_id, int):
            raise ValueError("user_id必须是整数")
            
        return self.kv_store.get(f"recs:{user_id}")
    
    def clear_user_cache(self, user_id: int) -> bool:
        """
        清除用户的所有缓存
        :param user_id: 用户ID
        :return: 是否清除成功
        """
        if not isinstance(user_id, int):
            raise ValueError("user_id必须是整数")
            
        try:
            self.kv_store.delete(f"user_prefs:{user_id}")
            self.kv_store.delete(f"recs:{user_id}")
            return True
        except Exception as e:
            print(f"清除用户缓存失败 ({user_id}): {str(e)}")
            return False
    
    def clear_expired_cache(self) -> int:
        """
        清理所有过期缓存
        :return: 清理的过期键数量
        """
        return self.kv_store.clear_expired()
    
    def clear_all_cache(self) -> None:
        """清空所有缓存"""
        self.kv_store.clear_all()
