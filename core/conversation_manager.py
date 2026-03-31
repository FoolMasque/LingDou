"""
会话管理系统 - 支持多轮对话的完整实现
"""
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import redis.asyncio as aioredis
import pickle
from dataclasses import dataclass, asdict, field
from enum import Enum

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StorageBackend(Enum):
    """存储后端类型"""
    MEMORY = "memory"  # 内存存储（开发测试用）
    FILE = "file"  # 文件存储（单机部署）
    REDIS = "redis"  # Redis存储（生产环境）


@dataclass
class Message:
    """单条消息"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    images: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "images": self.images,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """从字典创建"""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Conversation:
    """会话对象"""
    id: str
    business_id: str
    user_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 会话状态
    active: bool = True
    turn_count: int = 0  # 对话轮数

    def add_message(self, role: str, content: str, images: Optional[List[str]] = None):
        """添加消息"""
        msg = Message(
            role=role,
            content=content,
            images=images
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()

        if role == "user":
            self.turn_count += 1

        return msg

    def get_context_window(self, max_turns: int = 5, max_tokens: int = 2000) -> List[Message]:
        """获取上下文窗口中的消息

        Args:
            max_turns: 最大轮数（一轮=user+assistant）
            max_tokens: 最大token数（粗略估算）

        Returns:
            满足条件的历史消息列表
        """
        if not self.messages:
            return []

        # 从最新的消息开始，向前收集
        result = []
        token_count = 0
        turn_count = 0

        # 临时缓存，用来保存成对的 user+assistant
        pair_buffer = []

        for msg in reversed(self.messages):
            # 粗略估算token数（中文约1.5字符=1token）
            estimated_tokens = int(len(msg.content) // 1.5)

            if token_count + estimated_tokens > max_tokens:
                break

            pair_buffer.insert(0, msg)
            token_count += estimated_tokens

            if msg.role == "user":
                turn_count += 1
                # 达到轮次限制则停止
                if turn_count >= max_turns:
                    break
                # 将这一对加入结果
                result = pair_buffer + result
                pair_buffer = []

                # 如果没满一轮但有剩余，也加进去
        if pair_buffer:
            result = pair_buffer + result

        return result

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "business_id": self.business_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "active": self.active,
            "turn_count": self.turn_count
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Conversation':
        """从字典创建"""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["messages"] = [Message.from_dict(m) for m in data["messages"]]
        return cls(**data)


class ConversationStorage:
    """会话存储基类"""

    async def save(self, conversation: Conversation) -> bool:
        """保存会话"""
        raise NotImplementedError

    async def load(self, conversation_id: str) -> Optional[Conversation]:
        """加载会话"""
        raise NotImplementedError

    async def delete(self, conversation_id: str) -> bool:
        """删除会话"""
        raise NotImplementedError

    async def list_conversations(
            self,
            business_id: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Conversation]:
        """列出会话"""
        raise NotImplementedError

    async def cleanup_old_conversations(self, days: int = 7) -> int:
        """清理旧会话"""
        raise NotImplementedError


class MemoryStorage(ConversationStorage):
    """内存存储实现"""

    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}

    async def save(self, conversation: Conversation) -> bool:
        self._conversations[conversation.id] = conversation
        return True

    async def load(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    async def delete(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    async def list_conversations(
            self,
            business_id: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Conversation]:
        result = []
        for conv in self._conversations.values():
            if business_id and conv.business_id != business_id:
                continue
            if user_id and conv.user_id != user_id:
                continue
            result.append(conv)

        # 按更新时间排序
        result.sort(key=lambda x: x.updated_at, reverse=True)
        return result[:limit]

    async def cleanup_old_conversations(self, days: int = 7) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        to_delete = []

        for conv_id, conv in self._conversations.items():
            if conv.updated_at < cutoff:
                to_delete.append(conv_id)

        for conv_id in to_delete:
            del self._conversations[conv_id]

        return len(to_delete)

from asyncio import Lock
class FileStorage(ConversationStorage):
    """文件存储实现"""

    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _get_path(self, conversation_id: str) -> Path:
        """获取会话文件路径"""
        return self.storage_dir / f"{conversation_id}.json"

    async def save(self, conversation: Conversation) -> bool:
        async with self._lock:
            try:
                path = self._get_path(conversation.id)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                logger.error(f"保存会话失败: {e}")
                return False

    async def load(self, conversation_id: str) -> Optional[Conversation]:
        try:
            path = self._get_path(conversation_id)
            if not path.exists():
                return None

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"加载会话失败: {e}")
            return None

    async def delete(self, conversation_id: str) -> bool:
        try:
            path = self._get_path(conversation_id)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False

    async def list_conversations(
            self,
            business_id: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Conversation]:
        result = []

        for path in self.storage_dir.glob("*.json"):
            try:
                conv = await self.load(path.stem)
                if conv:
                    if business_id and conv.business_id != business_id:
                        continue
                    if user_id and conv.user_id != user_id:
                        continue
                    result.append(conv)
            except:
                continue

        result.sort(key=lambda x: x.updated_at, reverse=True)
        return result[:limit]

    async def cleanup_old_conversations(self, days: int = 7) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0

        for path in self.storage_dir.glob("*.json"):
            try:
                conv = await self.load(path.stem)
                if conv and conv.updated_at < cutoff:
                    await self.delete(conv.id)
                    deleted += 1
            except:
                continue

        return deleted


class RedisStorage(ConversationStorage):
    _redis_pool = None  # 类级别共享连接池
    """Redis存储实现"""
    def __init__(self, redis_url: str = settings.conversation.redis_url): # "redis://localhost:6379"
        self.redis_url = redis_url
        self.redis = None
        self.key_prefix = "conversation:"

    async def _ensure_connected(self):
        """确保Redis连接"""
        if not RedisStorage._redis_pool:
            RedisStorage._redis_pool = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
        self.redis = RedisStorage._redis_pool

    def _get_key(self, conversation_id: str) -> str:
        """获取Redis键"""
        return f"{self.key_prefix}{conversation_id}"

    async def save(self, conversation: Conversation) -> bool:
        try:
            await self._ensure_connected()
            key = self._get_key(conversation.id)
            data = pickle.dumps(conversation)

            # 设置过期时间（默认30天）
            await self.redis.setex(key, settings.conversation.redis_ttl, data) # 30 * 24 * 3600=2592000

            # 维护索引
            await self._update_index(conversation)

            return True
        except Exception as e:
            logger.error(f"Redis保存失败: {e}")
            return False

    async def load(self, conversation_id: str) -> Optional[Conversation]:
        try:
            await self._ensure_connected()
            key = self._get_key(conversation_id)
            data = await self.redis.get(key)

            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis加载失败: {e}")
            return None

    async def delete(self, conversation_id: str) -> bool:
        try:
            await self._ensure_connected()
            key = self._get_key(conversation_id)

            # 删除会话
            result = await self.redis.delete(key)

            # 更新索引
            await self._remove_from_index(conversation_id)

            return result > 0
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False

    async def _update_index(self, conversation: Conversation):
        """更新索引"""
        # 按业务ID索引
        if conversation.business_id:
            index_key = f"index:business:{conversation.business_id}"
            await self.redis.zadd(
                index_key,
                {conversation.id: conversation.updated_at.timestamp()}
            )

        # 按用户ID索引
        if conversation.user_id:
            index_key = f"index:user:{conversation.user_id}"
            await self.redis.zadd(
                index_key,
                {conversation.id: conversation.updated_at.timestamp()}
            )

    async def _remove_from_index(self, conversation_id: str):
        """从索引中移除"""
        # 这里简化处理，实际应该先查询会话获取business_id和user_id
        try:
            # 遍历所有索引键删除（可以优化为根据 metadata 保存索引键）
            async for key in self.redis.scan_iter(match="index:*"):
                await self.redis.zrem(key, conversation_id)
        except Exception as e:
            logger.warning(f"从索引移除失败: {e}")

    async def list_conversations(
            self,
            business_id: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Conversation]:
        try:
            await self._ensure_connected()

            # 确定使用哪个索引
            if business_id:
                index_key = f"index:business:{business_id}"
            elif user_id:
                index_key = f"index:user:{user_id}"
            else:
                # 如果都没有，返回空列表
                return []

            # 从索引获取最新的会话ID
            conv_ids = await self.redis.zrevrange(index_key, 0, limit - 1)

            # 加载会话
            result = []
            for conv_id in conv_ids:
                conv = await self.load(conv_id.decode() if isinstance(conv_id, bytes) else conv_id)
                if conv:
                    result.append(conv)

            return result
        except Exception as e:
            logger.error(f"列出会话失败: {e}")
            return []

    async def cleanup_old_conversations(self, days: int = 7) -> int:
        # Redis通过过期时间自动清理
        return 0


class ConversationManager:
    """会话管理器"""

    def __init__(
            self,
            storage_backend: StorageBackend = StorageBackend.FILE,
            storage_config: Optional[Dict[str, Any]] = None
    ):
        """初始化会话管理器

        Args:
            storage_backend: 存储后端类型
            storage_config: 存储配置
        """
        self.storage_backend = storage_backend
        self.storage = self._create_storage(storage_backend, storage_config or {})

        # 缓存最近使用的会话
        self._cache: Dict[str, Conversation] = {}
        self._cache_size = 100

        logger.info(f"会话管理器初始化完成，使用存储后端: {storage_backend.value}")

    def _create_storage(
            self,
            backend: StorageBackend,
            config: Dict[str, Any]
    ) -> ConversationStorage:
        """创建存储实例"""
        if backend == StorageBackend.MEMORY:
            return MemoryStorage()
        elif backend == StorageBackend.FILE:
            storage_dir = config.get("storage_dir", "conversations")
            return FileStorage(storage_dir)
        elif backend == StorageBackend.REDIS:
            redis_url = config.get("redis_url", "redis://localhost:6379")
            return RedisStorage(redis_url)
        else:
            raise ValueError(f"不支持的存储后端: {backend}")

    async def create_conversation(
            self,
            business_id: str,
            user_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """创建新会话"""
        conversation = Conversation(
            id=str(uuid.uuid4()),
            business_id=business_id,
            user_id=user_id,
            metadata=metadata or {}
        )

        await self.storage.save(conversation)
        self._cache[conversation.id] = conversation

        logger.info(f"创建新会话: {conversation.id} (业务: {business_id})")
        return conversation

    async def get_or_create_conversation(
            self,
            conversation_id: Optional[str],
            business_id: str,
            user_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """获取或创建会话"""
        if conversation_id:
            conversation = await self.get_conversation(conversation_id)
            if conversation:
                # 顺便检查一下：如果用户发起了新一轮 query 但又携带了新设定的 metadata，可以顺便热更新
                if metadata:
                    await self.update_metadata(conversation.id, metadata)
                return conversation

        return await self.create_conversation(business_id, user_id, metadata)

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取会话"""
        # 先检查缓存
        if conversation_id in self._cache:
            return self._cache[conversation_id]

        # 从存储加载
        conversation = await self.storage.load(conversation_id)

        if conversation:
            # 更新缓存
            self._update_cache(conversation)

        return conversation

    async def add_message(
            self,
            conversation_id: str,
            role: str,
            content: str,
            images: Optional[List[str]] = None
    ) -> Tuple[Conversation, Message]:
        """添加消息到会话"""
        conversation = await self.get_conversation(conversation_id)

        if not conversation:
            raise ValueError(f"会话不存在: {conversation_id}")

        # 添加消息
        message = conversation.add_message(role, content, images)

        # 保存更新
        await self.storage.save(conversation)

        logger.debug(f"添加消息到会话 {conversation_id}: {role}")

        return conversation, message

    async def get_context_for_query(
            self,
            conversation_id: Optional[str],
            max_turns: int = settings.conversation.default_max_turns,
            max_tokens: int = 2000,
            format_type: str = "lightrag"
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """获取查询的上下文

        Args:
            conversation_id: 会话ID
            max_turns: 最大轮数 默认5
            max_tokens: 最大token数
            format_type: 格式化类型 ("lightrag" or "simple")

        Returns:
            (formatted_context, message_list)
        """
        if not conversation_id:
            return None, []

        conversation = await self.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return None, []

        # 获取上下文窗口
        context_messages = conversation.get_context_window(max_turns, max_tokens)

        if not context_messages:
            return None, []

        # 转换为消息列表
        message_list = []
        for msg in context_messages:
            message_list.append({
                "role": msg.role,
                "content": msg.content
            })

        # 格式化上下文
        if format_type == "lightrag":
            # LightRAG格式：将历史对话整合为上下文描述
            formatted = self._format_for_lightrag(context_messages)
        else:
            # 简单格式：直接拼接
            formatted = self._format_simple(context_messages)

        return formatted, message_list

    def _format_for_lightrag(self, messages: List[Message]) -> str:
        """为LightRAG格式化上下文"""
        if not messages:
            return ""

        context_parts = ["基于以下对话历史：\n"]

        for i, msg in enumerate(messages):
            if msg.role == "user":
                context_parts.append(f"user: {msg.content}")
            else:
                context_parts.append(f"assistant: {msg.content}")

        context_parts.append("\n请基于上述对话历史，回答用户的新问题。")

        return "\n".join(context_parts)

    def _format_simple(self, messages: List[Message]) -> str:
        """简单格式化上下文"""
        parts = []
        for msg in messages:
            role_name = "user" if msg.role == "user" else "assistant"
            parts.append(f"{role_name}: {msg.content}")

        return "\n".join(parts)

    def _update_cache(self, conversation: Conversation):
        """更新缓存"""
        # 简单的LRU缓存策略
        if len(self._cache) >= self._cache_size:
            # 移除最旧的
            oldest = min(self._cache.items(), key=lambda x: x[1].updated_at)
            del self._cache[oldest[0]]

        self._cache[conversation.id] = conversation

    async def update_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        """更新会话的元数据（例如更新人物设定）"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False
            
        if not conversation.metadata:
            conversation.metadata = {}
            
        # 增量更新
        conversation.metadata.update(metadata)
        
        # 保存并更新缓存
        await self.storage.save(conversation)
        self._update_cache(conversation)
        return True

    async def list_conversations(
            self,
            business_id: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Conversation]:
        """列出会话"""
        return await self.storage.list_conversations(
            business_id=business_id,
            user_id=user_id,
            limit=limit
        )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """删除会话"""
        # 从缓存移除
        if conversation_id in self._cache:
            del self._cache[conversation_id]

        # 从存储删除
        result = await self.storage.delete(conversation_id)

        if result:
            logger.info(f"删除会话: {conversation_id}")

        return result

    async def cleanup_old_conversations(self, days: int = 7) -> int:
        """清理旧会话"""
        count = await self.storage.cleanup_old_conversations(days)
        logger.info(f"清理了 {count} 个旧会话（{days} 天前）")
        return count

    async def export_conversation(
            self,
            conversation_id: str,
            eformat: str = "json"
    ) -> Optional[str]:
        """导出会话"""
        conversation = await self.get_conversation(conversation_id)

        if not conversation:
            return None

        if eformat == "json":
            return json.dumps(conversation.to_dict(), ensure_ascii=False, indent=2)
        elif eformat == "text":
            lines = [
                f"会话ID: {conversation.id}",
                f"业务: {conversation.business_id}",
                f"创建时间: {conversation.created_at}",
                f"消息数: {len(conversation.messages)}",
                "=" * 50
            ]

            for msg in conversation.messages:
                role_name = "user" if msg.role == "user" else "assistant"
                lines.append(f"\n[{role_name}] {msg.timestamp}")
                lines.append(msg.content)
                if msg.images:
                    lines.append(f"附带图片: {', '.join(msg.images)}")

            return "\n".join(lines)

        return None