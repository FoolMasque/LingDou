"""
流式会话状态管理器 - 支持 SSE 断连续推与后台异步生成
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, AsyncGenerator
from enum import Enum

from utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamStatus(str, Enum):
    """流式生成状态枚举"""
    PENDING = "pending" ## 初始化分配stream id，等待启动
    STREAMING = "streaming" ## 开始流式输出，后台任务运行中
    COMPLETED = "completed" ## 后台完成回答，完整文本已存入DB
    CANCELLED = "cancelled" ## 主动中断推理，后台任务强制终止
    ERROR = "error" ## 后台任务中报错，例如API超时


@dataclass
class StreamEvent:
    event_id: int
    event_type: str  # "init" | "chunk" | "complete" | "error" | "cancelled"
    content: str = "" # 文本增量内容
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse_data(self) -> str:
        """格式化为 SSE 传输字符串"""
        data = {
            "event_id": self.event_id,
            "type": self.event_type,
            "content": self.content,
            "metadata": self.metadata
        }
        return f"id: {self.event_id}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class StreamSession:
    """单条流会话状态对象"""

    def __init__(self, stream_id: str, conversation_id: str, business_id: str):
        self.stream_id: str = stream_id
        self.conversation_id: str = conversation_id
        self.business_id: str = business_id
        self.status: StreamStatus = StreamStatus.PENDING
        self.events: List[StreamEvent] = []
        self.accumulated_content: str = ""
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.task: Optional[asyncio.Task] = None
        self.subscribers: Set[asyncio.Queue] = set() # 保存所有正在连接看这个回答的客户端队列
        self.event_counter: int = 0
        self.saved_to_db: bool = False
        self.error_message: Optional[str] = None

    def add_event(self, event_type: str, content: str = "", metadata: Optional[Dict[str, Any]] = None) -> StreamEvent:
        """追加事件并向当前监听者广播"""
        self.event_counter += 1
        event = StreamEvent(
            event_id=self.event_counter,
            event_type=event_type,
            content=content,
            metadata=metadata or {},
            timestamp=time.time()
        )
        self.events.append(event)
        self.updated_at = time.time()

        if event_type == "chunk":
            self.accumulated_content += content

        # 广播给活跃的监听队列
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except Exception as e:
                logger.warning(f"推送事件到订阅队列失败 [stream_id={self.stream_id}]: {e}")

        return event

    def subscribe(self) -> asyncio.Queue:
        """注册一个新的订阅队列"""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """解绑订阅队列"""
        self.subscribers.discard(queue)


class StreamManager:
    """流式会话全局管理器（单例模式）"""
    _instance: Optional['StreamManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, ttl_seconds: int = 1800):
        if self._initialized:
            return
        self._sessions: Dict[str, StreamSession] = {}
        self.ttl_seconds: int = ttl_seconds  # 会话状态留存时间（默认30分钟）
        self._initialized = True
        logger.info(f"StreamManager 初始化完成，TTL: {ttl_seconds}s")

    def create_session(self, conversation_id: str, business_id: str, stream_id: Optional[str] = None) -> StreamSession:
        """创建新流会话"""
        if not stream_id:
            stream_id = f"str_{uuid.uuid4().hex[:12]}"
        
        session = StreamSession(
            stream_id=stream_id,
            conversation_id=conversation_id,
            business_id=business_id
        )
        self._sessions[stream_id] = session
        self._cleanup_expired_sessions()
        logger.info(f"创建流式 Session: {stream_id} (conversation_id={conversation_id})")
        return session

    def get_session(self, stream_id: str) -> Optional[StreamSession]:
        """获取流会话"""
        return self._sessions.get(stream_id)

    async def start_background_generation(
            self,
            session: StreamSession,
            result_stream: AsyncGenerator[Any, None],
            conversation_manager: Any,
            user_images_count: int = 0
    ) -> asyncio.Task:
        """在后台启动 LLM 生成任务，即使前端断开连接也不停止"""

        async def _generation_task():
            session.status = StreamStatus.STREAMING
            start_time = time.time()
            chunk_count = 0
            
            # 发送初始事件
            session.add_event(
                event_type="init",
                metadata={
                    "stream_id": session.stream_id,
                    "conversation_id": session.conversation_id
                }
            )

            try:
                async for chunk in result_stream:
                    chunk_count += 1
                    if isinstance(chunk, dict):
                        # 处理带元数据的全量/特殊结构 chunk
                        content = chunk.get("content", "")
                        session.add_event("chunk", content=content, metadata=chunk.get("metadata", {}))
                    else:
                        # 增量文本 chunk
                        session.add_event("chunk", content=chunk)
                    
                    # 极小让度，保障异步任务调度流畅
                    await asyncio.sleep(0.001)

                processed_time = time.time() - start_time
                session.status = StreamStatus.COMPLETED

                # 发送完成事件
                session.add_event(
                    event_type="complete",
                    content=session.accumulated_content,
                    metadata={
                        "processing_time": round(processed_time, 2),
                        "chunk_count": chunk_count,
                        "conversation_id": session.conversation_id,
                        "user_images_count": user_images_count
                    }
                )
                logger.info(f"后台流生成完成 [stream_id={session.stream_id}], 耗时: {processed_time:.2f}s, 字符数: {len(session.accumulated_content)}")

            except asyncio.CancelledError:
                session.status = StreamStatus.CANCELLED
                logger.warning(f"后台流生成被显式取消 [stream_id={session.stream_id}]")
                session.add_event("cancelled", content=session.accumulated_content, metadata={"reason": "user_cancelled"})
                raise
            except Exception as e:
                session.status = StreamStatus.ERROR
                session.error_message = str(e)
                logger.error(f"后台流生成异常 [stream_id={session.stream_id}]: {e}", exc_info=True)
                session.add_event("error", content=str(e), metadata={"conversation_id": session.conversation_id})
            finally:
                # 无论成功、取消还是报错，只要生成了有效文本且尚未落库，就落库到 ConversationManager
                if session.accumulated_content and not session.saved_to_db and conversation_manager:
                    try:
                        from utils.url_helper import post_process_response_urls
                        final_text = post_process_response_urls(session.accumulated_content)
                    except Exception as ex:
                        logger.warning(f"URL 替换失败: {ex}")
                        final_text = session.accumulated_content

                    try:
                        await conversation_manager.add_message(
                            session.conversation_id,
                            role="assistant",
                            content=final_text
                        )
                        session.saved_to_db = True
                        logger.info(f"会话消息成功持久化到数据库 [conversation_id={session.conversation_id}]")
                    except Exception as ex:
                        logger.error(f"持久化保存会话消息失败 [conversation_id={session.conversation_id}]: {ex}")

        task = asyncio.create_task(_generation_task())
        session.task = task
        return task

    async def subscribe_stream(
            self,
            stream_id: str,
            last_event_id: int = 0
    ) -> AsyncGenerator[str, None]:
        """
        订阅 SSE 事件流
        支持断续追查：优先补发 (last_event_id, current_counter] 历史 chunks，
        然后再监听新事件。
        """
        session = self.get_session(stream_id)
        if not session:
            # 会话不存在或已过期
            err_event = StreamEvent(event_id=0, event_type="error", content=f"Stream session {stream_id} not found or expired")
            yield err_event.to_sse_data()
            return

        # 1. 补发历史断点事件 (Catch-up phase)
        missed_events = [e for e in session.events if e.event_id > last_event_id]
        for event in missed_events:
            yield event.to_sse_data()

        # 2. 如果在追赶阶段 Session 已经终止（已完成/被取消/出错），发送 done 信号并退出
        if session.status in (StreamStatus.COMPLETED, StreamStatus.CANCELLED, StreamStatus.ERROR):
            yield f"data: {json.dumps({'type': 'done', 'status': session.status.value})}\n\n"
            return

        # 3. 接入实时订阅 (Live phase)
        queue = session.subscribe()
        try:
            while True:
                # 阻塞获取新事件，设置超时以检查任务状态
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield event.to_sse_data()

                    if event.event_type in ("complete", "error", "cancelled"):
                        yield f"data: {json.dumps({'type': 'done', 'status': session.status.value})}\n\n"
                        break
                except asyncio.TimeoutError:
                    # 检查后台 task 是否已经结束
                    if session.status in (StreamStatus.COMPLETED, StreamStatus.CANCELLED, StreamStatus.ERROR):
                        yield f"data: {json.dumps({'type': 'done', 'status': session.status.value})}\n\n"
                        break
                    else:
                        # 正在生成或 RAG 思考检索中：发送 SSE 标准心跳注释 (Keep-Alive)，防止 Nginx/网关超时断开
                        yield ": ping\n\n"
        finally:
            session.unsubscribe(queue)

    def cancel_stream(self, stream_id: str) -> bool:
        """显式取消后台 Task"""
        session = self.get_session(stream_id)
        if not session:
            return False

        if session.task and not session.task.done():
            logger.info(f"显式取消流式后台 Task [stream_id={stream_id}]")
            session.task.cancel()
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """主动清理过期 Session，返回清理的数量"""
        now = time.time()
        expired_keys = [
            sid for sid, sess in list(self._sessions.items())
            if now - sess.updated_at > self.ttl_seconds and sess.status != StreamStatus.STREAMING
        ]
        for sid in expired_keys:
            self._sessions.pop(sid, None)
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期的流式 Session")
        return len(expired_keys)

    def _cleanup_expired_sessions(self):
        self.cleanup_expired_sessions()
