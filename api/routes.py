# api/routes.py
"""
API路由
"""
import base64
import json
import asyncio
import io
import os
import time
from typing import AsyncGenerator, Optional

import aiohttp
from PIL import Image
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from pathlib import Path

from api.models import ProcessRequest, QueryRequest, QueryResponse
from core.image_processor import ImageProcessor  # 抽取图片处理逻辑
from core.conversation_manager import ConversationManager, StorageBackend
from api.models import ChatMessage, ConversationCreate, ConversationListRequest
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

class Dependencies:
    """路由依赖管理"""
    core_system = None
    image_processor = None
    conversation_manager = None

    @classmethod
    def get_core_system(cls):
        if not cls.core_system:
            raise HTTPException(status_code=500, detail="系统未初始化")
        return cls.core_system

    @classmethod
    def get_image_processor(cls):
        if not cls.image_processor:
            cls.image_processor = ImageProcessor()
        return cls.image_processor

    @classmethod
    def get_conversation_manager(cls) -> ConversationManager:
        """获取会话管理器"""
        if not cls.conversation_manager:
            cls.conversation_manager = ConversationManager()
        return cls.conversation_manager

    @classmethod
    def init(cls, core_system):  # 修改：添加会话管理器初始化
        """初始化依赖"""
        cls.core_system = core_system

        # 初始化会话管理器
        storage_backend = os.getenv("CONVERSATION_STORAGE", "file")
        storage_config = {
            "storage_dir": os.getenv("CONVERSATION_DIR", "conversations")
        }

        cls.conversation_manager = ConversationManager(
            storage_backend=StorageBackend(storage_backend),
            storage_config=storage_config
        )

        logger.info(f"会话管理器初始化完成: {storage_backend}")

@router.post("/process_data")
async def process_data(request: ProcessRequest, background_tasks: BackgroundTasks):
    """处理数据API"""
    try:
        if not Path(request.json_file).exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {request.json_file}")
        core_system = Dependencies.get_core_system()

        if not core_system:
            raise HTTPException(status_code=500, detail="系统未初始化")

        background_tasks.add_task(
            core_system.process_crawler_data,
            request.business_id,
            request.json_file
        )

        logger.info(f"启动数据处理任务: {request.business_id}")
        return {"success": True, "message": "数据处理任务已启动"}

    except Exception as e:
        logger.error(f"处理数据请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest,
                core_system=Depends(Dependencies.get_core_system),
                conversation_manager=Depends(Dependencies.get_conversation_manager)):
    """
    统一查询接口

    Args:
        request: 查询请求
        - streaming=False: 返回JSON格式完整结果
        - streaming=True: 返回SSE流式结果

    Returns:
        QueryResponse或StreamingResponse
    """
    start_time = time.time()
    try:
        # 获取RAG实例
        if request.business_id not in core_system.rag_instances:
            await core_system.create_rag_instance(request.business_id)

        # rag_instance = Dependencies.core_system.rag_instances[request.business_id]

        # 会话管理逻辑
        conversation = None

        if request.conversation_id or request.max_history > 0:
            # 如果提供了会话ID或需要历史上下文
            conversation = await conversation_manager.get_or_create_conversation(
                conversation_id=request.conversation_id,
                business_id=request.business_id
            )

            # 添加用户消息
            await conversation_manager.add_message(
                conversation.id,
                "user",
                request.query,
                request.image_base64_list
            )

            # 获取上下文
            _, message_list  = await conversation_manager.get_context_for_query(
                conversation.id,
                max_turns=request.max_history,
                format_type="lightrag"
            )

        if request.streaming:
            return await _handle_streaming_query(request, core_system, conversation_manager, conversation.id, history=message_list)
        else:
            result =  await _handle_blocking_query(request, core_system, conversation_manager, conversation.id,history=message_list)

            # 添加回复到历史
            await conversation_manager.add_message(
                conversation.id,
                role="assistant",
                content=result.result
            )

            result.conversation_id = conversation.id

            start_time = time.time()
            result = await core_system.query(request.business_id, request.query, request.mode)
            processing_time = time.time() - start_time
            import re
            pattern = r'http://[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)'
            urls = re.findall(pattern, result)
            images = list({url for url in urls if url})
            return QueryResponse(
                success=True,
                query=request.query,
                result=result,
                images=images,
                processing_time=round(processing_time, 2)
            )
            # return result

    except Exception as e:
        logger.error(f"查询失败: {e}",exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def _handle_blocking_query(request: QueryRequest,
                                 core_system,
                                 conversation_manager: ConversationManager,
                                 conversation_id: str,
                                 history: list
                                 ) -> QueryResponse:
    """
        处理非流式查询（一次性返回完整结果）

        调用链路：
        routes.query() → core_system.query/query_multimodal()
                      → rag.aquery/aquery_multimodal()
                      → LightRAG + VLM
        """
    start_time = time.time()

    # 处理图片
    image_processor = Dependencies.get_image_processor()
    user_images_base64 = await image_processor.process_user_images(
        image_urls=request.image_urls,
        image_base64_list=request.image_base64_list
    )

    # 执行查询
    if user_images_base64:
        # 多模态查询
        logger.info(f"执行多模态查询: 用户图片 {len(user_images_base64)} 张")
        result_data = await core_system.query_multimodal(
            business_id=request.business_id,
            query=request.query,
            user_images=user_images_base64,
            history=history,
            mode=request.mode
        )
        result = result_data["result"]
        library_images_count = result_data.get("library_images_count", 0)
    else:
        # 纯文本查询
        logger.info("执行纯文本查询")
        result = await core_system.query(
            business_id=request.business_id,
            query=request.query,
            history=history,
            mode=request.mode
        )
        library_images_count = 0

    processing_time = time.time() - start_time

    logger.info(f"查询完成，耗时: {processing_time:.2f}s")

    return QueryResponse(
        success=True,
        query=request.query,
        result=result,
        processing_time=round(processing_time, 2),
        conversation_id=conversation_id,
        user_images_count=len(user_images_base64),
        library_images_count=library_images_count
    )

async def _handle_streaming_query(request: QueryRequest,
                                  core_system,
                                  conversation_manager: ConversationManager,
                                  conversation_id: str,
                                  history: list
                                  ):
    """
    处理流式查询（实时返回生成内容）

    调用链路：
    routes.query() → core_system.query_stream/query_multimodal_stream()
                  → rag.aquery_stream/aquery_multimodal_stream()
                  → LightRAG + VLM (stream=True)
    """

    async def generate_stream():
        """生成SSE流"""
        try:
            start_time = time.time()
            image_processor = Dependencies.get_image_processor()
            user_images_base64 = await image_processor.process_user_images(
                image_urls=request.image_urls,
                image_base64_list=request.image_base64_list
            )

            if user_images_base64:
                # 多模态流式查询
                logger.info(f"执行流式多模态查询: 用户图片 {len(user_images_base64)} 张")
                result_stream = core_system.query_multimodal_stream(
                    business_id=request.business_id,
                    query=request.query,
                    user_images=user_images_base64,
                    history=history,
                    mode=request.mode
                )
            else:
                # 纯文本流式查询
                logger.info("执行流式纯文本查询")
                result_stream = core_system.query_stream(
                    business_id=request.business_id,
                    query=request.query,
                    history=history,
                    mode=request.mode
                )

            # 逐块发送数据
            accumulated_content = ""
            chunk_count = 0

            async for chunk in result_stream:
                chunk_count += 1

                if isinstance(chunk, dict):
                    # 完整消息（带元数据）
                    accumulated_content = chunk.get("content", "")

                    data = {
                        "type": "complete",
                        "content": accumulated_content,
                        "conversation_id": conversation_id,
                        "metadata": {
                            "user_images_count": len(user_images_base64), # chunk.get("user_images_count", 0),
                            "library_images_count": chunk.get("library_images_count", 0),
                            "processing_time": time.time() - start_time,
                            "chunk_count": chunk_count
                        }
                    }
                else:
                    # 流式内容块
                    accumulated_content += chunk

                    data = {
                        "type": "chunk",
                        "content": chunk,
                        "accumulated": accumulated_content,
                        "conversation_id": conversation_id
                    }

                # 发送SSE格式数据
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 小延迟，让前端有时间渲染
                await asyncio.sleep(0.01)

            await conversation_manager.add_message(
                conversation_id,
                role="assistant",
                content=accumulated_content
            )
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            logger.info(f"流式查询完成，共 {chunk_count} 个chunk")

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            error_data = {
                "type": "error",
                "error": str(e),
                "conversation_id": conversation_id
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.get("/status/{business_id}")
async def get_status(business_id: str):
    """获取状态API"""
    try:
        core_system = Dependencies.get_core_system()
        if not core_system:
            raise HTTPException(status_code=500, detail="系统未初始化")

        return core_system.get_business_status(business_id)

    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system_info")
async def system_info():
    """系统信息API"""
    from config.settings import settings

    return {
        "provider": settings.provider,
        "models": {
            "llm": settings.llm_model,
            "vision": settings.vision_model,
            "embedding": settings.embedding_model
        },
        "chinese_prompts": settings.use_chinese_prompts,
        "static_base_url": settings.static_base_url,
        "version": "1.0.0"
    }


@router.get("/businesses")
async def list_businesses():
    core_system = Dependencies.get_core_system()
    """获取所有业务列表"""
    if not core_system:
        raise HTTPException(status_code=500, detail="系统未初始化")

    businesses = []
    for business_id, config in core_system.businesses.items():
        status = core_system.get_business_status(business_id)
        businesses.append({
            "business_id": business_id,
            "name": config.name,
            "status": status.get("status", "未知"),
            "initialized": status.get("initialized", False),
            "image_fields": config.image_fields,
            "text_fields": config.text_fields
        })

    return {
        "total_businesses": len(businesses),
        "businesses": businesses
    }


@router.get("/businesses/{business_id}/health")
async def business_health_check(business_id: str):
    core_system = Dependencies.get_core_system()
    """检查特定业务的健康状态"""
    if not core_system:
        raise HTTPException(status_code=500, detail="系统未初始化")

    if business_id not in core_system.businesses:
        raise HTTPException(status_code=404, detail=f"业务 {business_id} 不存在")

    status = core_system.get_business_status(business_id)
    rag_instance = core_system.rag_instances.get(business_id)

    health_info = {
        "business_id": business_id,
        "healthy": status.get("initialized", False) and status.get("api_configured", False),
        "details": status,
        "ready_for_processing": rag_instance is not None and rag_instance.initialized,
        "image_mappings_count": len(core_system.image_manager.mappings)
    }

    return health_info

@router.post("/conversations/new")
async def create_new_conversation(
        business_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None
):
    """创建新会话"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    conversation = await Dependencies.conversation_manager.create_conversation(
        business_id=business_id,
        user_id=user_id,
        metadata=metadata
    )

    return {
        "success": True,
        "conversation_id": conversation.id
    }


@router.get("/conversations")
async def list_conversations(
        business_id: Optional[str] = Query(None),
        user_id: Optional[str] = Query(None),
        limit: int = Query(10, ge=1, le=100)
):
    """列出会话历史"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    conversations = await Dependencies.conversation_manager.list_conversations(
        business_id=business_id,
        user_id=user_id,
        limit=limit
    )

    return {
        "success": True,
        "conversations": [
            {
                "id": conv.id,
                "business_id": conv.business_id,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "turn_count": conv.turn_count,
                "active": conv.active,
                "last_message": conv.messages[-1].content[:100] if conv.messages else None
            }
            for conv in conversations
        ]
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取会话详情"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    conversation = await Dependencies.conversation_manager.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {
        "success": True,
        "conversation": conversation.to_dict()
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除会话"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    result = await Dependencies.conversation_manager.delete_conversation(conversation_id)

    if not result:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {"success": True, "message": "会话已删除"}




@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
        conversation_id: str,
        format: str = Query("json", enum=["json", "text"])
):
    """导出会话"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    content = await Dependencies.conversation_manager.export_conversation(
        conversation_id,
        format
    )

    if not content:
        raise HTTPException(status_code=404, detail="会话不存在")

    media_type = "application/json" if format == "json" else "text/plain"
    filename = f"conversation_{conversation_id}.{format}"

    return StreamingResponse(
        io.StringIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/conversations/cleanup")
async def cleanup_old_conversations(
        days: int = Query(7, ge=1, le=90),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """清理旧会话"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    async def cleanup_task():
        count = await Dependencies.conversation_manager.cleanup_old_conversations(days)
        logger.info(f"清理完成，删除了 {count} 个旧会话")

    background_tasks.add_task(cleanup_task)

    return {
        "success": True,
        "message": f"已启动清理任务，将删除 {days} 天前的会话"
    }


# ==================== 流式响应支持 ====================

async def generate_stream_response(
        content: str,
        conversation_id: str
) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    # 发送会话ID
    yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"

    # 模拟流式输出（实际应该从LLM获取流式响应）
    words = content.split()
    buffer = []

    for i, word in enumerate(words):
        buffer.append(word)

        # 每5个词发送一次
        if len(buffer) >= 5 or i == len(words) - 1:
            chunk = " ".join(buffer)
            yield f"data: {json.dumps({'content': chunk + ' '})}\n\n"
            buffer = []
            await asyncio.sleep(0.05)  # 模拟延迟

    # 发送结束标记
    yield f"data: {json.dumps({'finished': True})}\n\n"
