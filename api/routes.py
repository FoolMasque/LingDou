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
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple

import aiohttp
from PIL import Image
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from pathlib import Path

from config.settings import settings
from api.models import ProcessRequest, QueryRequest, QueryResponse
from core.image_processor import ImageProcessor  # 抽取图片处理逻辑
from core.conversation_manager import ConversationManager, StorageBackend
from api.models import ChatMessage, ConversationCreate, ConversationListRequest, BusinessConfigUpdate
from core.components import BusinessConfig
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
        conv_conf = settings.conversation
        storage_backend = conv_conf.storage_backend
        storage_config = settings.get_storage_config()

        cls.conversation_manager = ConversationManager(
            storage_backend=StorageBackend(storage_backend),
            storage_config=storage_config
        )
        # logger.info(f"会话管理器初始化完成: {storage_backend}, 配置: {storage_config}")
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
        # 确保业务存在，不存在则自动创建
        if request.business_id not in core_system.rag_instances:
            logger.info(f"检测到新业务ID: {request.business_id}，自动创建默认配置...")
            default_config = _create_default_business_config(request.business_id)
            core_system.register_business(default_config)
            logger.info(f"✅ 新业务已自动注册: {default_config.name} ({request.business_id})")

        # 确保 RAG 实例已初始化
        await core_system._ensure_rag_initialized(request.business_id)

        # 获取RAG实例（确保使用正确的business_id）
        rag_instance = core_system.rag_instances[request.business_id]
        logger.info(f"✅ 使用业务 {request.business_id} 的RAG实例进行查询")

        # 会话管理逻辑
        conversation = None
        message_list = []

        if request.conversation_id or request.max_history > 0:
            # 如果提供了会话ID，先检查业务ID是否匹配
            if request.conversation_id:
                existing_conv = await conversation_manager.get_conversation(request.conversation_id)
                if existing_conv and existing_conv.business_id != request.business_id:
                    # 业务ID不匹配，创建新会话（清空历史记录）
                    logger.warning(f"会话 {request.conversation_id} 的业务ID ({existing_conv.business_id}) 与请求的业务ID ({request.business_id}) 不匹配，创建新会话")
                    conversation = await conversation_manager.create_conversation(
                        business_id=request.business_id,
                        metadata=request.metadata
                    )
                    # 新会话没有历史记录
                    message_list = []
                else:
                    # 业务ID匹配或会话不存在，使用get_or_create
                    conversation = await conversation_manager.get_or_create_conversation(
                        conversation_id=request.conversation_id,
                        business_id=request.business_id,
                        metadata=request.metadata
                    )
                    # 获取上下文（只获取当前会话的历史）
                    _, message_list = await conversation_manager.get_context_for_query(
                        conversation.id,
                        max_turns=request.max_history,
                        format_type="lightrag"
                    )
            else:
                # 没有提供会话ID，创建新会话
                conversation = await conversation_manager.create_conversation(
                    business_id=request.business_id,
                    metadata=request.metadata
                )
                # 新会话没有历史记录
                message_list = []
        # TODO: conversation_id没传的情况
        # else:
        #     import uuid
        #     request.conversation_id = str(uuid.uuid4())
        #     message_list = []
        #     await conversation_manager.add_message(
        #         conversation.id,
        #         "user",
        #         request.query,
        #         request.image_base64_list
        #     )
            # 添加用户消息
            await conversation_manager.add_message(
                conversation.id,
                "user",
                request.query,
                request.image_base64_list
            )


        if request.streaming:
            # 添加回复在函数内
            return await _handle_streaming_query(request, core_system, conversation_manager, conversation.id, history=message_list)
        else:
            result = await _handle_blocking_query(request, core_system, conversation_manager, conversation.id, history=message_list)

            # 添加回复到历史
            await conversation_manager.add_message(
                conversation.id,
                role="assistant",
                content=result.result
            )

            result.conversation_id = conversation.id

            # result.result 已经在 rag_instance.aquery_with_history 中通过 post_process_response_urls 处理过了
            # 这里只需要提取图片URL，不需要再次处理
            # 提取图片URL
            import re
            pattern = r'http[s]?://[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)'
            urls = re.findall(pattern, result.result)
            images = list({url for url in urls if url})
            
            # 更新结果
            result.images = images
            
            return result

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
            mode=request.mode,
            conversation_id=conversation_id,
            only_need_context=request.only_need_context,
            only_need_prompt=request.only_need_prompt
        )
        result = result_data["result"]
        # TODO：library_images_count目前全为空
        library_images_count = result_data.get("library_images_count", 0)
    else:
        # 纯文本查询
        logger.info("执行纯文本查询")
        result = await core_system.query(
            business_id=request.business_id,
            query=request.query,
            history=history,
            mode=request.mode,
            conversation_id=conversation_id,
            only_need_context=request.only_need_context,
            only_need_prompt=request.only_need_prompt
        )
        library_images_count = 0

    processing_time = time.time() - start_time

    logger.info(f"查询完成，耗时: {processing_time:.2f}s")

    # result 已经在 rag_instance.aquery_with_history 中通过 post_process_response_urls 处理过了
    # 这里只需要提取图片URL，不需要再次处理
    # 提取图片URL
    import re
    pattern = r'http[s]?://[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)'
    urls = re.findall(pattern, result)
    images = list({url for url in urls if url})

    return QueryResponse(
        success=True,
        query=request.query,
        result=result,  # result 已经在 rag_instance 中处理过了
        processing_time=round(processing_time, 2),
        conversation_id=conversation_id,
        images=images,
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
                    mode=request.mode,
                    conversation_id=conversation_id
                )
            else:
                # 纯文本流式查询
                logger.info("执行流式纯文本查询")
                result_stream = core_system.query_stream(
                    business_id=request.business_id,
                    query=request.query,
                    history=history,
                    mode=request.mode,
                    conversation_id=conversation_id
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
                    }

                # 发送SSE格式数据
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 小延迟，让前端有时间渲染
                await asyncio.sleep(0.01)

            try:
                from utils.url_helper import post_process_response_urls
                processed_content = post_process_response_urls(accumulated_content)
                accumulated_content = processed_content
            except Exception as e:
                logger.warning(f"URL 替换失败: {e}")

            await conversation_manager.add_message(
                conversation_id,
                role="assistant",
                content=accumulated_content
            )
            processed_time = time.time() - start_time
            # 发送最终的完整内容
            final_data = {
                "type": "complete",
                "content": accumulated_content,
                "conversation_id": conversation_id,
                "metadata": {
                            "user_images_count": len(user_images_base64), # chunk.get("user_images_count", 0),
                            "library_images_count": 0,
                            "processing_time": processed_time,
                            "chunk_count": chunk_count
                        }
            }
            # 发送完成信号
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            logger.info(f"流式查询完成，共耗时 {processed_time} s!")

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
            "embedding": settings.embedding_model,
            "rerank": settings.rerank.model
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


@router.delete("/businesses/{business_id}")
async def delete_business(business_id: str):
    """删除业务"""
    core_system = Dependencies.get_core_system()
    if not core_system:
        raise HTTPException(status_code=500, detail="系统未初始化")

    try:
        core_system.delete_business(business_id)
        return {"success": True, "message": f"业务 {business_id} 已删除"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"删除业务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除业务失败: {str(e)}")


@router.get("/businesses/{business_id}/config")
async def get_business_config(business_id: str):
    """获取业务配置"""
    core_system = Dependencies.get_core_system()
    if not core_system:
        raise HTTPException(status_code=500, detail="系统未初始化")

    if business_id not in core_system.businesses:
        raise HTTPException(status_code=404, detail="业务不存在")

    config = core_system.businesses[business_id]
    
    # 避免变量名冲突，这里显式导入
    import config.runtime_prompt_patch as runtime_patch

    # 返回配置详情
    return {
        "business_id": config.business_id,
        "name": config.name,
        "image_fields": config.image_fields,
        "text_fields": config.text_fields,
        "response_instruction": getattr(config, "response_instruction", None),
        "default_response_instruction": "请用简洁自然的方式回答问题",
        "field_mapping": getattr(config, "field_mapping", None),
        "caption_template": config.caption_template,
        "caption_instructions": config.caption_instructions,
        "vision_prompt_template": config.vision_prompt_template,
        "system_prompt_template": getattr(config, "system_prompt_template", None),
        "default_system_prompt_template": runtime_patch.system_prompt
    }


@router.put("/businesses/{business_id}/config")
async def update_business_config(business_id: str, config_update: BusinessConfigUpdate):
    """更新业务配置"""
    core_system = Dependencies.get_core_system()
    if not core_system:
        raise HTTPException(status_code=500, detail="系统未初始化")

    try:
        # 过滤None值
        updates = {k: v for k, v in config_update.dict().items() if v is not None}
        
        if not updates:
            return {"success": True, "message": "无配置更新"}

        updated_config = core_system.update_business_config(business_id, updates)
        
        return {
            "success": True, 
            "message": "配置已更新",
            "config": {
                "response_instruction": updated_config.response_instruction,
                "field_mapping": updated_config.field_mapping,
                "caption_template": updated_config.caption_template,
                "vision_prompt_template": updated_config.vision_prompt_template,
                "system_prompt_template": updated_config.system_prompt_template
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


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


@router.put("/conversations/{conversation_id}/metadata")
async def update_conversation_metadata(conversation_id: str, metadata: Dict[str, Any]):
    """更新会话元数据（例如更新 user_persona）"""
    if not Dependencies.conversation_manager:
        raise HTTPException(status_code=500, detail="会话管理器未初始化")

    success = await Dependencies.conversation_manager.update_metadata(conversation_id, metadata)

    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {"success": True, "message": "会话元数据已更新"}




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


from fastapi import UploadFile, File, Form
from pathlib import Path
import shutil


def _analyze_json_structure(json_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    分析JSON数据结构，自动识别文本字段和图片字段
    
    Args:
        json_data: JSON数据列表（至少包含一个样本）
        
    Returns:
        Tuple[List[str], List[str]]: (text_fields, image_fields)
    """
    if not json_data or not isinstance(json_data, list) or len(json_data) == 0:
        return ["title", "content"], ["cover_pic", "detail_images"]
    
    # 收集所有字段
    all_fields = set()
    for item in json_data[:10]:  # 只分析前10个样本
        if isinstance(item, dict):
            all_fields.update(item.keys())
    
    # 图片字段特征
    image_keywords = ['pic', 'image', 'img', 'photo', 'picture', 'icon', 'avatar', 'thumbnail']
    image_fields = []
    text_fields = []
    
    for field in all_fields:
        field_lower = field.lower()
        
        # 判断是否为图片字段
        is_image = False
        for keyword in image_keywords:
            if keyword in field_lower:
                is_image = True
                break
        
        # 检查值类型（如果值是URL，可能是图片）
        if not is_image and len(json_data) > 0:
            sample_value = json_data[0].get(field)
            if isinstance(sample_value, str):
                # 检查是否是URL
                if any(ext in sample_value.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', 'http://', 'https://']):
                    is_image = True
        
        if is_image:
            image_fields.append(field)
        else:
            # 排除明显不是文本的字段
            if field_lower not in ['id', '_id', 'created_at', 'updated_at', 'timestamp']:
                text_fields.append(field)
    
    # 如果没有识别到图片字段，使用默认值
    if not image_fields:
        image_fields = ["cover_pic", "detail_images"]
    
    # 如果没有识别到文本字段，使用默认值
    if not text_fields:
        text_fields = ["title", "content", "summary", "keyword"]
    
    return text_fields, image_fields


def _create_default_business_config(business_id: str, sample_json: Optional[List[Dict[str, Any]]] = None) -> BusinessConfig:
    """
    为新的业务ID创建默认配置
    
    Args:
        business_id: 业务ID
        sample_json: 可选的样本JSON数据，用于自动识别字段
        
    Returns:
        BusinessConfig: 默认业务配置
    """
    # 根据业务ID生成友好的名称
    name_map = {
        "paper": "论文文档",
        "document": "通用文档",
        "knowledge": "知识库",
    }
    
    name = name_map.get(business_id, f"业务-{business_id}")
    
    # 如果有样本数据，自动识别字段
    if sample_json:
        text_fields, image_fields = _analyze_json_structure(sample_json)
        logger.info(f"自动识别字段 - 文本: {text_fields}, 图片: {image_fields}")
    else:
        # 默认配置：通用文档处理
        text_fields = ["title", "content", "summary", "keyword"]
        image_fields = ["cover_pic", "detail_images"]
    
    return BusinessConfig(
        business_id=business_id,
        name=name,
        image_fields=image_fields,
        text_fields=text_fields
    )


@router.post("/ingest/document")
async def upload_document(
        file: UploadFile = File(...),
        business_id: str = Form(...),
        doc_type: str = Form("manual"),
        use_gpu: bool = Form(False),
        overwrite: bool = Form(False),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        core_system=Depends(Dependencies.get_core_system)
):
    """
    上传并处理单个文档

    支持格式：PDF, DOCX, PPTX, XLSX, 图片等
    
    特性：
    - 支持动态创建新业务：如果 business_id 不存在，会自动创建默认配置
    - 业务配置会在首次使用时懒加载初始化
    """
    temp_file = None
    try:
        # 1. 确保业务存在，不存在则自动创建
        if business_id not in core_system.rag_instances:
            logger.info(f"检测到新业务ID: {business_id}，自动创建默认配置...")
            default_config = _create_default_business_config(business_id)
            core_system.register_business(default_config)
            logger.info(f"✅ 新业务已自动注册: {default_config.name} ({business_id})")

        # 2. 确保 RAG 实例已初始化
        await core_system._ensure_rag_initialized(business_id)

        # 3. 保存临时文件
        temp_dir = Path("./data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / file.filename

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"文件已保存: {temp_file}")

        # 4. 获取 RAG 实例
        rag_instance = core_system.rag_instances[business_id]

        if not rag_instance.rag_anything:
            raise HTTPException(400, "RAGAnything 未启用，请检查配置")

        # 5. 处理文档
        # TODO: 异步处理
        result = await rag_instance.insert_document(
            file_path=str(temp_file),
            doc_type=doc_type,
            use_gpu=use_gpu,
            overwrite=overwrite
        )

        # 6. 返回结果
        if result.get("status") == "success":
            return {
                "success": True,
                "message": "文档处理成功",
                "file": result.get("file"),
                "status": result.get("status"),
                "parse_method": result.get("parse_method"),
                "chunks_count": result.get("chunks_inserted", 0),
                "parse_time": result.get("parse_time"),
                "insert_time": result.get("insert_time"),
                "total_time": result.get("total_time"),
                "device": result.get("device"),
                "entity_extract_rounds": result.get("entity_extract_rounds"),
                "pdf_stats": result.get("pdf_stats"),
            }
        else:
            raise HTTPException(500, result.get("error", "文档处理失败"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"临时文件已清理: {temp_file}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


@router.post("/ingest/batch_documents")
async def upload_batch_documents(
        files: list[UploadFile] = File(...),
        business_id: str = Form(...),
        doc_type: str = Form("manual"),
        use_gpu: bool = Form(False),
        overwrite: bool = Form(False),
        core_system=Depends(Dependencies.get_core_system)
):
    """批量上传文档"""
    temp_files = []

    try:
        # 1. 确保业务存在，不存在则自动创建
        if business_id not in core_system.rag_instances:
            logger.info(f"检测到新业务ID: {business_id}，自动创建默认配置...")
            default_config = _create_default_business_config(business_id)
            core_system.register_business(default_config)
            logger.info(f"✅ 新业务已自动注册: {default_config.name} ({business_id})")

        await core_system._ensure_rag_initialized(business_id)

        # 2. 保存所有临时文件
        temp_dir = Path("./data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            temp_file = temp_dir / file.filename
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(str(temp_file))

        logger.info(f"保存了 {len(temp_files)} 个临时文件")

        # 3. 批量处理
        rag_instance = core_system.rag_instances[business_id]

        if not rag_instance.rag_anything:
            raise HTTPException(400, "RAGAnything 未启用")

        result = await rag_instance.insert_document_batch(
            file_paths=temp_files,
            doc_type=doc_type,
            use_gpu=use_gpu,
            overwrite=overwrite
        )

        return {
            "success": True,
            "total": result.get("total", 0),
            "success_count": result.get("success", 0),
            "failed_count": result.get("failed", 0),
            "results": result.get("results", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理所有临时文件
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
            except Exception as e:
                logger.warning(f"清理临时文件失败 {temp_file}: {e}")

        if temp_files:
            logger.info(f"清理了 {len(temp_files)} 个临时文件")

@router.delete("/ingest/document/{business_id}/{doc_id}")
async def delete_document(
        business_id: str,
        doc_id: str,
        core_system=Depends(Dependencies.get_core_system)
):
    """
    删除 RAG 引擎中的某篇文档及其产生的知识分块与实体图谱。
    doc_id 通常是该文档的原始文件名。
    """
    if business_id not in core_system.rag_instances:
        raise HTTPException(status_code=404, detail="业务实例不存在")
        
    await core_system._ensure_rag_initialized(business_id)
        
    rag_instance = core_system.rag_instances[business_id]
    result = await rag_instance.delete_document(doc_id)
    
    if result.get("status") == "success":
        return result
    else:
        msg = result.get("message", "删除失败")
        status_code = 404 if "未找到" in msg else 500
        raise HTTPException(status_code=status_code, detail=msg)


@router.post("/ingest/folder")
async def upload_folder(
        business_id: str = Form(...),
        folder_path: str = Form(...),
        doc_type: str = Form("manual"),
        use_gpu: bool = Form(False),
        recursive: bool = Form(True),
        background_tasks: BackgroundTasks = None,
        core_system=Depends(Dependencies.get_core_system)
):
    """
    处理整个文件夹（服务器本地路径）

    Args:
        folder_path: 服务器上的文件夹路径
        recursive: 是否递归处理子文件夹
    """
    try:
        # 1. 检查路径
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(400, f"文件夹不存在: {folder_path}")

        # 2. 确保业务存在，不存在则自动创建
        if business_id not in core_system.rag_instances:
            logger.info(f"检测到新业务ID: {business_id}，自动创建默认配置...")
            default_config = _create_default_business_config(business_id)
            core_system.register_business(default_config)
            logger.info(f"✅ 新业务已自动注册: {default_config.name} ({business_id})")

        await core_system._ensure_rag_initialized(business_id)

        # 3. 收集文件
        pattern = "**/*" if recursive else "*"
        supported_extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}

        file_paths = [
            str(f) for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not file_paths:
            raise HTTPException(400, f"文件夹中没有支持的文档: {folder_path}")

        logger.info(f"找到 {len(file_paths)} 个文档文件")

        # 4. 批量处理
        rag_instance = core_system.rag_instances[business_id]

        if not rag_instance.rag_anything:
            raise HTTPException(400, "RAGAnything 未启用")

        # 如果文件很多，使用后台任务
        if len(file_paths) > 10 and background_tasks:
            background_tasks.add_task(
                rag_instance.insert_document_batch,
                file_paths=file_paths,
                doc_type=doc_type,
                use_gpu=use_gpu
            )

            return {
                "success": True,
                "message": f"后台任务已启动，将处理 {len(file_paths)} 个文件",
                "total": len(file_paths),
                "files": [Path(f).name for f in file_paths[:10]]  # 只显示前10个
            }
        else:
            # 同步处理
            result = await rag_instance.insert_document_batch(
                file_paths=file_paths,
                doc_type=doc_type,
                use_gpu=use_gpu
            )

            return {
                "success": True,
                "total": result.get("total", 0),
                "success_count": result.get("success", 0),
                "failed_count": result.get("failed", 0),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件夹处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business/register")
async def register_business(
        business_id: str = Form(...),
        name: Optional[str] = Form(None),
        image_fields: Optional[str] = Form(None),  # JSON字符串或逗号分隔
        text_fields: Optional[str] = Form(None),  # JSON字符串或逗号分隔
        sample_json_file: Optional[UploadFile] = File(None),  # 可选的样本JSON文件，用于自动识别字段
        # 定制化配置
        caption_template: Optional[str] = Form(None),  # 自定义图片描述模板（支持{字段名}占位符）
        caption_fields: Optional[str] = Form(None),  # JSON格式：{"显示名": "JSON字段名"}
        caption_instructions: Optional[str] = Form(None),  # JSON格式：["指令1", "指令2"]
        entity_name_field: Optional[str] = Form(None),  # 实体名称字段
        vision_prompt_template: Optional[str] = Form(None),  # 自定义视觉分析提示词模板
        core_system=Depends(Dependencies.get_core_system)
):
    """
    手动注册新业务（可选，如果业务不存在会自动创建默认配置）
    
    Args:
        business_id: 业务ID（必需）
        name: 业务名称（可选，默认根据business_id生成）
        image_fields: 图片字段列表，JSON字符串或逗号分隔（可选）
        text_fields: 文本字段列表，JSON字符串或逗号分隔（可选）
        sample_json_file: 可选的样本JSON文件，用于自动识别字段结构
    """
    try:
        # 检查业务是否已存在
        if business_id in core_system.rag_instances:
            return {
                "success": True,
                "message": f"业务 {business_id} 已存在",
                "business_id": business_id,
                "existing": True
            }
        
        sample_json_data = None
        
        # 如果提供了样本JSON文件，解析并分析
        if sample_json_file:
            try:
                import json
                content = await sample_json_file.read()
                sample_json_data = json.loads(content.decode('utf-8'))
                if not isinstance(sample_json_data, list):
                    sample_json_data = [sample_json_data]
                logger.info(f"已读取样本JSON，包含 {len(sample_json_data)} 条记录")
            except Exception as e:
                logger.warning(f"解析样本JSON失败: {e}，将使用手动指定的字段")
        
        # 解析字段列表
        def parse_fields(fields_str: Optional[str]) -> List[str]:
            if not fields_str:
                return []
            try:
                # 尝试解析JSON
                import json
                return json.loads(fields_str)
            except:
                # 否则按逗号分隔
                return [f.strip() for f in fields_str.split(",") if f.strip()]
        
        image_fields_list = parse_fields(image_fields)
        text_fields_list = parse_fields(text_fields)
        
        # 如果提供了样本数据且未手动指定字段，自动识别
        if sample_json_data and (not image_fields_list or not text_fields_list):
            auto_text, auto_image = _analyze_json_structure(sample_json_data)
            if not text_fields_list:
                text_fields_list = auto_text
            if not image_fields_list:
                image_fields_list = auto_image
            logger.info(f"自动识别字段 - 文本: {text_fields_list}, 图片: {image_fields_list}")
        
        # 创建业务配置
        if name:
            business_name = name
        else:
            name_map = {
                "paper": "论文文档",
                "document": "通用文档",
                "knowledge": "知识库",
            }
            business_name = name_map.get(business_id, f"业务-{business_id}")
        
        # 使用提供的字段或默认字段
        if not image_fields_list:
            image_fields_list = ["cover_pic", "detail_images"]
        if not text_fields_list:
            text_fields_list = ["title", "content", "summary", "keyword"]
        
        # 解析定制化配置
        caption_fields_dict = None
        if caption_fields:
            try:
                import json
                caption_fields_dict = json.loads(caption_fields)
            except Exception as e:
                logger.warning(f"解析caption_fields失败: {e}")
        
        caption_instructions_list = None
        if caption_instructions:
            try:
                import json
                caption_instructions_list = json.loads(caption_instructions)
            except Exception as e:
                logger.warning(f"解析caption_instructions失败: {e}")
        
        config = BusinessConfig(
            business_id=business_id,
            name=business_name,
            image_fields=image_fields_list,
            text_fields=text_fields_list,
            caption_template=caption_template,
            caption_fields=caption_fields_dict,
            caption_instructions=caption_instructions_list,
            entity_name_field=entity_name_field,
            vision_prompt_template=vision_prompt_template
        )
        
        core_system.register_business(config)
        logger.info(f"✅ 业务手动注册成功: {business_name} ({business_id})")
        
        return {
            "success": True,
            "message": f"业务 {business_id} 注册成功",
            "business_id": business_id,
            "name": business_name,
            "image_fields": image_fields_list,
            "text_fields": text_fields_list,
            "caption_template": caption_template,
            "caption_fields": caption_fields_dict,
            "caption_instructions": caption_instructions_list,
            "entity_name_field": entity_name_field,
            "vision_prompt_template": vision_prompt_template,
            "existing": False,
            "auto_detected": sample_json_data is not None and (not image_fields or not text_fields)
        }
        
    except Exception as e:
        logger.error(f"业务注册失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))