# api/routes.py
"""
API路由
"""
import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path

from api.models import ProcessRequest, QueryRequest, QueryResponse
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)
# 注意：这里需要在server.py中注入core_system
core_system = None


def set_core_system(system):
    """设置核心系统实例"""
    global core_system
    core_system = system


@router.post("/process_data")
async def process_data(request: ProcessRequest, background_tasks: BackgroundTasks):
    """处理数据API"""
    try:
        if not Path(request.json_file).exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {request.json_file}")

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
async def query(request: QueryRequest):
    """查询API"""
    try:
        if not core_system:
            raise HTTPException(status_code=500, detail="系统未初始化")

        start_time = time.time()
        result = await core_system.query(request.business_id, request.query, request.mode)
        processing_time = time.time() - start_time

        logger.info(f"查询完成: {request.query[:50]}... 耗时: {processing_time:.2f}s")

        return QueryResponse(
            success=True,
            query=request.query,
            result=result,
            processing_time=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{business_id}")
async def get_status(business_id: str):
    """获取状态API"""
    try:
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


