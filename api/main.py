from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
from pathlib import Path

from core.unified_processor import UnifiedDataProcessor, ProcessingConfig, DataSourceType
from adapters.crawler_adapter import CrawlerDataAdapter
from adapters.document_adapter import DocumentAdapter

app = FastAPI(title="统一数据处理API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/images", StaticFiles(directory="images"), name="images")

# 全局处理器实例
processor = UnifiedDataProcessor()


class DataProcessingRequest(BaseModel):
    business_id: str
    source_type: str  # "crawler" or "documents"
    data_path: str  # JSON文件路径或文档目录路径
    config: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    business_id: str
    query: str
    mode: str = "hybrid"


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    # 预注册默认业务
    await register_default_businesses()


@app.post("/api/v1/process_data")
async def process_data(request: DataProcessingRequest, background_tasks: BackgroundTasks):
    """处理数据接口"""
    try:
        # 创建处理配置
        config = ProcessingConfig(
            business_id=request.business_id,
            source_type=DataSourceType(request.source_type),
            batch_size=request.config.get("batch_size", 10) if request.config else 10,
            enable_image_download=request.config.get("enable_image_download", True) if request.config else True
        )

        # 注册业务（如果未注册）
        if request.business_id not in processor.rag_instances:
            await processor.register_business(request.business_id, config)

        # 创建数据源
        if request.source_type == "crawler":
            from core.unified_processor import CrawlerDataSource
            data_source = CrawlerDataSource(request.data_path)
        else:
            from core.unified_processor import DocumentsDataSource
            data_source = DocumentsDataSource(request.data_path)

        # 后台处理
        background_tasks.add_task(
            process_data_background,
            request.business_id,
            data_source
        )

        return {
            "success": True,
            "message": f"数据处理任务已启动",
            "business_id": request.business_id,
            "source_type": request.source_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/api/v1/query")
async def query_knowledge(request: QueryRequest):
    """查询接口"""
    try:
        if request.business_id not in processor.rag_instances:
            raise HTTPException(status_code=404, detail=f"业务未注册: {request.business_id}")

        result = await processor.query(
            business_id=request.business_id,
            query=request.query,
            mode=request.mode
        )

        return {
            "success": True,
            "data": result,
            "business_id": request.business_id,
            "query": request.query
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/api/v1/upload_crawler_data")
async def upload_crawler_data(
        business_id: str,
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None
):
    """上传爬虫数据文件"""
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="只支持JSON格式文件")

    try:
        # 保存文件
        upload_path = Path("data/crawler") / f"{business_id}_{file.filename}"
        upload_path.parent.mkdir(exist_ok=True)

        content = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(content)

        # 后台处理
        if background_tasks:
            from core.unified_processor import CrawlerDataSource
            data_source = CrawlerDataSource(str(upload_path))
            background_tasks.add_task(
                process_data_background,
                business_id,
                data_source
            )

        return {
            "success": True,
            "message": f"文件上传成功，正在后台处理",
            "filename": file.filename,
            "business_id": business_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# Dify集成接口
@app.post("/dify/knowledge/query")
async def dify_knowledge_query(
        business_id: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
):
    """Dify外部知识库接口"""
    try:
        result = await processor.query(
            business_id=business_id,
            query=query,
            mode="hybrid"
        )

        return {
            "records": [{
                "content": result,
                "score": 0.9,
                "title": f"{business_id}产品信息",
                "metadata": {
                    "business_id": business_id,
                    "source": "Unified RAG System"
                }
            }]
        }

    except Exception as e:
        return {
            "error": str(e),
            "records": []
        }


async def process_data_background(business_id: str, data_source):
    """后台数据处理任务"""
    try:
        print(f"开始处理 {business_id} 数据")
        result = await processor.process_data_source(business_id, data_source)
        print(f"处理完成: {result}")
    except Exception as e:
        print(f"后台处理失败: {e}")


async def register_default_businesses():
    """注册默认业务"""
    # 家具业务
    furniture_config = ProcessingConfig(
        business_id="furniture",
        source_type=DataSourceType.CRAWLER,
        batch_size=10,
        enable_image_download=True
    )
    await processor.register_business("furniture", furniture_config)

    # 可以添加更多默认业务...