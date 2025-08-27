"""
增强版主程序 - enhanced_main.py
集成真实RAG-Anything引擎的完整系统
"""

import asyncio
import json
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
import time
import os

# 导入基础组件（从原main.py）
from main import SimpleImageManager, SimpleMultiModalProcessor, BusinessConfig

# 导入RAG集成组件
from real_rag_integration import EnhancedCoreSystem, RAG_AVAILABLE

# FastAPI相关导入
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 全局核心系统实例
enhanced_core_system = EnhancedCoreSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("=== 增强版RAG系统启动 ===")

    # 检查RAG-Anything可用性
    if RAG_AVAILABLE:
        print("✓ RAG-Anything 可用")
    else:
        print("⚠ RAG-Anything 未安装，将使用模拟版本")
        print("  安装命令: pip install rag-anything==1.2.7")

    # 检查API配置
    api_key = enhanced_core_system.api_config.get("api_key")
    if api_key:
        provider = enhanced_core_system.api_config.get("provider", "openai")
        print(f"✓ API配置已加载 (提供商: {provider})")
    else:
        print("⚠ 未配置API密钥，将使用模拟RAG")
        print("  配置方法: 设置环境变量 API_KEY 或创建 config.json")

    # 注册默认业务
    furniture_config = BusinessConfig(
        business_id="furniture",
        name="侘界家具",
        image_fields=["cover_pic", "detail_images"],
        text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
    )
    enhanced_core_system.register_business(furniture_config)

    print("=== 系统启动完成 ===")

    yield  # 应用运行期间

    # 关闭时执行
    print("系统正在关闭...")


# 创建FastAPI应用
app = FastAPI(
    title="增强版RAG系统",
    version="1.0.0",
    description="集成RAG-Anything的多业务智能问答系统",
    lifespan=lifespan
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
os.makedirs("images", exist_ok=True)
os.makedirs("rag_storage_furniture", exist_ok=True)

# 挂载静态文件服务
app.mount("/images", StaticFiles(directory="images"), name="images")


# API数据模型
class ProcessDataRequest(BaseModel):
    business_id: str
    json_file: str


class QueryRequest(BaseModel):
    business_id: str
    query: str
    mode: str = "hybrid"


class BusinessRegistration(BaseModel):
    business_id: str
    name: str
    image_fields: List[str]
    text_fields: List[str]


class APIConfigRequest(BaseModel):
    provider: str  # "openai" 或 "zhipu"
    api_key: str
    base_url: Optional[str] = None
    llm_model: Optional[str] = None
    vision_model: Optional[str] = None


# API路由
@app.post("/api/register_business")
async def register_business(request: BusinessRegistration):
    """注册新业务"""
    config = BusinessConfig(
        business_id=request.business_id,
        name=request.name,
        image_fields=request.image_fields,
        text_fields=request.text_fields
    )
    enhanced_core_system.register_business(config)
    return {"success": True, "message": f"业务 {request.business_id} 注册成功"}


@app.post("/api/configure")
async def configure_api(request: APIConfigRequest):
    """配置API密钥和模型"""
    try:
        # 更新API配置
        enhanced_core_system.api_config.update({
            "provider": request.provider,
            "api_key": request.api_key,
            "base_url": request.base_url or (
                "https://api.openai.com/v1" if request.provider == "openai" else "https://open.bigmodel.cn/api/paas/v4/"),
            "llm_model": request.llm_model or ("gpt-4o-mini" if request.provider == "openai" else "glm-4-flash"),
            "vision_model": request.vision_model or ("gpt-4o" if request.provider == "openai" else "glm-4v-flash")
        })

        # 保存配置到文件
        config_data = {"api": enhanced_core_system.api_config}
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return {"success": True, "message": "API配置已更新并保存"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置更新失败: {str(e)}")


@app.post("/api/process_data")
async def process_data(request: ProcessDataRequest, background_tasks: BackgroundTasks):
    """处理数据"""
    try:
        # 检查文件是否存在
        if not Path(request.json_file).exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {request.json_file}")

        # 后台处理任务
        background_tasks.add_task(
            enhanced_core_system.process_crawler_data,
            request.business_id,
            request.json_file
        )
        return {"success": True, "message": f"数据处理任务已启动，正在后台处理 {request.json_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(request: QueryRequest):
    """智能查询"""
    try:
        start_time = time.time()
        result = await enhanced_core_system.query(request.business_id, request.query, request.mode)
        processing_time = time.time() - start_time

        return {
            "success": True,
            "query": request.query,
            "result": result,
            "processing_time": round(processing_time, 2),
            "mode": request.mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{business_id}")
async def get_status(business_id: str):
    """获取业务状态"""
    try:
        status = enhanced_core_system.get_business_status(business_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system_info")
async def get_system_info():
    """获取系统信息"""
    return {
        "rag_anything_available": RAG_AVAILABLE,
        "api_configured": bool(enhanced_core_system.api_config.get("api_key")),
        "provider": enhanced_core_system.api_config.get("provider"),
        "businesses": list(enhanced_core_system.businesses.keys()),
        "working_directories": [f"rag_storage_{bid}" for bid in enhanced_core_system.businesses.keys()]
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    try:
        businesses_status = {}
        for business_id in enhanced_core_system.businesses.keys():
            businesses_status[business_id] = enhanced_core_system.get_business_status(business_id)

        return {
            "status": "healthy",
            "rag_available": RAG_AVAILABLE,
            "api_configured": bool(enhanced_core_system.api_config.get("api_key")),
            "businesses": businesses_status
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Dify集成接口
@app.post("/dify/knowledge/query")
async def dify_query(business_id: str, query: str, top_k: int = 3):
    """Dify外部知识库接口"""
    try:
        result = await enhanced_core_system.query(business_id, query)
        return {
            "records": [{
                "content": result,
                "score": 0.9,
                "title": f"{business_id}智能推荐",
                "metadata": {
                    "business_id": business_id,
                    "source": "Enhanced RAG System",
                    "rag_type": enhanced_core_system.get_business_status(business_id).get("rag_type", "Unknown")
                }
            }]
        }
    except Exception as e:
        return {
            "error": str(e),
            "records": []
        }


# 便民接口
@app.post("/api/quick_setup")
async def quick_setup():
    """快速设置 - 创建示例配置文件"""
    try:
        # 创建示例环境变量文件
        env_example = """# RAG系统环境变量配置示例
# 复制此文件为 .env 并填入真实的API密钥

# API提供商: openai 或 zhipu
LLM_PROVIDER=zhipu

# API密钥
API_KEY=your-api-key-here

# API基础URL (可选)
BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# 模型配置 (可选)
LLM_MODEL=glm-4-flash
VISION_MODEL=glm-4v-flash
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=2048

# OpenAI配置示例 (注释掉上面的，启用下面的)
# LLM_PROVIDER=openai
# API_KEY=your-openai-api-key
# BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
# VISION_MODEL=gpt-4o
# EMBEDDING_MODEL=text-embedding-3-large
# EMBEDDING_DIM=3072
"""

        with open(".env.example", "w", encoding="utf-8") as f:
            f.write(env_example)

        return {
            "success": True,
            "message": "示例配置文件已创建",
            "files_created": ["config.example.json", ".env.example"],
            "next_steps": [
                "1. 安装RAG-Anything: pip install rag-anything==1.2.7",
                "2. 复制 config.example.json 为 config.json 并填入API密钥",
                "3. 或复制 .env.example 为 .env 并填入环境变量",
                "4. 重启服务以加载配置"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建配置文件失败: {str(e)}")


# 使用示例和说明
async def example_usage():
    """使用示例"""
    print("=== 增强版RAG系统使用示例 ===")

    system = EnhancedCoreSystem()

    # 1. 注册业务
    furniture_config = BusinessConfig(
        business_id="furniture",
        name="侘界家具",
        image_fields=["cover_pic", "detail_images"],
        text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
    )
    system.register_business(furniture_config)

    # 2. 处理数据
    if Path("mosyy_goods.json").exists():
        await system.process_crawler_data("furniture", "mosyy_goods.json")
    else:
        print("未找到 mosyy_goods.json 文件")

    # 3. 测试查询
    queries = [
        "推荐侘寂风的茶桌",
        "有什么木与石的家具",
        "适合小户型的长凳"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        result = await system.query("furniture", query)
        print(f"回答: {result[:200]}..." if len(result) > 200 else f"回答: {result}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        # 运行示例
        asyncio.run(example_usage())
    else:
        # 启动API服务
        import uvicorn

        print("""
=== 增强版RAG系统 ===

启动选项:
1. 标准启动: python enhanced_main.py
2. 运行示例: python enhanced_main.py --example

配置方法:
1. 环境变量: export API_KEY="your-key"
2. 配置文件: 创建 config.json
3. 在线配置: 启动后访问 POST /api/configure

API文档: http://localhost:8000/docs
健康检查: http://localhost:8000/api/health
系统信息: http://localhost:8000/api/system_info

快速设置: curl -X POST http://localhost:8000/api/quick_setup
        """)

        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
        # 配置文件
        example_config = {
            "api": {
                "provider": "zhipu",
                "api_key": "your-zhipu-api-key-here",
                "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                "llm_model": "glm-4-flash",
                "vision_model": "glm-4v-flash",
                "embedding_model": "embedding-3",
                "embedding_dim": 2048
            },
            "openai_alternative": {
                "provider": "openai",
                "api_key": "your-openai-api-key-here",
                "base_url": "https://api.openai.com/v1",
                "llm_model": "gpt-4o-mini",
                "vision_model": "gpt-4o",
                "embedding_model": "text-embedding-3-large",
                "embedding_dim": 3072
            }
        }

        with open("config.example.json", "w", encoding="utf-8") as f:
            json.dump(example_config, f, ensure_ascii=False, indent=2)

        # 创建示创