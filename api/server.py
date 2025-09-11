# api/server.py
"""
FastAPI服务器
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from core.system import ProductionCoreSystem
from core.components import BusinessConfig
from api.routes import router, set_core_system
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 全局系统实例
core_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global core_system

    logger.info("=== 生产环境RAG系统启动 ===")
    logger.info(f"配置: provider={settings.provider}, model={settings.llm_model}")

    # 创建核心系统
    core_system = ProductionCoreSystem()

    # 注入到路由中
    set_core_system(core_system)

    # 注册默认业务
    try:
        furniture_config = BusinessConfig(
            business_id="furniture",
            name="侘界家具",
            image_fields=["cover_pic", "detail_images"],
            text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
        )

        core_system.register_business(furniture_config)
        logger.info("默认业务注册完成")

    except Exception as e:
        logger.error(f"业务注册失败: {e}")

    logger.info("=== 系统启动完成 ===")

    yield

    # 清理资源
    logger.info("系统关闭中...")
    if core_system:
        for business_id, rag in core_system.rag_instances.items():
            try:
                if hasattr(rag, 'finalize'):
                    await rag.finalize()
            except Exception as e:
                logger.error(f"清理业务 {business_id} 失败: {e}")

    logger.info("系统关闭完成")


# 创建FastAPI应用
app = FastAPI(
    title="生产环境家具RAG系统",
    description="基于多模态知识图谱的中文家具智能问答",
    version="1.0.0",
    lifespan=lifespan
)

# 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需要配置具体域名
    allow_methods=["*"],
    allow_headers=["*"]
)

# 静态文件服务
image_dir = Path(settings.image_storage)
image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(image_dir)), name="images")

# 注册路由
app.include_router(router, prefix="/api")


# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "provider": settings.provider,
        "chinese_mode": settings.use_chinese_prompts
    }


if __name__ == "__main__":
    import uvicorn
    import os

    os.environ[
        'API_KEY'] = "sk-proj-xxxx"
    os.environ['LLM_PROVIDER'] = "openai"

    print(f"""
=== 生产环境家具RAG系统 ===

配置信息:
- API提供商: {settings.provider}
- LLM模型: {settings.llm_model}  
- 视觉模型: {settings.vision_model}
- 嵌入模型: {settings.embedding_model}
- 服务端口: {settings.port}
- 图片服务: {settings.static_base_url}
- 中文模式: {settings.use_chinese_prompts}

访问地址:
- API文档: http://localhost:{settings.port}/docs
- 系统信息: http://localhost:{settings.port}/api/system_info
- 健康检查: http://localhost:{settings.port}/health

使用示例:
curl -X POST "http://localhost:{settings.port}/api/query" \\
  -H "Content-Type: application/json" \\
  -d '{{"business_id": "furniture", "query": "推荐一个茶桌", "mode": "hybrid"}}'
""")

    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        reload=False  # 生产环境关闭自动重载
    )
