# api/server.py
"""
FastAPI服务器
"""
import asyncio
import os
import sys
from pathlib import Path

import uvicorn

# 获取项目根目录但不改变工作目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# 只添加到Python路径，不改变工作目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from core.system import ProductionCoreSystem
from core.components import BusinessConfig
from api.routes import router, Dependencies
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
    Dependencies.init(core_system)

    # Dependencies.set_core_system(core_system)

    # 注册默认业务
    try:
        furniture_config = BusinessConfig(
            business_id="furniture",
            name="侘界家具",
            image_fields=["cover_pic", "detail_images"],
            text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
        )

        # 马桶业务 - 预留
        toilet_config = BusinessConfig(
            business_id="toilet",
            name="智能马桶",
            image_fields=["cover_pic", "detail_images", "extra_pic"],
            text_fields=["produce", "cls", "type", "subtitle", "keyword"]
        )

        # 电器业务 - 预留
        electronics_config = BusinessConfig(
            business_id="electronics",
            name="智能电器",
            image_fields=["cover_pic", "detail_images"],
            text_fields=["商品名", "品牌", "型号", "功能特点", "技术参数", "适用场景"]
        )
        businesses = [furniture_config, toilet_config, electronics_config]
        for business in businesses:
            core_system.register_business(business)
            logger.info(f"业务注册完成: {business.name} ({business.business_id})")
        logger.info(f"所有业务注册完成，共 {len(businesses)} 个业务线")

        # core_system.register_business(furniture_config)
        # logger.info("默认业务注册完成")

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


# ==================== 配置管理 ====================

def get_storage_config():
    """获取存储配置"""
    backend = os.getenv("CONVERSATION_STORAGE", "file").lower()
    config = {}

    if backend == "redis":
        config = {
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "redis_db": int(os.getenv("REDIS_DB", "0")),
            "redis_prefix": os.getenv("REDIS_PREFIX", "lingdou:")
        }
    elif backend == "file":
        config = {
            "storage_dir": os.getenv("CONVERSATION_DIR", "conversations")
        }
    elif backend == "memory":
        logger.warning("⚠️  使用内存存储，重启后会话将丢失！")

    return {
        "backend": backend,
        "config": config
    }


# ==================== 定期任务 ====================

async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            # 每天凌晨2点执行清理
            await asyncio.sleep(24 * 3600)  # 等待24小时

            if Dependencies.conversation_manager:
                # 清理7天前的会话
                count = await Dependencies.conversation_manager.cleanup_old_conversations(days=7)
                logger.info(f"定期清理完成，删除了 {count} 个旧会话")
        except Exception as e:
            logger.error(f"定期清理失败: {e}")
            await asyncio.sleep(3600)  # 出错后等待1小时再试


# ==================== FastAPI应用 ====================
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


def main():
    """主函数"""
    # 配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8008"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))

    # 打印启动信息
    logger.info("=" * 60)
    logger.info("🚀 灵豆RAG系统 v1.0.0")
    logger.info(f"📍 地址: http://{host}:{port}")
    logger.info(f"📚 文档: http://{host}:{port}/docs")
    logger.info(f"💾 存储: {get_storage_config()['backend']}")
    logger.info(f"🔧 调试: {'开启' if settings.debug else '关闭'}")
    logger.info("=" * 60)

    # 启动服务
    if workers > 1:
        # 多进程模式（生产环境）
        uvicorn.run(
            "api.server:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
    else:
        # 单进程模式（开发环境）
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="debug" if settings.debug else "info"
        )


if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = "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"
    main()
# if __name__ == "__main__":
#     import uvicorn
#     import os
#
#     os.environ[
#         'API_KEY'] = "sk-proj-xxxx"
#     os.environ['LLM_PROVIDER'] = "openai"
#
#     print(f"""
# === 生产环境家具RAG系统 ===
#
# 配置信息:
# - API提供商: {settings.provider}
# - LLM模型: {settings.llm_model}
# - 视觉模型: {settings.vision_model}
# - 嵌入模型: {settings.embedding_model}
# - 服务端口: {settings.port}
# - 图片服务: {settings.static_base_url}
# - 中文模式: {settings.use_chinese_prompts}
#
# 访问地址:
# - API文档: http://localhost:{settings.port}/docs
# - 系统信息: http://localhost:{settings.port}/api/system_info
# - 健康检查: http://localhost:{settings.port}/health
#
# 使用示例:
# curl -X POST "http://localhost:{settings.port}/api/query" \\
#   -H "Content-Type: application/json" \\
#   -d '{{"business_id": "furniture", "query": "推荐一个茶桌", "mode": "hybrid"}}'
# """)
#
#     uvicorn.run(
#         "api.server:app",
#         host=settings.host,
#         port=settings.port,
#         reload=False  # 生产环境关闭自动重载
#     )
