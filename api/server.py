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


# 在Windows上自动检测系统字体（SimSun、SimHei、Microsoft YaHei等）
# 我们只需要确保系统有中文字体即可
from config.settings import settings

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote
import mimetypes
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

    # 注册默认业务，不初始化， 懒加载
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

        # # ✅ 关键：初始化所有 RAG 实例
        # logger.info("开始初始化所有 RAG 实例...")
        # for business_id, rag_instance in core_system.rag_instances.items():
        #     logger.info(f"初始化 {business_id} RAG 实例...")
        #     await rag_instance.initialize()
        #     logger.info(f"✅ {business_id} RAG 实例初始化完成")
        #
        # logger.info(f"✅ 所有业务注册完成，共 {len(businesses)} 个业务线")

    except Exception as e:
        logger.error(f"业务注册失败: {e}")

    logger.info("=== 系统启动完成 ===")

    yield

    # 清理资源
    logger.info("系统关闭中...")
    if core_system:
        for business_id, rag in core_system.rag_instances.items():
            try:
                # if hasattr(rag, 'finalize'):
                #     await rag.finalize()
                if hasattr(rag, 'cleanup'):
                    rag.cleanup()
            except Exception as e:
                logger.error(f"清理业务 {business_id} 失败: {e}")

    logger.info("系统关闭完成")


# ==================== 配置管理 ====================

def get_storage_config():
    """获取存储配置"""
    backend = settings.conversation.storage_backend
    return {
        "backend": backend,
        "config": settings.get_storage_config()
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

# 静态文件服务 - 支持多种图片路径格式和URL编码的中文路径
# 1. static/images/{business_id}/filename (旧格式，结构化数据)
# 2. rag_storage_{business_id}/parsed/{doc}/images/filename (新格式，文档解析)
# 3. rag_storage_{business_id}/images/filename (新格式，结构化数据)
# 
# 使用自定义路由而不是StaticFiles，因为StaticFiles不支持URL编码的中文路径
# Notes：不再创建static/images目录，现在图片存储在rag_storage_{business_id}目录下
project_root = Path(__file__).parent.parent

@app.get("/images/{file_path:path}")
async def serve_image(file_path: str):
    """
    自定义图片服务端点，支持URL编码的中文路径
    
    支持路径格式：
    1. rag_storage_{business_id}/parsed/{doc}/images/{filename}
    2. rag_storage_{business_id}/images/{filename}
    3. static/images/{business_id}/{filename}
    
    Args:
        file_path: 图片路径（支持URL编码的中文字符）
    
    Returns:
        FileResponse: 图片文件响应
    """
    try:
        # URL解码路径（处理中文字符）
        # 例如：M400-AR%E6%99%BA%E8%83%BD%E7%9C%BC%E9%95%9C -> M400-AR智能眼镜
        decoded_path = unquote(file_path)
        
        # 构建完整文件路径
        file_full_path = project_root / decoded_path
        
        # 安全检查：确保文件在项目根目录下（防止路径遍历攻击）
        try:
            file_full_path.resolve().relative_to(project_root.resolve())
        except ValueError:
            logger.warning(f"非法路径访问尝试: {decoded_path}")
            raise HTTPException(status_code=403, detail="访问被拒绝：文件不在项目目录内")
        
        # 检查文件是否存在
        if not file_full_path.exists():
            logger.debug(f"图片文件不存在: {file_full_path}")
            raise HTTPException(status_code=404, detail=f"文件不存在: {decoded_path}")
        
        # 检查是否是文件（不是目录）
        if not file_full_path.is_file():
            logger.debug(f"路径不是文件: {file_full_path}")
            raise HTTPException(status_code=404, detail="路径不是文件")
        
        # 根据文件扩展名判断媒体类型
        media_type, _ = mimetypes.guess_type(str(file_full_path))
        if not media_type:
            # 默认使用 image/jpeg，如果无法判断
            if file_full_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                media_type = f"image/{file_full_path.suffix.lower().lstrip('.')}"
            else:
                media_type = "application/octet-stream"
        
        # 返回文件
        return FileResponse(
            str(file_full_path),
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图片服务错误: {file_path}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

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
    host = settings.host
    port = settings.port
    # reload 和 workers 是运行时参数，保留从环境变量读取
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
    # os.environ['OPENAI_API_KEY'] = "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"
    main()

# DEBUG用
# import asyncio
# import uvicorn
#
# def debug_main():
#     config = uvicorn.Config(
#         app="api.server:app",
#         host="0.0.0.0",
#         port=8008,
#         reload=False,
#         log_level="info",
#     )
#     server = uvicorn.Server(config)
#
#     # 手动创建事件循环，避免 PyCharm patch 冲突
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#
#     loop.run_until_complete(server.serve())
#     loop.close()
#
#
# if __name__ == "__main__":
#     debug_main()