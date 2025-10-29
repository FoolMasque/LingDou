# core/system.py
"""
核心系统类
"""
import asyncio
from dataclasses import field
from typing import Dict, Any, List, Literal, cast
from pathlib import Path
import json
import aiofiles
from config.settings import settings
from core.rag_instance import ProductionRAGInstance
from core.components import MultiModalProcessor, BusinessConfig, ImageManager, ImageMapping
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ProductionCoreSystem:
    """生产环境核心系统"""

    def __init__(self):
        self.businesses: Dict[str, BusinessConfig] = {}
        self.rag_instances: Dict[str, ProductionRAGInstance] = {}
        self.processors: Dict[str, MultiModalProcessor] = {}

        # 直接初始化组件
        self.image_manager = ImageManager()

        # 并发控制
        self.max_concurrent_items = 5  # 同时处理的最大商品数
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_items)

        logger.info(f"系统初始化完成，配置: provider={settings.provider}, model={settings.llm_model}")

    def register_business(self, config: BusinessConfig):
        """注册业务"""
        self.businesses[config.business_id] = config
        # 创建生产RAG实例
        self.rag_instances[config.business_id] = ProductionRAGInstance(config.business_id)
        # 直接创建处理器
        self.processors[config.business_id] = MultiModalProcessor(config.business_id)
        logger.info(f"业务注册成功: {config.name}")

    async def process_crawler_data(self, business_id: str, json_file: str):
        """处理爬虫数据"""
        if business_id not in self.businesses:
            raise ValueError(f"未注册的业务: {business_id}")
        async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        logger.info(f"开始处理 {len(data)} 条数据")

        processor = self.processors[business_id]
        rag = self.rag_instances[business_id]

        # 确保RAG实例已初始化
        await rag.ensure_initialized()

        # 第一步：批量下载所有图片
        all_image_urls = self._extract_all_image_urls(data)
        logger.info(f"总计需要下载 {len(all_image_urls)} 张图片")

        image_mappings = await self.image_manager.download_images(all_image_urls, business_id)
        logger.info(f"图片下载完成，成功 {len(image_mappings)} 张")

        # 第二步：并发处理商品数据
        success_count = 0
        failed_items = []

        # 创建处理任务
        tasks = []
        for i, item in enumerate(data):
            task = self._process_single_item(
                item, i, business_id, processor, rag, image_mappings
            )
            tasks.append(task)

        # 批量执行任务
        batch_size = 10  # 每批处理10个
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            try:
                results = await asyncio.gather(*batch, return_exceptions=True)

                # 统计结果
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_items.append((i + j, str(result)))
                        logger.error(f"处理失败 item {i + j}: {result}")
                    else:
                        success_count += 1
                # 批次间短暂等待
                if i + batch_size < len(tasks):
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                # 标记整个批次为失败
                for j in range(len(batch)):
                    failed_items.append((i + j, f"批次失败: {str(e)}"))

            logger.info(f"批次进度: {min(i + batch_size, len(tasks))}/{len(tasks)}")

        # 输出处理结果
        logger.info(f"数据处理完成，成功 {success_count}/{len(data)} 条")
        if failed_items:
            logger.warning(f"失败的项目: {failed_items[:5]}...")  # 只显示前5个

        return {
            "total": len(data),
            "success": success_count,
            "failed": len(failed_items),
            "failed_items": failed_items[:10],  # 返回前10个失败项,
            "image_download_success": len(image_mappings),
            "image_download_total": len(all_image_urls)
        }

    async def _process_single_item(self, item: Dict[str, Any], index: int,
                                   business_id: str, processor: MultiModalProcessor,
                                   rag: ProductionRAGInstance,
                                   image_mappings: Dict[str, ImageMapping]):
        """处理单个商品 - 带并发控制"""
        async with self.processing_semaphore:
            try:
                entity_name = item.get("商品名", f"商品_{index}")

                # 更新item中的图片URL为本地路径（用于处理）
                self._update_item_with_mappings(item, image_mappings)

                # 构建多模态内容
                modal_content = processor.build_modal_content(item, image_manager=self.image_manager)

                # 验证多模态内容
                img_path_dict = modal_content.get("img_path", {})
                image_count = len([p for p in img_path_dict.values() if p])

                logger.info(f"处理商品 [{index + 1}]: {entity_name}, 唯一图片 {image_count} 张")

                # 处理多模态内容
                await rag.process_multimodal_content(
                    modal_content=modal_content,
                    entity_name=entity_name,
                    file_path=f"{business_id}_{index}.json",
                    image_manager=self.image_manager
                )

                logger.info(f"处理完成 [{index + 1}]: {entity_name}")
                return True

            except Exception as e:
                logger.error(f"处理数据项 {index} 失败: {e}")
                raise e

    def _extract_all_image_urls(self, data: List[Dict[str, Any]]) -> List[str]:
        """提取所有图片URL，包括详情图片"""
        all_urls = set()  # 使用set去重

        for item in data:
            # 封面图片
            if item.get("cover_pic"):
                all_urls.add(item["cover_pic"])

            # 详情图片 - 确保正确处理
            detail_images = item.get("detail_images")
            if detail_images:
                if isinstance(detail_images, list):
                    # 是列表，逐一添加
                    for img_url in detail_images:
                        if img_url and isinstance(img_url, str):
                            all_urls.add(img_url)
                elif isinstance(detail_images, str):
                    # 是单个字符串，直接添加
                    all_urls.add(detail_images)

        logger.info(f"从 {len(data)} 条数据中提取到 {len(all_urls)} 个唯一图片URL")
        return list(all_urls)

    def _update_item_with_mappings(self, item: Dict[str, Any],
                                         image_mappings: Dict[str, ImageMapping]):
        """
        更新item映射
        保留原始URL用于后续查找
        """
        # 处理封面图
        if item.get("cover_pic") and item["cover_pic"] in image_mappings:
            mapping = image_mappings[item["cover_pic"]]
            item["cover_pic_original"] = item["cover_pic"]  # 保存原始URL
            item["cover_pic"] = mapping.local_path
            item["cover_pic_remote"] = mapping.remote_url
            logger.debug(f"映射封面图: {item['cover_pic_original'][-30:]} -> {Path(mapping.local_path).name}")

        # 处理详情图片
        detail_images = item.get("detail_images")
        if detail_images:
            if isinstance(detail_images, list):
                updated_images = []
                remote_images = []
                original_images = []

                for i, img_url in enumerate(detail_images):
                    if img_url and img_url in image_mappings:
                        mapping = image_mappings[img_url]
                        updated_images.append(mapping.local_path)
                        remote_images.append(mapping.remote_url)
                        original_images.append(img_url)
                        logger.debug(f"映射详情图{i}: {img_url[-30:]} -> {Path(mapping.local_path).name}")
                    else:
                        # 保持原样
                        updated_images.append(img_url)
                        remote_images.append(img_url)
                        original_images.append(img_url)
                        if img_url:
                            logger.warning(f"详情图{i}无映射: {img_url[-30:]}")

                item["detail_images_original"] = original_images  # 关键：保存原始URL列表
                item["detail_images"] = updated_images
                item["detail_images_remote"] = remote_images

            elif isinstance(detail_images, str) and detail_images in image_mappings:
                mapping = image_mappings[detail_images]
                item["detail_images_original"] = detail_images
                item["detail_images"] = mapping.local_path
                item["detail_images_remote"] = mapping.remote_url

    async def query(self, business_id: str, query: str, mode=cast(Literal["local", "global", "hybrid", "naive", "mix", "bypass"], "hybrid"), history=None) -> str:
        """纯文本查询接口"""
        if history is None:
            history = field(default_factory=list)
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]
        # result = await rag.aquery(query, mode)
        result = await rag.aquery_with_history(query=query, mode=mode,history=history)

        return result

    async def query_multimodal(self,
                               business_id: str,
                               query: str,
                               user_images: List[str] = None,
                               history=None,
                               mode=cast(Literal["local", "global", "hybrid", "naive", "mix", "bypass"], "hybrid")) -> Dict[str, Any]:
        """
        多模态查询接口

        Args:
            business_id: 业务ID
            query: 文本查询
            user_images: 用户上传的图片base64列表
            mode: 查询模式

        Returns:
            {
                "result": str,  # Markdown格式回答
                "library_images_count": int,  # 检索到的图片数量
            }
        """
        if history is None:
            history = field(default_factory=list)
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]

        logger.info(f"[{business_id}] 多模态查询: 用户图片 {len(user_images) if user_images else 0} 张")

        # 调用RAG实例的多模态查询
        result_data = await rag.aquery_multimodal_with_history(
            query=query,
            user_images=user_images,
            history=history,
            mode=mode
        )

        return result_data

    # ====== 流式方法 ======

    async def query_stream(self, business_id: str, query: str, mode=cast(Literal["local", "global", "hybrid", "naive", "mix", "bypass"], "hybrid"), history=None):
        """
        纯文本流式查询

        被调用路径：
        api/routes.py::_handle_streaming_query()
            → 这里
            → rag.aquery_stream()
        """
        if history is None:
            history = field(default_factory=list)
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]

        async for chunk in rag.aquery_stream(query, mode,history):
            yield chunk

    async def query_multimodal_stream(self,
                                      business_id: str,
                                      query: str,
                                      user_images: List[str] = None,
                                      history=None,
                                      mode: str = "hybrid"):
        """
        多模态流式查询

        被调用路径：
        api/routes.py::_handle_streaming_query()
            → 这里
            → rag.aquery_multimodal_stream()
        """
        if history is None:
            history = field(default_factory=list)
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]

        async for chunk in rag.aquery_multimodal_stream(query, user_images, mode,history):
            yield chunk

    def get_business_status(self, business_id: str) -> Dict[str, Any]:
        """获取业务状态"""
        if business_id not in self.businesses:
            return {"error": "业务不存在"}

        rag_instance = self.rag_instances[business_id]

        return {
            "business_id": business_id,
            "name": self.businesses[business_id].name,
            "rag_type": "ProductionRAG",
            "initialized": rag_instance.initialized,
            "api_configured": bool(settings.api_key),
            "provider": settings.provider,
            "models": {
                "llm": settings.llm_model,
                "vision": settings.vision_model,
                "embedding": settings.embedding_model
            },
            "max_concurrent": self.max_concurrent_items,
            "status": "运行中",
            "chinese_prompts": settings.use_chinese_prompts,
            "image_mappings": len(self.image_manager.mappings)
        }
