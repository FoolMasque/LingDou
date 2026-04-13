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

        self._init_locks: Dict[str, asyncio.Lock] = {} # 初始化锁（避免并发初始化）

        # 业务配置持久化文件路径
        self.business_config_file = Path("./data/business_configs.json")

        # 直接初始化组件
        self.image_manager = ImageManager()

        # 并发控制
        self.max_concurrent_items = 5  # 同时处理的最大商品数
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_items)

        # 加载已保存的业务配置
        self._load_business_configs()

        logger.info(f"系统初始化完成，配置: provider={settings.provider}, model={settings.llm_model}")

    def register_business(self, config: BusinessConfig):
        """注册业务"""
        self.businesses[config.business_id] = config
        # 创建生产RAG实例
        self.rag_instances[config.business_id] = ProductionRAGInstance(config.business_id)
        # 直接创建处理器
        self.processors[config.business_id] = MultiModalProcessor(config.business_id, config=config)
        # 创建初始化锁
        self._init_locks[config.business_id] = asyncio.Lock()
        
        # 保存业务配置到文件
        self._save_business_configs()
        
        
        logger.info(f"业务注册成功: {config.name}")

    def delete_business(self, business_id: str):
        """删除业务"""
        if business_id not in self.businesses:
            raise ValueError(f"业务不存在: {business_id}")

        # 1. 移除RAG实例（如果存在）
        if business_id in self.rag_instances:
            # TODO: 实现RAG实例的清理逻辑（如果有资源占用）
            del self.rag_instances[business_id]
        
        # 2. 移除处理器
        if business_id in self.processors:
            del self.processors[business_id]

        # 3. 移除锁
        if business_id in self._init_locks:
            del self._init_locks[business_id]

        # 4. 移除业务配置
        del self.businesses[business_id]

        # 5. 保存配置
        self._save_business_configs()

        # 6. 删除数据目录 (rag_storage_{business_id})
        # 注意：这里我们尝试删除，但如果被占用可能会失败，日志记录即可
        import shutil
        try:
            rag_storage_dir = Path(f"./rag_storage_{business_id}")
            if rag_storage_dir.exists():
                shutil.rmtree(rag_storage_dir)
                logger.info(f"✅ 已删除业务数据目录: {rag_storage_dir}")
        except Exception as e:
            logger.error(f"删除业务数据目录失败: {e}")

        logger.info(f"业务已删除: {business_id}")
        return True

    def update_business_config(self, business_id: str, updates: Dict[str, Any]):
        """更新业务配置"""
        if business_id not in self.businesses:
            raise ValueError(f"业务不存在: {business_id}")
        
        config = self.businesses[business_id]
        
        # 更新允许的字段
        allowed_updates = [
            "response_instruction","default_response_instruction", "field_mapping", "caption_template", 
            "caption_instructions", "vision_prompt_template", "system_prompt_template"
        ]
        
        updated = False
        for key, value in updates.items():
            if key in allowed_updates:
                setattr(config, key, value)
                updated = True
        
        if updated:
            self._save_business_configs()
            # 重新初始化处理器以应用新配置（主要是caption builder）
            self.processors[business_id] = MultiModalProcessor(business_id, config=config)
            logger.info(f"业务配置已更新: {business_id}")
        
        return config

    async def _ensure_rag_initialized(self, business_id: str):
        """
        确保RAG实例已初始化（懒加载）

        使用锁避免并发初始化同一个实例
        """
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag_instance = self.rag_instances[business_id]

        # 如果已经初始化，直接返回
        if rag_instance.initialized:
            return

        # 使用锁避免并发初始化
        async with self._init_locks[business_id]:
            # 双重检查（进入锁后再检查一次）
            if rag_instance.initialized:
                return

            logger.info(f"🔄 首次使用 {business_id}，开始初始化 RAG 实例...")
            await rag_instance.initialize()
            logger.info(f"✅ {business_id} RAG 实例初始化完成")

    async def process_crawler_data(self, business_id: str, json_file: str):
        """处理爬虫数据"""
        if business_id not in self.businesses:
            raise ValueError(f"未注册的业务: {business_id}")
        await self._ensure_rag_initialized(business_id)
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
                # 获取实体名称字段（优先使用配置的字段）
                business_config = self.businesses.get(business_id)
                entity_name = None

                # 1. 优先使用 field_mapping 中的 "product_name"
                if business_config and business_config.field_mapping:
                    mapping = business_config.field_mapping
                    if "product_name" in mapping:
                        source_field = mapping["product_name"]
                        entity_name = item.get(source_field)

                # 2. 其次使用 entity_name_field
                if not entity_name and business_config and business_config.entity_name_field:
                    entity_name = item.get(business_config.entity_name_field)
                
                # 3. 再次使用第一个 text_field
                if not entity_name and business_config and business_config.text_fields:
                    entity_name = item.get(business_config.text_fields[0])
                
                # 4. 降级：尝试常见字段名
                if not entity_name:
                    for field in ["商品名", "produce", "name", "title", "产品名"]:
                        if field in item:
                            entity_name = item[field]
                            break
                
                if not entity_name:
                    entity_name = f"商品_{index}"

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

        # 尝试获取当前业务的 mapping，但这里可能不好获取 business_id，
        # 暂时使用通用逻辑，或者假设调用方会确保 key 的正确性。
        # 为了更好地支持 field_mapping，我们需要遍历 item 动态获取
        
        for item in data:
            # 简单策略：遍历所有看起来像URL的value
            # 或者复用 _collect_all_images 的逻辑? 
            # 鉴于 _collect_all_images 比较复杂且依赖 processor，这里先用简单逻辑 + 常见字段
            
            potential_image_fields = ["cover_pic", "detail_images", "image", "img", "pic", "photos", "images"]
            
            for key, value in item.items():
                if not value:
                    continue
                    
                is_potential = False
                if key in potential_image_fields:
                    is_potential = True
                elif "pic" in key.lower() or "img" in key.lower() or "image" in key.lower():
                    is_potential = True
                
                if is_potential:
                    if isinstance(value, str) and (value.startswith("http") or value.startswith("/")):
                        all_urls.add(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and (v.startswith("http") or v.startswith("/")):
                                all_urls.add(v)

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

    async def query(self, business_id: str, query: str, mode=cast(Literal["local", "global", "hybrid", "naive", "mix", "bypass"], "hybrid"), history=None, conversation_id: str = None, only_need_context: bool = False, only_need_prompt: bool = False) -> str:
        """纯文本查询接口"""
        if history is None:
            history = field(default_factory=list)
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")
        await self._ensure_rag_initialized(business_id)

        rag = self.rag_instances[business_id]
        logger.info(f"[{business_id}] 执行查询，模式: {mode}, 历史记录数: {len(history) if history else 0}")
        result = await rag.aquery_with_history(query=query, mode=mode, history=history, conversation_id=conversation_id, only_need_context=only_need_context, only_need_prompt=only_need_prompt)

        return result

    async def query_multimodal(self,
                               business_id: str,
                               query: str,
                               user_images: List[str] = None,
                               history=None,
                               mode=cast(Literal["local", "global", "hybrid", "naive", "mix", "bypass"], "hybrid"),
                               conversation_id: str = None) -> Dict[str, Any]:
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
        await self._ensure_rag_initialized(business_id)

        rag = self.rag_instances[business_id]

        logger.info(f"[{business_id}] 多模态查询: 用户图片 {len(user_images) if user_images else 0} 张")

        # 调用RAG实例的多模态查询
        result_data = await rag.aquery_multimodal_with_history(
            query=query,
            user_images=user_images,
            history=history,
            mode=mode,
            conversation_id=conversation_id
        )

        return result_data

    # ====== 流式方法 ======

    async def query_stream(self, business_id: str, query: str, mode=cast(Literal["local", "global", "hybrid", "naive", "mix", "bypass"], "hybrid"), history=None, conversation_id: str = None):
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
        await self._ensure_rag_initialized(business_id)

        rag = self.rag_instances[business_id]

        async for chunk in rag.aquery_stream(query, business_id, mode, history, conversation_id=conversation_id):
            yield chunk

    async def query_multimodal_stream(self,
                                      business_id: str,
                                      query: str,
                                      user_images: List[str] = None,
                                      history=None,
                                      mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
                                      conversation_id: str = None):
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
        await self._ensure_rag_initialized(business_id)

        rag = self.rag_instances[business_id]

        async for chunk in rag.aquery_multimodal_stream(query, business_id, user_images, mode, history, conversation_id=conversation_id):
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
    
    def _save_business_configs(self):
        """保存业务配置到文件"""
        try:
            # 确保目录存在
            self.business_config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 将BusinessConfig转换为字典
            configs_dict = {}
            for business_id, config in self.businesses.items():
                configs_dict[business_id] = {
                    "business_id": config.business_id,
                    "name": config.name,
                    "image_fields": config.image_fields,
                    "text_fields": config.text_fields,
                    "caption_template": config.caption_template,
                    "caption_fields": config.caption_fields,
                    "caption_instructions": config.caption_instructions,
                    "entity_name_field": config.entity_name_field,
                    "vision_prompt_template": config.vision_prompt_template,
                    "response_instruction": config.response_instruction,
                    "field_mapping": config.field_mapping
                }
            
            # 保存到文件
            with open(self.business_config_file, 'w', encoding='utf-8') as f:
                json.dump(configs_dict, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"业务配置已保存: {len(configs_dict)} 个业务")
        except Exception as e:
            logger.error(f"保存业务配置失败: {e}", exc_info=True)
    
    def _load_business_configs(self):
        """从文件加载业务配置"""
        try:
            if not self.business_config_file.exists():
                logger.info("业务配置文件不存在，跳过加载")
                return
            
            with open(self.business_config_file, 'r', encoding='utf-8') as f:
                configs_dict = json.load(f)
            
            # 加载业务配置
            loaded_count = 0
            for business_id, config_data in configs_dict.items():
                try:
                    config = BusinessConfig(
                        business_id=config_data["business_id"],
                        name=config_data["name"],
                        image_fields=config_data.get("image_fields", []),
                        text_fields=config_data.get("text_fields", []),
                        caption_template=config_data.get("caption_template"),
                        caption_fields=config_data.get("caption_fields"),
                        caption_instructions=config_data.get("caption_instructions"),

                        entity_name_field=config_data.get("entity_name_field"),
                        vision_prompt_template=config_data.get("vision_prompt_template"),
                        response_instruction=config_data.get("response_instruction"),
                        field_mapping=config_data.get("field_mapping")
                    )
                    
                    # 注册业务（但不保存，避免循环）
                    self.businesses[config.business_id] = config
                    self.rag_instances[config.business_id] = ProductionRAGInstance(config.business_id)
                    self.processors[config.business_id] = MultiModalProcessor(config.business_id, config=config)
                    self._init_locks[config.business_id] = asyncio.Lock()
                    
                    loaded_count += 1
                    logger.info(f"✅ 已加载业务配置: {config.name} ({business_id})")
                except Exception as e:
                    logger.error(f"加载业务配置失败 {business_id}: {e}", exc_info=True)
            
            if loaded_count > 0:
                logger.info(f"✅ 从文件加载了 {loaded_count} 个业务配置")
        except Exception as e:
            logger.error(f"加载业务配置文件失败: {e}", exc_info=True)
