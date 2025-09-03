# core/rag_instance.py
"""
生产环境RAG实例
"""
import asyncio
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple, Iterable, Set
import re
import openai
from lightrag import QueryParam
from config.settings import settings
from config.prompts import ChinesePrompts
from core.components import ImageOptimizer
from utils.url_helper import post_process_response_urls, path_manager
from utils.logger import setup_logger
from raganything.modalprocessors import ImageModalProcessor
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.openai import openai_embed

logger = setup_logger(__name__)


class ProductionRAGInstance:
    """生产环境RAG实例"""

    def __init__(self, business_id: str):
        self.business_id = business_id
        self.working_dir = f"{settings.working_dir}_{business_id}"

        self.lightrag_instance = None
        self.image_processor = None
        self.initialized = False

        # 缓存产品图片映射
        self.product_image_cache = {}

        # 记录已处理的图片内容哈希，避免重复处理
        self.processed_content_hashes: Set[str] = set()

        # 创建图片优化器（用于VLM处理）
        self.image_optimizer = ImageOptimizer(
            max_dimension=settings.max_dimension,  # 可配置
            jpeg_quality=settings.jpeg_quality  # 可配置
        )

        logger.info(f"创建RAG实例: {business_id}")

    async def initialize(self):
        """初始化RAG实例"""
        if self.initialized:
            return

        logger.info(f"初始化RAG实例: {self.business_id}")

        try:
            # 创建LightRAG实例
            await self._create_lightrag()

            # 创建图像处理器
            self._create_image_processor()

            self.initialized = True
            logger.info(f"RAG初始化完成: {self.business_id}")

        except Exception as e:
            logger.error(f"RAG初始化失败: {e}")
            raise

    async def _create_lightrag(self):
        """创建LightRAG实例"""

        # 使用中文提示词的LightRAG配置
        self.lightrag_instance = LightRAG(
            working_dir=self.working_dir,
            embedding_func=self._get_embedding_func(),
            llm_model_func=self._get_chinese_llm_func()
        )

        self._apply_chinese_prompts_to_lightrag()

        await self.lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        logger.info(f"LightRAG创建完成: {self.working_dir}")

    def _apply_chinese_prompts_to_lightrag(self):
        """将中文prompt应用到LightRAG实例"""
        if not self.lightrag_instance:
            return

        # 直接覆盖LightRAG内部的提示词
        try:
            # 如果LightRAG有prompt配置属性，直接修改
            if hasattr(self.lightrag_instance, 'llm_model_func'):

                original_llm_func = self.lightrag_instance.llm_model_func

                def chinese_wrapper_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                    # 强制使用中文系统提示词
                    if not system_prompt or "entity" in prompt.lower():
                        system_prompt = ChinesePrompts.ENTITY_EXTRACTION_SYSTEM
                    elif "query" in prompt.lower() or "search" in prompt.lower():
                        system_prompt = ChinesePrompts.QUERY_RESPONSE_SYSTEM

                    return original_llm_func(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        **kwargs
                    )

                self.lightrag_instance.llm_model_func = chinese_wrapper_llm_func
                logger.info("成功应用中文提示词到LightRAG")

        except Exception as e:
            logger.warning(f"应用中文提示词失败: {e}")

    def _get_chinese_llm_func(self):
        """中文LLM函数"""

        def chinese_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # 使用中文系统提示词
            if not system_prompt:
                if "entity" in prompt.lower() or "extract" in prompt.lower():
                    system_prompt = ChinesePrompts.ENTITY_EXTRACTION_SYSTEM
                elif "query" in prompt.lower() or "search" in prompt.lower():
                    system_prompt = ChinesePrompts.QUERY_RESPONSE_SYSTEM
                else:
                    system_prompt = ChinesePrompts.ENTITY_EXTRACTION_SYSTEM

            return openai_complete_if_cache(
                model=settings.llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=settings.api_key,
                base_url=settings.base_url,
                **kwargs
            )

        return chinese_llm_func

    def _get_embedding_func(self):
        """获取embedding函数"""

        return EmbeddingFunc(
            embedding_dim=settings.embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=settings.embedding_model,
                api_key=settings.api_key,
                base_url=settings.base_url
            )
        )

    def _create_image_processor(self):
        """创建图像处理器"""
        self.image_processor = ImageModalProcessor(
            lightrag=self.lightrag_instance,
            modal_caption_func=self._get_vision_func()
        )

        logger.info("图像处理器创建完成")

    def _get_vision_func(self):
        async def vision_model_func(
                prompt,
                system_prompt=None,
                history_messages=[],
                image_data=None,
                **kwargs
        ):
            try:

                # 统一使用异步OpenAI客户端
                client = openai.AsyncOpenAI(
                    api_key=settings.api_key,
                    base_url=settings.base_url
                )

                # 强制使用中文图像分析系统提示词
                chinese_system_prompt = ChinesePrompts.IMAGE_ANALYSIS_SYSTEM

                messages = [
                    {
                        "role": "system",
                        "content": chinese_system_prompt
                    }
                ]

                # 添加历史消息
                if history_messages:
                    messages.extend(history_messages)

                if image_data:
                    if isinstance(image_data, str) and not image_data.startswith('data:'):
                        # 如果是文件路径，读取并优化
                        if Path(image_data).exists():
                            logger.debug(f"优化本地图片: {image_data}")
                            optimized_base64 = self.image_optimizer.get_base64_optimized(image_data)
                            image_data = optimized_base64
                        else:
                            # 可能已经是base64
                            pass
                    # 有图片时的消息格式
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            }
                        ]
                    })

                    model_to_use = settings.vision_model
                else:
                    # 没有图片时的消息格式
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })

                    model_to_use = settings.llm_model

                try:
                    response = await client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        **kwargs
                    )
                    result = response.choices[0].message.content
                    logger.debug(f"中文Vision分析结果: {result[:100]}...")
                    return result
                finally:
                    await client.close()

            except Exception as e:
                logger.error(f"中文Vision function error: {e}")
                return f"图像处理出错: {str(e)}"

        return vision_model_func



    async def ensure_initialized(self):
        """确保已初始化"""
        if not self.initialized:
            await self.initialize()

    async def process_multimodal_content(self, modal_content: Dict[str, Any],
                                         entity_name: str, file_path: str,
                                         image_manager=None):
        """处理多模态内容 - 基于内容哈希去重"""
        await self.ensure_initialized()

        logger.info(f"处理商品: {entity_name}")

        try:
            # 获取所有图片路径
            img_path_dict = modal_content.get("img_path", {})

            # 统计
            total_images = len(img_path_dict)
            processed_count = 0
            skipped_count = 0

            # 逐一处理每张图片（已经在build_modal_content中去重）
            for img_key, img_path in img_path_dict.items():
                if not img_path or not isinstance(img_path, str):
                    continue

                # 检查是否已处理过（基于内容哈希）
                if image_manager:
                    mapping = image_manager.get_mapping(img_path)
                    if mapping and mapping.content_hash:
                        if mapping.content_hash in self.processed_content_hashes:
                            skipped_count += 1
                            logger.debug(f"跳过已处理内容: {img_key} (hash: {mapping.content_hash[:8]})")
                            continue
                        else:
                            self.processed_content_hashes.add(mapping.content_hash)

                logger.info(f"开始处理图片 - {entity_name} - {img_key}: {Path(img_path).name}")

                # 为每张图片创建单独的modal content
                single_image_content = {
                    "img_path": img_path,  # 本地路径
                    "img_caption": modal_content.get("img_caption", []),
                    "img_footnote": modal_content.get("img_footnote", [])
                }

                try:
                    # ImageModalProcessor会调用我们的optimized_vision_func
                    # 图片会在那里被压缩
                    result = await self.image_processor.process_multimodal_content(
                        modal_content=single_image_content,
                        content_type="image",
                        file_path=f"{file_path}_{img_key}",
                        entity_name=f"{entity_name} - {img_key}"
                    )

                    if result:
                        processed_count += 1
                        logger.info(f"图片处理成功: {entity_name} - {img_key}")

                except Exception as e:
                    logger.error(f"处理图片{entity_name} -  {img_key} 失败: {e}")
                    continue

            logger.info(
                f"商品处理完成: {entity_name}, 处理 {processed_count} 张，跳过 {skipped_count} 张（共 {total_images} 张）")
            return True

        except Exception as e:
            logger.error(f"多模态处理失败: {e}")
            await self._fallback_text_processing(modal_content, entity_name)
            return False

    def _prepare_image_content_format(self, modal_content: Dict[str, Any],
                                      entity_name: str, use_local_path: bool = True) -> Dict[str, Any]:
        ## TODO:只处理了cover_pic
        """
        准备符合RAG-Anything格式的图像内容
        use_local_path: True时使用本地路径(用于LLM处理)，False时使用远程URL(用于输出)
        """
        img_path = modal_content.get("img_path", {})

        # 获取正确的路径
        cover_pic = ""
        if isinstance(img_path, dict):
            cover_pic = img_path.get("cover_pic", "")
        elif isinstance(img_path, str):
            cover_pic = img_path

        # 如果需要本地路径且当前是远程URL，则转换
        if use_local_path and cover_pic.startswith("http"):
            cover_pic = path_manager.get_local_path(cover_pic)
        # 如果需要远程URL且当前是本地路径，则转换
        elif not use_local_path and not cover_pic.startswith("http"):
            cover_pic = path_manager.get_remote_url(cover_pic)

        captions = modal_content.get("img_caption", [f"分析{self.business_id}产品: {entity_name}"])
        footnotes = modal_content.get("img_footnote", [])

        return {
            "img_path": cover_pic,  # 根据需要使用本地或远程路径
            "img_caption": captions,
            "img_footnote": footnotes
        }

    async def _fallback_text_processing(self, modal_content: Dict[str, Any], entity_name: str):
        """备用文本处理"""
        text_content = self._convert_modal_to_text(modal_content, entity_name)
        await self.lightrag_instance.ainsert(text_content)
        logger.info(f"备用文本处理完成: {entity_name}")

    def _convert_modal_to_text(self, modal_content: Dict[str, Any], entity_name: str) -> str:
        """转换为文本内容"""
        text_parts = [f"商品名称: {entity_name}"]

        captions = modal_content.get("img_caption", [])
        if captions:
            for caption in captions:
                lines = caption.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- '):
                        field_info = line[2:]
                        if ':' in field_info and len(field_info) > 3:
                            text_parts.append(field_info)
                    elif line and len(line) > 3:
                        skip_prefixes = ['请', '以下', '重点', '1.', '2.', '3.', '4.']
                        if not any(line.startswith(prefix) for prefix in skip_prefixes):
                            text_parts.append(line)

        img_path = modal_content.get("img_path", {})
        if img_path:
            if isinstance(img_path, dict):
                for key, path in img_path.items():
                    if path:
                        text_parts.append(f"产品图片({key}): {path}")
            elif isinstance(img_path, str):
                text_parts.append(f"产品图片: {img_path}")

        return "\n".join(text_parts)

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """中文查询功能"""
        await self.ensure_initialized()

        logger.info(f"处理查询: {query}")

        try:

            # 使用家具专用的中文查询提示词
            chinese_query = ChinesePrompts.get_furniture_query_prompt(query)

            result = await self.lightrag_instance.aquery(
                chinese_query,
                param=QueryParam(mode=mode)
            )
            logger.info(f"<UNK>: {result}")

            if result:
                # 确保图片URL是远程访问格式
                processed_result = post_process_response_urls(result)
                logger.info(f"查询完成，结果长度: {len(processed_result)}")
                return processed_result

        except Exception as e:
            logger.error(f"查询失败: {e}")

        return f"抱歉，暂无与「{query}」相关的产品推荐信息。请稍后再试或换个关键词。"
