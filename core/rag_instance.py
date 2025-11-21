# core/rag_instance.py

import base64
import logging
import os
import json
import re
import time
import shutil

import openai
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Set, List, Tuple, Literal, cast
from lightrag import QueryParam
from lightrag.prompt import PROMPTS
# 延迟导入 RAG-Anything，以便在设置环境变量后再导入
# from raganything import RAGAnything
# from raganything.modalprocessors import ImageModalProcessor

import config.runtime_prompt_patch
from api.models import ChatMessage
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed
import asyncio
import inspect

from api.routes import Dependencies
# from api.server import core_system
from config.settings import settings
from core.components import ImageOptimizer, BusinessConfig
from utils.url_helper import post_process_response_urls, path_manager
from utils.logger import setup_logger
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

logger = setup_logger(__name__)


class ProductionRAGInstance:
    """生产环境RAG实例"""

    def __init__(self, business_id: str):
        self.business_id = business_id
        # ✅ 确保每个业务有独立的存储目录
        # 格式：./rag_storage_{business_id} 或 rag_storage_{business_id}
        base_dir = settings.working_dir.rstrip('/').rstrip('\\')
        if base_dir.endswith('rag_storage'):
            # 如果base_dir是rag_storage，直接拼接business_id
            self.working_dir = f"{base_dir}_{business_id}"
        else:
            # 否则使用标准格式
            self.working_dir = f"./rag_storage_{business_id}"
        logger.info(f"业务 {business_id} 的存储目录: {self.working_dir}")

        self.lightrag_instance = None
        self.image_processor = None
        self.initialized = False

        self.rag_anything = None

        # 缓存产品图片映射
        self.product_image_cache = {}

        # 记录已处理的图片内容哈希，避免重复处理
        self.processed_content_hashes: Set[str] = set()

        # 创建图片优化器（用于VLM处理）
        self.image_optimizer = ImageOptimizer(
            max_dimension=settings.max_dimension,
            jpeg_quality=settings.jpeg_quality
        )

        # 批处理缓存
        self.embedding_cache = {}
        self.batch_texts = []
        self.batch_threshold = 20
        self.processed_items_count = 0

        logger.info(f"创建RAG实例: {business_id}")

    async def initialize(self):
        """初始化RAG实例"""
        if self.initialized:
            logger.info(f"{self.business_id} 已经初始化过了，跳过")
            return

        logger.info(f"开始初始化RAG-Anything实例: {self.business_id}")

        try:
            # 应用中文提示词
            from config.runtime_prompt_patch import apply_chinese_prompts_runtime
            apply_chinese_prompts_runtime()

            await self._create_lightrag()
            logger.info(f"✅ LightRAG 创建完成")

            # 创建图像处理器
            self._create_image_processor()
            logger.info(f"✅ 图像处理器创建完成")

            if settings.rag_anything.enabled:
                logger.info(f"🔄 启用 RAG-Anything 文档解析...")
                await self._init_rag_anything()
                logger.info(f"✅ RAG-Anything 初始化完成")
            else:
                logger.warning(f"⚠️  RAG-Anything 未启用")

            self.initialized = True
            logger.info(f"RAG初始化完成: {self.business_id}")


        except Exception as e:
            logger.error(f"RAG初始化失败: {e}")
            raise

    async def _create_lightrag(self):
        """创建LightRAG实例 """
        chunk_size = max(200, settings.rag_anything.chunk_size)
        chunk_overlap = max(0, min(chunk_size // 2, settings.rag_anything.chunk_overlap))
        entity_rounds = max(0, settings.rag_anything.entity_extract_rounds)
        
        # 知识图谱抽取策略：根据配置模式决定effective_rounds
        kg_mode = settings.rag_anything.kg_extraction_mode
        effective_rounds = entity_rounds
        
        if entity_rounds > 0:
            if kg_mode == "all":
                # 全部抽取，不限制
                logger.info("知识图谱抽取模式: 全部抽取")
            elif kg_mode == "adaptive":
                # 自适应模式：基于时间限制
                max_time = settings.rag_anything.kg_max_extraction_time
                logger.info(f"知识图谱抽取模式: 自适应（最大时间 {max_time}秒）")
                # 注意：LightRAG本身不支持时间限制，这里只是记录配置
                # 实际的时间控制需要在文档处理层面实现
            elif kg_mode == "ratio":
                # 比例模式：基于chunk比例
                ratio = max(0.0, min(1.0, settings.rag_anything.kg_extraction_ratio))
                logger.info(f"知识图谱抽取模式: 比例抽取（{ratio*100:.1f}%）")
                # 注意：LightRAG不支持直接按比例抽取，需要通过调整gleaning近似实现
                # 这里只是记录配置，实际控制需要在文档处理层面实现
            elif kg_mode == "limit":
                # 限制数量模式
                max_chunks = settings.rag_anything.kg_max_chunks_per_doc
                if max_chunks > 0:
                    logger.info(f"知识图谱抽取模式: 限制数量（最多 {max_chunks} 个chunk）")
                    # 注意：LightRAG不支持直接限制chunk数，需要通过调整gleaning近似实现
            else:
                logger.warning(f"未知的知识图谱抽取模式: {kg_mode}，使用默认配置")
        
        kwargs = dict(
            working_dir=self.working_dir,
            embedding_func=self._get_embedding_func(),
            llm_model_func=self._get_llm_func(),
            chunk_token_size=chunk_size,  # 可通过配置调整
            chunk_overlap_token_size=chunk_overlap,
            entity_extract_max_gleaning=effective_rounds,  # 0=完全关闭实体抽取，1+=开启
            max_parallel_insert=8,
            # 性能优化：当entity_extract_max_gleaning=0时，减少不必要的处理
            max_entity_tokens=6000 if effective_rounds > 0 else 0,  # 关闭实体抽取时设为0
            max_relation_tokens=8000 if effective_rounds > 0 else 0,  # 关闭关系抽取时设为0
        )

        if settings.rerank.enabled:
            kwargs.update(
                rerank_model_func=self._get_rerank_func(),
                min_rerank_score=settings.rerank.score_threshold,
            )

        self.lightrag_instance = LightRAG(**kwargs)

        await self.lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        logger.info(f"LightRAG创建完成: {self.working_dir} (业务: {self.business_id})")
        # 验证working_dir是否正确
        working_path = Path(self.working_dir)
        if working_path.exists():
            # 检查是否有数据文件
            graph_file = working_path / "graph_chunk_entity_relation.graphml"
            vdb_chunks = working_path / "vdb_chunks.json"
            if graph_file.exists() or vdb_chunks.exists():
                logger.info(f"✅ 存储目录存在且包含数据: {working_path.absolute()}")
            else:
                logger.warning(f"⚠️  存储目录存在但为空: {working_path.absolute()}")
        else:
            logger.warning(f"⚠️  存储目录不存在，将创建: {working_path.absolute()}")

    async def _init_rag_anything(self):
        """
        ✅ 关键：初始化 RAGAnything，但使用已有的 LightRAG

        现在调用 RAGAnything 和原来查询逻辑共享同一个存储
        
        Notes: RAG-Anything的convert_text_to_pdf方法硬编码了Linux字体路径。
        通过monkey patch让它使用我们检测到的字体路径。
        """
        import os
        from config.settings import SystemConfig
        
        # ✅ 检测中文字体路径
        if not settings.rag_anything.chinese_font_path:
            detected_font = SystemConfig._detect_chinese_font()
            if detected_font:
                settings.rag_anything.chinese_font_path = detected_font
                logger.info(f"✅ 检测到系统中文字体: {detected_font}")
            else:
                logger.warning("⚠️ 未检测到中文字体，TXT文件中的中文可能显示为黑框")
        else:
            logger.info(f"✅ 配置的中文字体路径: {settings.rag_anything.chinese_font_path}")
        
        # ✅ Monkey patch: 修改RAG-Anything的convert_text_to_pdf方法，使用我们检测到的字体
        if settings.rag_anything.chinese_font_path:
            self._patch_rag_anything_font(settings.rag_anything.chinese_font_path)
        
        # 延迟导入RAG-Anything（在monkey patch之后）
        from raganything import RAGAnything, RAGAnythingConfig

        # 创建配置
        rag_config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser=settings.rag_anything.parser or "mineru",
            parse_method=settings.rag_anything.parse_method,

            # 多模态处理配置
            enable_image_processing=settings.rag_anything.enable_image_processing,
            enable_table_processing=settings.rag_anything.enable_table,
            enable_equation_processing=settings.rag_anything.enable_formula,
            max_concurrent_files=3,
            display_content_stats=True,
        )

        # 传入已有的 LightRAG 实例
        self.rag_anything = RAGAnything(
            config=rag_config,
            lightrag=self.lightrag_instance,
            llm_model_func=self._get_llm_func(),  # 分析内容
            vision_model_func=self._get_vision_func(),  # 处理图片需要
            embedding_func=self._get_embedding_func(),
        )

        logger.info(f"✅ RAGAnything 已绑定到 LightRAG（路径: {self.working_dir}）")
    
    def _create_image_processor(self):
        """创建图像处理器"""
        # 延迟导入，避免在设置环境变量之前导入
        from raganything.modalprocessors import ImageModalProcessor
        self.image_processor = ImageModalProcessor(
            lightrag=self.lightrag_instance,
            modal_caption_func=self._get_vision_func()
        )

        logger.info("图像处理器创建完成")

    def _get_llm_func(self):
        """获取LLM函数"""
        from lightrag.llm.openai import openai_complete_if_cache

        def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            if history_messages is None:
                history_messages = []
            return openai_complete_if_cache(
                model=settings.llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=settings.api_key,
                base_url=settings.base_url,
                **kwargs
            )

        return llm_func

    def _get_rerank_func(self):
        """获取embedding函数"""

        from lightrag.rerank import ali_rerank
        from functools import partial

        return partial(
            ali_rerank,
            model=settings.rerank.model,
            api_key=settings.rerank.api_key,
            base_url=settings.rerank.base_url
        )

    def _get_embedding_func(self):
        """获取embedding函数"""
        from lightrag.utils import EmbeddingFunc
        from lightrag.llm.openai import openai_embed

        # 根据模型选择处理策略
        if settings.embedding_model in ["text-embedding-v4", "text-embedding-v3"]:
            return self._get_qwen_v4_embedding_func()
        else:
            return EmbeddingFunc(
                embedding_dim=settings.embedding_dim,
                max_token_size=settings.max_token_size,
                func=lambda texts: openai_embed(
                    texts,
                    model=settings.embedding_model,
                    api_key=settings.api_key,
                    base_url=settings.base_url
                )
            )

    def _get_qwen_v4_embedding_func(self):
        """Qwen v4/v3模型专用embedding函数 - 限制批量大小"""

        async def qwen_v4_optimized_embed(texts):
            """专门针对Qwen v4/v3模型的embedding处理"""
            if not texts:
                return []

            BATCH_SIZE = 10
            MAX_RETRIES = 2

            all_embeddings = []
            total_batches = (len(texts) - 1) // BATCH_SIZE + 1

            if len(texts) > 5:
                logger.info(f"Qwen v4 embedding: {len(texts)}个文本, {total_batches}个批次")

            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]

                for retry in range(MAX_RETRIES):
                    try:
                        result = openai_embed(
                            batch_texts,
                            model=settings.embedding_model,
                            api_key=settings.api_key,
                            base_url=settings.base_url
                        )

                        if inspect.iscoroutine(result):
                            batch_embeddings = await result
                        else:
                            batch_embeddings = result

                        if batch_embeddings and len(batch_embeddings[0]) != settings.embedding_dim:
                            logger.warning(f"维度不匹配: 期望{settings.embedding_dim}, 实际{len(batch_embeddings[0])}")

                        all_embeddings.extend(batch_embeddings)
                        break

                    except Exception as e:
                        error_msg = str(e)
                        if "batch size" in error_msg.lower() or "行数" in error_msg:
                            if len(batch_texts) > 1:
                                logger.warning(f"批次大小{len(batch_texts)}仍超限，拆分为单个")
                                single_embeddings = []
                                for text in batch_texts:
                                    try:
                                        single_result = openai_embed(
                                            [text],
                                            model=settings.embedding_model,
                                            api_key=settings.api_key,
                                            base_url=settings.base_url
                                        )
                                        if inspect.iscoroutine(single_result):
                                            single_result = await single_result
                                        single_embeddings.extend(single_result)
                                        await asyncio.sleep(0.1)
                                    except:
                                        dummy = [0.0] * settings.embedding_dim
                                        single_embeddings.append(dummy)
                                all_embeddings.extend(single_embeddings)
                                break
                            else:
                                dummy = [0.0] * settings.embedding_dim
                                all_embeddings.append(dummy)
                                break

                        if retry < MAX_RETRIES - 1:
                            await asyncio.sleep(0.5)
                        else:
                            dummy_embeddings = [[0.0] * settings.embedding_dim] * len(batch_texts)
                            all_embeddings.extend(dummy_embeddings)

                if i + BATCH_SIZE < len(texts):
                    await asyncio.sleep(0.2)

            if len(texts) > 5:
                logger.info(f"Qwen v4 embedding完成: {len(all_embeddings)}个向量")

            return all_embeddings

        return EmbeddingFunc(
            embedding_dim=settings.embedding_dim,
            max_token_size=settings.max_token_size,
            func=qwen_v4_optimized_embed
        )

    def _get_vision_func(self):
        """
        获取vision函数
        关键：支持 messages 参数（RAG-Anything + VLM增强查询需要）
        """

        async def vision_func(
                prompt,
                system_prompt=None,
                history_messages=None,
                image_data=None,
                messages=None,  # 新增：支持VLM增强查询
                **kwargs
        ):
            client = openai.AsyncOpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=90.0
            )

            try:
                #  优先级1：VLM增强查询（messages格式）
                if messages:
                    logger.debug("使用VLM增强查询模式（messages格式）")
                    response = await client.chat.completions.create(
                        model=settings.vision_model,
                        messages=messages,  # 直接使用RAG-Anything构建的messages
                        temperature=0.1,
                        max_tokens=2000,
                        **kwargs
                    )
                    result = response.choices[0].message.content
                    await client.close()
                    return result

                # 优先级2：单图分析（入库时用）
                elif image_data:
                    logger.debug("使用单图分析模式")
                    response = await client.chat.completions.create(  # type: ignore
                        model=settings.vision_model,
                        messages=[
                            {"role": "system", "content": system_prompt or "分析图片并返回JSON格式结果"},
                            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url",
                                                                                            "image_url": {
                                                                                                "url": f"data:image/jpeg;base64,{image_data}"}}]}
                        ],
                        temperature=0.1,
                        max_tokens=1000,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "image_analysis",
                                "schema": {
                                    "type": "object",
                                    "required": ["detailed_description", "entity_info"],
                                    "properties": {
                                        "detailed_description": {"type": "string"},
                                        "entity_info": {
                                            "type": "object",
                                            "required": ["entity_name", "entity_type", "summary"],
                                            "properties": {
                                                "entity_name": {"type": "string"},
                                                "entity_type": {"type": "string"},
                                                "summary": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    )

                    raw_content = response.choices[0].message.content
                    await client.close()

                    # 解析响应
                    parsed = self._smart_parse_response(raw_content)
                    return parsed if parsed else self._create_default_response()

                # 优先级3：纯文本
                else:
                    logger.debug("使用纯文本模式")
                    response = await client.chat.completions.create(  # type: ignore
                        model=settings.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt or ""},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2000,
                        **kwargs
                    )
                    result = response.choices[0].message.content
                    await client.close()
                    return result

            except Exception as e:
                logger.error(f"Vision/LLM调用失败: {e}")
                await client.close()
                return self._create_default_response()

        return vision_func

    def _smart_parse_response(self, response: str) -> Optional[str]:
        """智能解析不同模型的响应格式"""
        if not response or not response.strip():
            return None

        response = response.strip()

        # 清理GLM特殊标记
        if "<|begin_of_box|>" in response or "<|end_of_box|>" in response:
            response = re.sub(r'<\|begin_of_box\|>', '', response)
            response = re.sub(r'<\|end_of_box\|>', '', response)
            response = response.strip()

        # 清理markdown标记
        if "```json" in response or "```" in response:
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0]
                    response = json_part.strip()
            elif "```" in response:
                parts = response.split("```")
                if len(parts) > 1:
                    response = parts[1].strip()

        # 尝试解析JSON
        try:
            data = json.loads(response)
            validated = self._validate_response_data(data)
            if validated:
                return json.dumps(validated, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON对象
        json_patterns = [
            r'\{[^{}]*"detailed_description"[^{}]*:[^{}]*"[^"]*"[^{}]*\}(?:[^{}]*\{[^{}]*\}[^{}]*)*',
            r'\{.*?"detailed_description".*?\}(?:.*?\{.*?\})*',
            r'\{[^}]+\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    fixed = self._fix_json_issues(match)
                    data = json.loads(fixed)
                    validated = self._validate_response_data(data)
                    if validated:
                        return json.dumps(validated, ensure_ascii=False)
                except:
                    continue

        # 从文本构建JSON
        desc = self._extract_description(response)
        if desc:
            return self._build_response_from_text(desc)

        return None

    def _fix_json_issues(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""
        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')

        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        return json_str

    def _extract_description(self, text: str) -> Optional[str]:
        """从文本中提取描述内容"""
        patterns = [
            r'"detailed_description"\s*:\s*"([^"]+)"',
            r'detailed_description[:\s]+([^,}]+)',
            r'"description"\s*:\s*"([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                if desc and len(desc) > 5:
                    return desc

        if "家具" in text or "茶几" in text or "床" in text or "材质" in text:
            clean_text = re.sub(r'[{}\[\]":]', ' ', text)
            clean_text = ' '.join(clean_text.split())
            if len(clean_text) > 10:
                return clean_text[:200]

        return None

    def _validate_response_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """验证和修复响应数据"""
        if not isinstance(data, dict):
            return None

        desc = data.get("detailed_description", "").strip()

        if not desc or desc == "''":
            alternatives = [
                data.get("description", ""),
                data.get("content", ""),
                data.get("analysis", ""),
                data.get("entity_info", {}).get("summary", ""),
                data.get("entity_info", {}).get("description", ""),
            ]

            for alt in alternatives:
                if alt and len(str(alt).strip()) > 5:
                    desc = str(alt).strip()
                    break

        if not desc or len(desc) < 5:
            return None

        result = {
            "detailed_description": desc,
            "entity_info": data.get("entity_info", {})
        }

        if not result["entity_info"]:
            result["entity_info"] = {}

        if not result["entity_info"].get("entity_name"):
            result["entity_info"]["entity_name"] = "产品图片"

        if not result["entity_info"].get("entity_type"):
            result["entity_info"]["entity_type"] = "image"

        if not result["entity_info"].get("summary"):
            result["entity_info"]["summary"] = desc[:100] if len(desc) > 100 else desc

        return result

    def _build_response_from_text(self, text: str) -> str:
        """从文本构建响应"""
        return json.dumps({
            "detailed_description": text,
            "entity_info": {
                "entity_name": "产品图片",
                "entity_type": "image",
                "summary": text[:100] if len(text) > 100 else text
            }
        }, ensure_ascii=False)

    def _create_default_response(self) -> str:
        """创建默认响应"""
        return json.dumps({
            "detailed_description": "产品图片，展示产品的设计和材质特征",
            "entity_info": {
                "entity_name": "产品",
                "entity_type": "image",
                "summary": "产品展示"
            }
        }, ensure_ascii=False)

    async def ensure_initialized(self):
        """确保已初始化"""
        if not self.initialized:
            await self.initialize()

    async def process_multimodal_content(self, modal_content: Dict[str, Any],
                                         entity_name: str, file_path: str,
                                         image_manager=None):
        """
        处理多模态内容
        """
        await self.ensure_initialized()

        logger.info(f"处理商品: {entity_name}")

        try:
            # 获取所有图片路径
            img_path_dict = modal_content.get("img_path", {})

            # 统计
            total_images = len(img_path_dict)
            processed_count = 0
            skipped_count = 0

            # 记录当前商品已处理的图片哈希
            current_item_hashes = set()

            # 逐一处理每张图片
            for img_key, local_path in img_path_dict.items():
                if not local_path or not isinstance(local_path, str):
                    continue

                # 检查是否已处理
                skip_reason = None
                content_hash = None

                if image_manager:
                    for orig_url, mapping in image_manager.mappings.items():
                        if mapping.local_path == local_path:
                            content_hash = mapping.content_hash
                            break

                if content_hash:
                    if content_hash in self.processed_content_hashes:
                        skip_reason = f"全局已处理 (hash: {content_hash[:8]})"
                    elif content_hash in current_item_hashes:
                        skip_reason = f"商品内重复 (hash: {content_hash[:8]})"
                    else:
                        current_item_hashes.add(content_hash)
                        self.processed_content_hashes.add(content_hash)

                if skip_reason:
                    skipped_count += 1
                    logger.debug(f"跳过 {img_key}: {skip_reason}")
                    continue

                # 处理图片
                logger.info(f"处理图片 [{entity_name}] {img_key}: {Path(local_path).name}")

                # 构建单张图片内容
                single_image_content = {
                    "img_path": local_path,
                    "img_caption": modal_content.get("img_caption", []),
                    "img_footnote": modal_content.get("img_footnote", [])
                }

                try:
                    result = await self.image_processor.process_multimodal_content(
                        modal_content=single_image_content,
                        content_type="image",
                        file_path=f"{file_path}_{img_key}",
                        entity_name=f"{entity_name} - {img_key}"
                    )

                    if result:
                        processed_count += 1
                        logger.info(f"图片处理成功: {img_key}")

                except Exception as e:
                    logger.error(f"处理失败 {img_key}: {e}")
                    logger.exception(e)
                    continue

            # 处理完成后增加计数
            self.processed_items_count += 1

            if self.processed_items_count % 20 == 0:
                logger.info(f"总体进度：已处理 {self.processed_items_count} 个商品")

            logger.info(
                f"商品处理完成: {entity_name}, "
                f"处理 {processed_count} 张，跳过 {skipped_count} 张（共 {total_images} 张）"
            )

            return True

        except Exception as e:
            logger.error(f"多模态处理失败: {e}")
            logger.exception(e)
            return False

    def clear_cache_if_needed(self):
        """定期清理缓存避免内存溢出"""
        if len(self.embedding_cache) > 1000:
            items = list(self.embedding_cache.items())
            self.embedding_cache = dict(items[-500:])
            logger.info("缓存已清理，保留最近500条记录")

    # ==================== 文档数据录入（新增功能） ====================

    async def insert_multimodal_content(
            self,
            content_list: list[dict],
            doc_id: str = None
    ):
        """
        直接插入多模态内容列表（跳过解析）

        适用场景：
        - 已经解析好的内容
        - 来自外部源的数据
        - 需要自定义处理的数据

        Args:
            content_list: 内容列表，格式如下：
                [
                    {
                        "type": "text",
                        "content": "文本内容"
                    },
                    {
                        "type": "image",
                        "content": "图片路径或描述",
                        "metadata": {...}
                    },
                    ...
                ]
        """
        # 使用lightrag_instance的insert_content_direct方法
        if hasattr(self.lightrag_instance, 'insert_content_direct'):
            await self.lightrag_instance.insert_content_direct(
                content_list=content_list,
                doc_id=doc_id or f"{self.business_id}_custom"
            )
        else:
            # 如果没有insert_content_direct，使用RAG-Anything处理
            logger.warning("LightRAG没有insert_content_direct方法，使用RAG-Anything处理")
            raise NotImplementedError("LightRAG不支持直接插入内容，请使用RAG-Anything处理文档")
    # ==================== 查询====================

    async def aquery_with_history(
            self,
            query: str,
            history=None,  # List[ChatMessage] = None,
            mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid"
    ) -> str:
        """带历史的查询"""
        if history is None:
            history = field(default_factory=list)
        await self.ensure_initialized()
        
        # ✅ 关键：确认使用的working_dir
        logger.info(f"[{self.business_id}] 执行查询，working_dir: {Path(self.working_dir).absolute()}, mode: {mode}, 历史记录数: {len(history) if history else 0}")
        
        # prompt = config.runtime_prompt_patch.rag_response
        
        # ✅ 关键：在user_prompt中包含business_id，确保缓存键包含业务信息
        # 这样不同业务的相似查询不会互相干扰
        user_prompt_with_business = f"请基于{self.business_id}业务的知识库回答问题。请用简洁自然的方式回答问题。"
        
        result = await self.lightrag_instance.aquery(query=query,
                                                   param=QueryParam(conversation_history=history,
                                                                    mode=mode,
                                                                    # only_need_context=True, # 检索的内容
                                                                    # only_need_prompt=True, # 输入LLM的提示词
                                                                    user_prompt=user_prompt_with_business,
                                                                    top_k=5),
                                                   system_prompt=config.runtime_prompt_patch.system_prompt,
                                                   )
        return post_process_response_urls(result)

    async def aquery_multimodal_with_history(self,
                                             query: str,
                                             user_images: List[str] = None,
                                             history=None,
                                             mode: Literal[
                                                 "local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid") -> \
    Dict[str, Any]:
        """
        多模态查询 - 用户图片 + 库中图片

        Args:
            query: 文本查询
            user_images: 用户上传的图片base64列表
            mode: 查询模式
            history:历史记录

        Returns:
            {
                "result": str,
                "library_images_count": int
            }
        """
        if history is None:
            history = field(default_factory=list)
        await self.ensure_initialized()

        logger.info(f"多模态查询: query='{query[:50]}...', 用户图片={len(user_images) if user_images else 0}张")

        try:
            # 步骤1: 分析用户图片（如果有）
            user_descriptions = []
            if user_images:
                for idx, img_base64 in enumerate(user_images):
                    desc = await self._analyze_user_image(img_base64, idx)
                    if desc:
                        user_descriptions.append(desc)

            # 步骤2: 构建增强查询
            enhanced_query = query
            if user_descriptions:
                enhanced_query = f"""{query}

                **用户提供的参考图片特征**:
                {chr(10).join(user_descriptions)}

                **要求**: 优先推荐与参考图风格、材质、设计相似的产品。"""

            # 步骤3: 获取检索prompt（包含库中图片路径）
            query_param = QueryParam(mode=mode, only_need_prompt=True)
            raw_prompt = await self.lightrag_instance.aquery(enhanced_query, param=query_param)

            # 步骤4: 提取并编码库中图片 TODO：可能没有
            enhanced_prompt, library_images = await self._extract_images_from_prompt(raw_prompt)

            logger.info(f"检索到 {len(library_images)} 张产品图片")

            # 步骤5: VLM综合分析（用户图 + 库中图）
            if user_images or library_images:
                result = await self._vlm_analyze_all_images(
                    prompt=enhanced_prompt,
                    query=enhanced_query,
                    user_images=user_images or [],
                    library_images=library_images
                )
            else:
                # 纯文本查询
                result = await self.lightrag_instance.aquery(query=enhanced_query,
                                                             param=QueryParam(conversation_history=history,
                                                                              mode=mode,
                                                                              user_prompt="请用简洁自然的方式回答问题",
                                                                              top_k=5),
                                                             )

            # 步骤6: 后处理URL
            result = post_process_response_urls(result)

            return {
                "result": result,
                "library_images_count": len(library_images)
            }

        except Exception as e:
            logger.error(f"多模态查询失败: {e}")
            logger.exception(e)
            raise

    # ==================== 辅助方法：VLM调用 ====================
    def _extract_and_encode_images(self, prompt: str):
        """从 prompt 提取图片路径并编码"""
        images_base64 = []

        # 正则提取 Image Path: xxx.jpg
        pattern = r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png))"
        matches = re.findall(pattern, prompt, re.IGNORECASE)

        for image_path in matches:
            if Path(image_path).exists():
                base64_str = self.image_optimizer.get_base64_optimized(image_path)
                images_base64.append(base64_str)

        # 替换路径为标记
        enhanced = re.sub(pattern, lambda m: f"[IMAGE_{len(images_base64)}]", prompt)

        return enhanced, images_base64

    async def _call_vlm_with_images(self, prompt: str, query: str, images: List[str]) -> str:
        """VLM看库中图片（纯文本查询时用）"""
        content_parts = []

        # 添加检索文本
        content_parts.append({"type": "text", "text": f"=== 检索到的产品 ({len(images)}张) ==="})

        # 添加所有图片
        for idx, img_base64 in enumerate(images):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
            content_parts.append({"type": "text", "text": f"[产品图{idx + 1}]"})

        # 添加上下文和问题
        content_parts.append({"type": "text", "text": f"""
            \n=== 检索上下文 ===
            {prompt}

            === 用户问题 ===
            {query}

            === 回答要求 ===
            请基于检索到的产品图片和上下文信息，为用户提供详细的产品推荐：
            1. 推荐最匹配的产品（名称、特点）
            2. 每个推荐产品都要用Markdown格式显示图片：`![产品名](图片URL)`
            3. 说明推荐理由
            4. 提供选购建议

            请用Markdown格式回答。
            """})

        vision_func = self._get_vision_func()

        result = await vision_func(
            "",
            messages=[
                {"role": "system", "content": "你是产品的专业导购"},
                {"role": "user", "content": content_parts}
            ]
        )

        return result

    async def _analyze_user_image(self, image_base64: str, index: int) -> str:
        """分析用户上传的图片"""
        try:
            vision_func = self._get_vision_func()

            prompt = """分析这张产品图片，提取关键特征：
                    1. 产品类型
                    2. 风格
                    3. 材质特征
                    4. 颜色和造型
                    5. 适用场景
                    6. 产品主体细节

                    用简洁中文描述，便于匹配相似产品。"""

            result = await vision_func(
                prompt=prompt,
                system_prompt="你是产品分析专家",
                image_data=image_base64
            )

            if result:
                try:
                    data = json.loads(result)
                    desc = data.get("detailed_description", "")
                except:
                    desc = result if isinstance(result, str) else ""

                if desc and len(desc) > 10:
                    return f"[参考图{index + 1}]: {desc}"

            return ""

        except Exception as e:
            logger.error(f"分析用户图片失败: {e}")
            return ""

    async def _extract_images_from_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """从检索prompt提取图片"""
        import re

        images_base64 = []
        pattern = r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp))"

        matches = re.findall(pattern, prompt, re.IGNORECASE)

        for img_path in matches:
            if Path(img_path).exists():
                try:
                    base64_str = self.image_optimizer.get_base64_optimized(img_path)
                    images_base64.append(base64_str)
                except Exception as e:
                    logger.error(f"编码图片失败 {img_path}: {e}")

        # 替换路径为标记
        enhanced = prompt
        for idx in range(len(images_base64)):
            enhanced = re.sub(pattern, f"[产品图{idx + 1}]", enhanced, count=1)

        return enhanced, images_base64

    async def _vlm_analyze_all_images(self,
                                      prompt: str,
                                      query: str,
                                      user_images: List[str],
                                      library_images: List[str]) -> str:
        """VLM分析所有图片"""

        content_parts = []

        # 添加用户图
        if user_images:
            content_parts.append({"type": "text", "text": f"=== 用户参考图 ({len(user_images)}张) ==="})
            for idx, img in enumerate(user_images):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
                content_parts.append({"type": "text", "text": f"[用户图{idx + 1}]"})

        # 添加库中图
        if library_images:
            content_parts.append({"type": "text", "text": f"\n=== 产品图 ({len(library_images)}张) ==="})
            for idx, img in enumerate(library_images):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
                content_parts.append({"type": "text", "text": f"[产品图{idx + 1}]"})

        # 添加检索文本和任务
        content_parts.append({"type": "text", "text": f"""
            \n=== 检索上下文 ===
            {prompt}

            === 用户问题 ===
            {query}

            === 回答要求 ===
            请基于以上信息推荐产品：
            1. 对比参考图与产品图，说明相似之处
            2. 推荐最匹配的产品，包含产品名、特点、图片
            3. 图片用Markdown格式: ![产品名](图片路径)
            4. 提供选购建议

            用Markdown格式回答。
            """})

        vision_func = self._get_vision_func()

        result = await vision_func(
            "",
            messages=[
                {"role": "system", "content": "你是产品的专业导购"},
                {"role": "user", "content": content_parts}
            ]
        )

        return result

    async def aquery_stream(self, query: str,
                            business_id: str,
                            mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
                            history=None, ):
        """
        流式查询 - 纯文本
        """
        if history is None:
            history = field(default_factory=list)
        await self.ensure_initialized()

        logger.info(f"流式文本查询: {query[:50]}...")

        try:
            # 1. 获取检索prompt
            query_param = QueryParam(mode=mode, only_need_prompt=True, conversation_history=history,response_type="Single Paragraph")  # 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'
            raw_prompt = await self.lightrag_instance.aquery(query, param=query_param,system_prompt=config.runtime_prompt_patch.system_prompt)

            # 无图片，直接调用LLM
            async for chunk in self._call_llm_stream(raw_prompt, business_id):
                yield chunk

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"\n\n❌ 错误: {str(e)}"

    async def aquery_multimodal_stream(self,
                                       query: str,
                                       business_id: str,
                                       user_images: List[str] = None,
                                       mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
                                       history=None):
        """
        流式多模态查询
        """
        if history is None:
            history = field(default_factory=list)
        await self.ensure_initialized()

        logger.info(f"流式多模态查询: {query[:50]}..., 用户图片={len(user_images) if user_images else 0}张")

        try:
            # 1. 分析用户图片
            user_descriptions = []
            if user_images:
                for idx, img_base64 in enumerate(user_images):
                    desc = await self._analyze_user_image(img_base64, idx)
                    if desc:
                        user_descriptions.append(desc)

            # 2. 构建增强查询
            enhanced_query = query
            if user_descriptions:
                enhanced_query = f"""{query}

                **用户提供的参考图片特征**:
                {chr(10).join(user_descriptions)}
            
                **要求**: 优先推荐与参考图风格、材质、设计相似的产品。"""

            # 3. 获取检索prompt
            query_param = QueryParam(mode=mode, only_need_prompt=True, conversation_history=history)
            raw_prompt = await self.lightrag_instance.aquery(enhanced_query, param=query_param)

            # 4. 提取库中图片
            enhanced_prompt, library_images = await self._extract_images_from_prompt(raw_prompt)

            logger.info(f"检索到 {len(library_images)} 张产品图片")

            # 5. 流式VLM分析
            if user_images or library_images:
                async for chunk in self._vlm_analyze_all_images_stream(
                        prompt=enhanced_prompt,
                        query=query,
                        user_images=user_images or [],
                        library_images=library_images
                ):
                    yield chunk
            else:
                async for chunk in self._call_llm_stream(enhanced_query,business_id):
                    yield chunk

        except Exception as e:
            logger.error(f"流式多模态查询失败: {e}")
            yield f"\n\n❌ 错误: {str(e)}"

    #  核心流式方法
    async def _call_llm_stream(self, prompt: str, business_id: str):
        """流式调用LLM"""
        client = openai.AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=90.0
        )
        core_system = Dependencies.get_core_system()
        business_name = core_system.businesses.get(business_id).name or "产品"

        try:
            stream = await client.chat.completions.create(  # type: ignore
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": f"你是{business_name}专业的智能助手,输出自然精炼"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.75,
                stream=True  # 关键：启用流式
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content

        finally:
            await client.close()

    async def _call_vlm_with_images_stream(self, prompt: str, query: str, images: List[str]):
        """流式VLM（库中图片）"""
        content_parts = []

        content_parts.append({"type": "text", "text": f"=== 检索到的产品 ({len(images)}张) ==="})

        for idx, img_base64 in enumerate(images):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
            content_parts.append({"type": "text", "text": f"[产品图{idx + 1}]"})

        content_parts.append({"type": "text", "text": f"""
    \n=== 检索上下文 ===
    {prompt}

    === 用户问题 ===
    {query}

    === 回答要求 ===
    请基于检索到的产品图片和上下文信息提供详细推荐，用Markdown格式回答。
    """})

        client = openai.AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=90.0
        )

        try:
            stream = await client.chat.completions.create(  # type: ignore
                model=settings.vision_model,
                messages=[
                    {"role": "system", "content": "你是侘寂家具的专业导购,输出自然精炼"},
                    {"role": "user", "content": content_parts}
                ],
                temperature=0.75,
                stream=True  # 流式
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # URL后处理
                    processed = post_process_response_urls(content)
                    yield processed

        finally:
            await client.close()

    async def _vlm_analyze_all_images_stream(self,
                                             prompt: str,
                                             query: str,
                                             user_images: List[str],
                                             library_images: List[str]):
        """流式VLM分析所有图片"""
        content_parts = []

        # 添加用户图
        if user_images:
            content_parts.append({"type": "text", "text": f"=== 用户参考图 ({len(user_images)}张) ==="})
            for idx, img in enumerate(user_images):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
                content_parts.append({"type": "text", "text": f"[用户参考图{idx + 1}]"})

        # 添加库中图
        if library_images:
            content_parts.append({"type": "text", "text": f"\n=== 产品图 ({len(library_images)}张) ==="})
            for idx, img in enumerate(library_images):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
                content_parts.append({"type": "text", "text": f"[产品图{idx + 1}]"})

        content_parts.append({"type": "text", "text": f"""
        \n=== 检索上下文 ===
        {prompt}
    
        === 用户问题 ===
        {query}
    
        === 回答要求 ===
        请基于以上信息推荐产品，对比用户参考图与产品图，用Markdown格式回答。
        """})

        client = openai.AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=90.0
        )

        try:
            stream = await client.chat.completions.create(  # type: ignore
                model=settings.vision_model,
                messages=[
                    {"role": "system", "content": "你是侘寂家具的专业产品顾问"},
                    {"role": "user", "content": content_parts}
                ],
                temperature=0.1,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # URL后处理
                    processed = post_process_response_urls(content)
                    yield processed

        finally:
            await client.close()
    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ========== 清理方法 ==========
    def cleanup(self):
        """清理资源（预留，当前无需清理）"""
        # 当前所有资源都由RAG-Anything和LightRAG管理，无需手动清理
        pass

    async def insert_document(
            self,
            file_path: str,
            doc_type: str = "manual",
            use_gpu: Optional[bool] = None,
            start_page: Optional[int] = None,
            end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        ✅ 文档录入（使用 RAGAnything 的完整流程）

        调用链：
        process_document_complete → 解析 → 多模态处理 → 存储到 LightRAG
        """
        if not self.rag_anything:
            return {
                "status": "error",
                "message": "RAGAnything 未启用"
            }

        total_start = time.perf_counter()
        selected_parse_method: str = settings.rag_anything.parse_method or "auto"
        pdf_stats: Optional[Dict[str, Any]] = None

        try:
            logger.info(f"开始处理文档: {file_path}")

            # GPU 策略
            gpu_available = self._check_gpu_available()
            if use_gpu is None:
                gpu_enabled = gpu_available
            else:
                gpu_enabled = bool(use_gpu) and gpu_available
                if use_gpu and not gpu_available:
                    logger.warning("请求使用GPU解析，但未检测到可用GPU，自动回退到CPU")

            logger.info(f"文档解析将使用 {'GPU' if gpu_enabled else 'CPU'}")

            # 智能选择解析方式
            selected_parse_method, pdf_stats = await self._resolve_parse_method(
                file_path=file_path,
                configured_method=settings.rag_anything.parse_method
            )

            # ✅ 统一使用RAG-Anything处理所有文档（包括TXT/MD）
            # RAG-Anything会根据parse_method自动选择处理方式：
            # - parse_method="txt" → 直接读取文本，不转PDF
            # - parse_method="auto"/"mineru" → 使用MinerU完整解析
            file_suffix = Path(file_path).suffix.lower()
            if file_suffix in {".txt", ".md"}:
                logger.info(f"检测到纯文本文件 {file_suffix}，使用RAG-Anything的txt模式处理（统一流程）")

            # 输出解析目录
            output_dir = Path(self.working_dir) / "parsed" / Path(file_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            parser_kwargs = self._build_parser_kwargs(
                device="cuda" if gpu_enabled else "cpu",
                start_page=start_page,
                end_page=end_page
            )

            # ✅ 生成唯一的文档ID（添加时间戳避免重复）
            # 格式：{business_id}:{filename}_{timestamp}
            # 这样可以支持同名文件多次上传
            file_stem = Path(file_path).stem
            timestamp = int(time.time() * 1000)  # 毫秒时间戳
            doc_id = f"{self.business_id}:{file_stem}_{timestamp}"

            result = await self.rag_anything.process_document_complete(
                file_path=file_path,
                output_dir=str(output_dir),
                parse_method=selected_parse_method,
                doc_id=doc_id,
                **parser_kwargs,
            )

            # ✅ 关键优化：注册文档解析产生的图片路径映射
            # 图片保持在parsed目录下，只注册URL映射，不迁移文件
            await self._register_parsed_images(output_dir, doc_id)

            total_time = time.perf_counter() - total_start

            chunks_inserted = 0
            parse_time = None
            insert_time = None
            success = True

            if isinstance(result, dict):
                chunks_inserted = int(result.get("chunks_inserted", 0) or 0)
                parse_time = result.get("parse_time")
                insert_time = result.get("insert_time")
                success = bool(result.get("success", True))

            logger.info(
                f"✅ 文档处理完成: {Path(file_path).name} "
                f"(method={selected_parse_method}, chunks={chunks_inserted}, total={total_time:.2f}s)"
            )

            response: Dict[str, Any] = {
                "status": "success" if success else "partial",
                "file": Path(file_path).name,
                "doc_type": doc_type,
                "parse_method": selected_parse_method,
                "chunks_inserted": chunks_inserted,
                "total_time": round(total_time, 2),
                "device": "cuda" if gpu_enabled else "cpu",
                "entity_extract_rounds": max(0, settings.rag_anything.entity_extract_rounds),
            }

            if parse_time is not None:
                response["parse_time"] = round(parse_time, 2)
            if insert_time is not None:
                response["insert_time"] = round(insert_time, 2)
            if pdf_stats:
                response["pdf_stats"] = pdf_stats

            return response

        except Exception as e:
            logger.error(f"❌ 文档处理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "file": Path(file_path).name,
                "parse_method": selected_parse_method,
                "error": str(e)
            }

    # ✅ 批量处理也很简单
    async def insert_document_batch(
            self,
            file_paths: List[str],
            doc_type: str = "manual",
            use_gpu: bool = False
    ) -> Dict[str, Any]:
        """批量文档录入"""
        if not self.rag_anything:
            return {
                "status": "error",
                "message": "RAGAnything 未启用"
            }

        try:
            logger.info(f"开始批量处理 {len(file_paths)} 个文档")

            results = []
            success_count = 0

            for file_path in file_paths:
                result = await self.insert_document(
                    file_path=file_path,
                    doc_type=doc_type,
                    use_gpu=use_gpu
                )

                results.append(result)

                if result.get("status") == "success":
                    success_count += 1

            logger.info(f"批量处理完成: 成功 {success_count}/{len(file_paths)}")

            return {
                "status": "success",
                "total": len(file_paths),
                "success": success_count,
                "failed": len(file_paths) - success_count,
                "results": results
            }

        except Exception as e:
            logger.error(f"批量处理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def _resolve_parse_method(
            self,
            file_path: str,
            configured_method: Optional[str]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        根据配置与文档特征动态选择解析方式。
        返回 (parse_method, stats)
        """
        method = (configured_method or "auto").lower()

        if not settings.rag_anything.smart_parse:
            return method or "auto", None

        suffix = Path(file_path).suffix.lower()

        # 显式指定解析方式时直接返回
        if method in {"txt", "ocr"}:
            return method, None

        # 纯文本文件直接走txt
        if suffix in {".txt", ".md"}:
            return "txt", None

        if suffix != ".pdf":
            return method or "auto", None

        stats = await asyncio.to_thread(self._collect_pdf_stats, file_path)
        if not stats:
            return method or "auto", None

        stats["configured_method"] = method or "auto"

        qualifies = (
            stats["page_count"] <= settings.rag_anything.max_txt_pages and
            stats["file_size_mb"] <= settings.rag_anything.max_txt_file_mb and
            stats["avg_chars"] >= settings.rag_anything.text_density_threshold and
            stats["image_ratio"] <= settings.rag_anything.image_page_ratio_threshold and
            stats["text_ratio"] >= 0.6  # 60% 采样页具备有效文本
        )

        stats["qualifies_txt"] = qualifies

        if qualifies:
            logger.info(
                "PDF 文档检测为文本主导，自动使用 txt 解析。"
                f" pages={stats['page_count']}, avg_chars={stats['avg_chars']:.1f}, "
                f"image_ratio={stats['image_ratio']:.2f}, text_ratio={stats['text_ratio']:.2f}"
            )
            stats["decision"] = "txt"
            return "txt", stats

        stats["decision"] = method or "auto"
        return method or "auto", stats

    def _collect_pdf_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """采样 PDF 文档页，评估文本密度与图片占比。"""
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            logger.warning(f"导入 pypdf 失败，无法进行PDF智能解析: {exc}")
            return None

        try:
            reader = PdfReader(file_path)
        except Exception as exc:
            logger.warning(f"PDF 预分析失败，跳过智能解析: {exc}")
            return None

        page_count = len(reader.pages)
        if page_count == 0:
            return None

        sample_limit = max(1, min(settings.rag_anything.sample_page_limit, page_count))
        indices = set()
        if page_count <= sample_limit:
            indices = set(range(page_count))
        else:
            step = max(page_count // sample_limit, 1)
            indices = {min(i, page_count - 1) for i in range(0, page_count, step)}
            extra_idx = 0
            while len(indices) < sample_limit and extra_idx < page_count:
                indices.add(extra_idx)
                extra_idx += 1

        if not indices:
            indices = {0}

        text_chars_total = 0
        text_pages = 0
        image_pages = 0

        for idx in sorted(indices):
            page = reader.pages[idx]
            text = ""
            try:
                extracted = page.extract_text()
                if extracted:
                    text = extracted.strip()
            except Exception:
                text = ""

            char_count = len(text)
            text_chars_total += char_count
            if char_count >= settings.rag_anything.min_text_chars:
                text_pages += 1

            if self._page_has_image(page):
                image_pages += 1

        sampled_pages = len(indices)
        avg_chars = text_chars_total / sampled_pages if sampled_pages else 0
        image_ratio = image_pages / sampled_pages if sampled_pages else 0.0
        text_ratio = text_pages / sampled_pages if sampled_pages else 0.0
        file_size_mb = round(Path(file_path).stat().st_size / (1024 ** 2), 3)

        return {
            "page_count": page_count,
            "sampled_pages": sampled_pages,
            "avg_chars": avg_chars,
            "total_chars": text_chars_total,
            "text_pages": text_pages,
            "image_pages": image_pages,
            "image_ratio": image_ratio,
            "text_ratio": text_ratio,
            "file_size_mb": file_size_mb,
        }

    @staticmethod
    def _page_has_image(page) -> bool:
        """判断 PDF 页面是否包含图片资源。"""
        try:
            images = getattr(page, "images", None)
            if images and len(images) > 0:
                return True
        except Exception:
            pass

        try:
            resources = page.get("/Resources")
            if not resources:
                return False
            xobjects = resources.get("/XObject")
            if not xobjects:
                return False
            for obj in xobjects.values():
                try:
                    subtype = obj.get("/Subtype")
                    name = getattr(subtype, "name", subtype)
                    if name in ("/Image", "Image"):
                        return True
                except Exception:
                    continue
        except Exception:
            return False

        return False

    def _build_parser_kwargs(
            self,
            device: str,
            start_page: Optional[int],
            end_page: Optional[int]
    ) -> Dict[str, Any]:
        """构建 MinerU 解析参数。"""
        kwargs: Dict[str, Any] = {"device": device}

        if start_page is not None and start_page >= 0:
            kwargs["start_page"] = int(start_page)
        if end_page is not None and end_page >= 0:
            kwargs["end_page"] = int(end_page)

        kwargs["formula"] = bool(settings.rag_anything.enable_formula)
        kwargs["table"] = bool(settings.rag_anything.enable_table)

        if settings.rag_anything.backend:
            kwargs["backend"] = settings.rag_anything.backend
        if settings.rag_anything.lang:
            kwargs["lang"] = settings.rag_anything.lang

        return kwargs

    async def _register_parsed_images(self, parsed_output_dir: Path, doc_id: str):
        """
        注册文档解析产生的图片路径映射（图片保持在parsed目录下，不迁移）
        
        新的目录结构：
        rag_storage_{business_id}/
        ├── images/              ← 结构化录入的图片
        ├── parsed/              ← 文档解析的图片
        │   ├── doc1/
        │   │   └── images/
        │   └── doc2/
        │       └── images/
        
        Args:
            parsed_output_dir: RAGAnything解析输出目录（如 rag_storage_ARglasses/parsed/M400开发文档/...）
            doc_id: 文档ID（如 ARglasses:M400开发文档）
        """
        try:
            parsed_dir = Path(parsed_output_dir)
            
            # 查找所有images目录（RAGAnything可能在多层目录下创建images）
            images_dirs = []
            for root, dirs, files in os.walk(parsed_dir):
                if 'images' in dirs:
                    images_dirs.append(Path(root) / 'images')
            
            if not images_dirs:
                logger.debug(f"未找到图片目录: {parsed_dir}")
                return
            
            registered_count = 0
            
            for images_dir in images_dirs:
                if not images_dir.exists():
                    continue
                
                # 遍历所有图片文件
                for img_file in images_dir.glob("*"):
                    if not img_file.is_file():
                        continue
                    
                    # 检查文件扩展名
                    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                        continue
                    
                    # ✅ 关键：构建相对于working_dir的路径
                    # 例如：rag_storage_ARglasses/parsed/M400-AR智能眼镜/M400-AR智能眼镜/auto/images/xxx.jpg
                    try:
                        # 获取相对于working_dir的路径
                        rel_path = img_file.relative_to(Path(self.working_dir))
                        # 转换为URL路径格式（使用正斜杠）
                        url_path = str(rel_path).replace('\\', '/')
                        
                        # ✅ 关键：确保路径包含rag_storage_{business_id}前缀
                        # working_dir格式：./rag_storage_{business_id} 或 rag_storage_{business_id}
                        working_dir_name = Path(self.working_dir).name
                        if not working_dir_name.startswith('rag_storage_'):
                            # 如果working_dir是相对路径，提取rag_storage_部分
                            working_dir_str = str(Path(self.working_dir)).replace('\\', '/')
                            if 'rag_storage_' in working_dir_str:
                                idx = working_dir_str.find('rag_storage_')
                                working_dir_name = working_dir_str[idx:]
                            else:
                                working_dir_name = f"rag_storage_{self.business_id}"
                        
                        # 构建完整路径：rag_storage_{business_id}/parsed/...
                        full_url_path = f"{working_dir_name}/{url_path}"
                        
                        # URL编码中文字符
                        from urllib.parse import quote
                        path_parts = full_url_path.split('/')
                        encoded_parts = [quote(part, safe='') for part in path_parts]
                        encoded_path = '/'.join(encoded_parts)
                        
                        # 构建远程URL
                        # 格式：http://host/images/rag_storage_{business_id}/parsed/{doc}/images/{filename}
                        remote_url = f"{settings.static_base_url}/images/{encoded_path}"
                        
                        # 注册路径映射（支持多种路径格式）
                        local_path_str = str(img_file)  # 绝对路径
                        local_path_normalized = local_path_str.replace('\\', '/')  # 标准化路径
                        path_manager.register_mapping(local_path_str, remote_url)  # 绝对路径（Windows格式）
                        path_manager.register_mapping(local_path_normalized, remote_url)  # 绝对路径（正斜杠格式）
                        path_manager.register_mapping(url_path, remote_url)  # 相对路径（未编码，相对于working_dir）
                        path_manager.register_mapping(full_url_path, remote_url)  # 完整相对路径（包含rag_storage_前缀，未编码）
                        path_manager.register_mapping(encoded_path, remote_url)  # 完整相对路径（包含rag_storage_前缀，已编码）
                        
                        registered_count += 1
                        logger.debug(f"图片路径已注册: {img_file.name} -> {remote_url}")
                    except ValueError:
                        # 如果路径不在working_dir下，尝试从绝对路径中提取rag_storage_部分
                        local_path_str = str(img_file).replace('\\', '/')
                        if 'rag_storage_' in local_path_str:
                            idx = local_path_str.find('rag_storage_')
                            if idx >= 0:
                                rel_path = local_path_str[idx:]
                                # URL编码
                                from urllib.parse import quote
                                path_parts = rel_path.split('/')
                                encoded_parts = [quote(part, safe='') for part in path_parts]
                                encoded_path = '/'.join(encoded_parts)
                                remote_url = f"{settings.static_base_url}/images/{encoded_path}"
                                # 注册绝对路径映射
                                path_manager.register_mapping(local_path_str, remote_url)
                                path_manager.register_mapping(str(img_file), remote_url)
                                registered_count += 1
                                logger.debug(f"图片路径已注册（绝对路径）: {img_file.name} -> {remote_url}")
                                continue
                        logger.warning(f"图片路径不在working_dir下且无法提取相对路径: {img_file}")
                        continue
            
            if registered_count > 0:
                logger.info(f"✅ 已注册 {registered_count} 张文档图片路径映射（保持在parsed目录）")
            else:
                logger.debug(f"未发现需要注册的图片")
                
        except Exception as e:
            logger.warning(f"注册文档图片路径失败: {e}", exc_info=True)
    
    def _patch_rag_anything_font(self, font_path: str):
        """
        Monkey patch RAG-Anything的convert_text_to_pdf方法，使用我们检测到的字体路径
        
        RAG-Anything硬编码了Linux字体路径 `/usr/share/fonts/wqy-microhei/wqy-microhei.ttc`，
        我们通过monkey patch在它找不到Linux路径时，使用我们检测到的字体路径。
        """
        try:
            from raganything.parser import Parser
            from pathlib import Path
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import logging
            
            # 保存原始方法
            original_convert_text_to_pdf = Parser.convert_text_to_pdf
            
            @staticmethod
            def patched_convert_text_to_pdf(text_path, output_dir=None):
                """使用我们检测到的字体路径的convert_text_to_pdf方法"""
                from pathlib import Path as PathType
                from typing import Union, Optional
                import logging
                
                # 在调用原始方法之前，先尝试注册我们检测到的字体
                # 这样如果RAG-Anything找不到Linux路径，我们的字体已经被注册了
                font_name = "WenQuanYi"  # 使用相同的字体名，这样RAG-Anything会直接使用
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    font_file = PathType(font_path)
                    if font_file.exists():
                        try:
                            pdfmetrics.registerFont(TTFont(font_name, str(font_file)))
                            logging.info(f"✅ 成功注册中文字体（monkey patch）: {font_path}")
                        except Exception as e:
                            logging.warning(f"⚠️ 注册字体文件失败: {e}，将使用RAG-Anything的默认逻辑")
                
                # 调用原始方法（它会检查WenQuanYi是否已注册，如果已注册就直接使用）
                result = original_convert_text_to_pdf(text_path, output_dir)
                
                return result
            
            # 替换方法
            Parser.convert_text_to_pdf = patched_convert_text_to_pdf
            logger.info(f"✅ 已为RAG-Anything应用字体monkey patch: {font_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Monkey patch字体失败: {e}，将使用RAG-Anything的默认字体检测逻辑", exc_info=True)

