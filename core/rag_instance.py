# core/rag_instance.py

import base64
import logging
import os
import json
import re
import openai
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Set, List, Tuple, Literal, cast
from lightrag import QueryParam
from lightrag.prompt import PROMPTS
from raganything.modalprocessors import ImageModalProcessor

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
from utils.url_helper import post_process_response_urls
from utils.logger import setup_logger
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

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
            return

        logger.info(f"初始化RAG-Anything实例: {self.business_id}")

        try:
            # 应用中文提示词
            from config.runtime_prompt_patch import apply_chinese_prompts_runtime
            apply_chinese_prompts_runtime()

            await self._create_lightrag()

            # 创建图像处理器
            self._create_image_processor()

            self.initialized = True
            logger.info(f"RAG初始化完成: {self.business_id}")


        except Exception as e:
            logger.error(f"RAG初始化失败: {e}")
            raise

    async def _create_lightrag(self):
        """创建LightRAG实例 """
        kwargs = dict(
            working_dir=self.working_dir,
            embedding_func=self._get_embedding_func(),
            llm_model_func=self._get_llm_func(),
            chunk_token_size=3000, # 增大分块大小
            chunk_overlap_token_size=150, # 减小重叠
            entity_extract_max_gleaning=1, # 减少实体提取轮次
        )

        if settings.rerank.enabled:
            kwargs.update(
                rerank_model_func=self._get_rerank_func(),
                min_rerank_score=settings.rerank.score_threshold,
            )

        self.lightrag_instance = LightRAG(**kwargs)


        await self.lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        logger.info(f"LightRAG创建完成: {self.working_dir}")

    def _create_image_processor(self):
        """创建图像处理器"""
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
        关键：支持 messages 参数（RAG-Anything 1.2.7+ VLM增强查询需要）
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
        # prompt = config.runtime_prompt_patch.rag_response

        result = await self.lightrag_instance.aquery(query=query,
                                                   param=QueryParam(conversation_history=history,
                                                                    mode=mode,
                                                                    # only_need_context=True, # 检索的内容
                                                                    # only_need_prompt=True, # 输入LLM的提示词
                                                                    user_prompt="请用简洁自然的方式回答问题。",
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

                **要求**: 优先推荐与参考图风格、材质、设计相似的产品。输出请忽略References参考文献的部分"""

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
