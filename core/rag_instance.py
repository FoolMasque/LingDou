# core/rag_instance.py
"""
生产环境RAG实例 - 移除ChinesePrompts依赖
"""
from pathlib import Path
import json
import re
from typing import Optional, Dict, Any, Set
import openai
from lightrag import QueryParam
from config.settings import settings
from core.components import ImageOptimizer
from utils.url_helper import post_process_response_urls
from utils.logger import setup_logger
from raganything.modalprocessors import ImageModalProcessor
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.openai import openai_embed

logger = setup_logger(__name__)

_global_openai_client = None


async def get_shared_openai_client():
    """获取共享的OpenAI客户端"""
    global _global_openai_client

    if _global_openai_client is None:
        import openai
        _global_openai_client = openai.AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0
        )

    return _global_openai_client


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
            # 每次初始化时都应用中文提示词
            from config.runtime_prompt_patch import apply_chinese_prompts_runtime
            apply_chinese_prompts_runtime()

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
        """创建LightRAG实例 """

        self.lightrag_instance = LightRAG(
            working_dir=self.working_dir,
            embedding_func=self._get_embedding_func(),
            llm_model_func=self._get_llm_func()
        )

        await self.lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        logger.info(f"LightRAG创建完成: {self.working_dir}")

    def _get_llm_func(self):
        def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
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

    # TODO：无法适配qwen的接口（测试免费版）
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

    def _get_simple_vision_func(self):
        """待废弃：vision函数 - 依赖运行时替换的提示词，无法适配glm模型"""

        async def simple_vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):

            try:
                client = openai.AsyncOpenAI(
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    timeout=60.0
                )

                # 使用传入的system_prompt（已经在运行时被替换为中文）
                messages = [
                    {"role": "system", "content": system_prompt or "你是专业的家具分析师。"}
                ]
                # logger.info(f"<UNK>提示词: {prompt[:50]}")
                if image_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    })
                    model = settings.vision_model
                else:
                    messages.append({"role": "user", "content": prompt})
                    model = settings.llm_model

                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                        max_tokens=600,
                        **kwargs
                    )
                    result = response.choices[0].message.content
                    logger.info(f"<UNK>响应: {result}")
                    return result
                finally:
                    await client.close()

            except Exception as e:
                logger.error(f"Vision调用失败: {e}")
                return f"分析失败: {str(e)}"

        return simple_vision_func

    def _get_vision_func(self):
        """兼容多种模型格式的vision函数"""

        async def multi_model_vision_func(prompt, system_prompt=None,
                                          history_messages=[], image_data=None, **kwargs):

            try:
                client = openai.AsyncOpenAI(
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    timeout=90.0
                )

                # 构建消息
                messages = [
                    {"role": "system", "content": system_prompt or "分析图片并返回JSON格式结果"},
                ]

                if image_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    })
                    model = settings.vision_model
                else:
                    messages.append({"role": "user", "content": prompt})
                    model = settings.llm_model

                # 记录请求
                logger.debug(f"发送Vision请求到模型: {model}")

                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1000,
                        # **kwargs
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
                                            },
                                            "additionalProperties": True
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            }
                        }
                    )
                    logger.info(f"Vision<UNK>: {response}")

                    if response and response.choices:
                        raw_content = response.choices[0].message.content
                        logger.info(f"收到响应，长度: {len(raw_content) if raw_content else 0}")

                        if raw_content:
                            # 记录原始响应用于调试
                            logger.debug(f"原始响应预览: {raw_content[:300]}...")

                            # 使用智能解析器
                            parsed = self._smart_parse_response(raw_content)

                            if parsed:
                                logger.info("响应解析成功")
                                logger.info(f"响应输出{parsed}")
                                return parsed
                            else:
                                logger.warning("响应解析失败，使用默认值")
                                return self._create_default_response()
                        else:
                            logger.warning("响应内容为空")
                            return self._create_default_response()
                    else:
                        logger.warning("无有效响应")
                        return self._create_default_response()

                finally:
                    await client.close()

            except Exception as e:
                logger.error(f"Vision调用失败: {e}")
                return self._create_default_response()

        return multi_model_vision_func

    def _smart_parse_response(self, response: str) -> Optional[str]:
        """智能解析不同模型的响应格式"""

        if not response or not response.strip():
            return None

        response = response.strip()

        # 步骤1: 清理GLM特殊标记
        if "<|begin_of_box|>" in response or "<|end_of_box|>" in response:
            logger.debug("检测到GLM格式，清理特殊标记")
            response = re.sub(r'<\|begin_of_box\|>', '', response)
            response = re.sub(r'<\|end_of_box\|>', '', response)
            response = response.strip()

        # 步骤2: 清理OpenAI的markdown标记
        if "```json" in response or "```" in response:
            logger.debug("检测到markdown格式，提取JSON内容")
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0]
                    response = json_part.strip()
            elif "```" in response:
                parts = response.split("```")
                if len(parts) > 1:
                    response = parts[1].strip()

        # 步骤3: 尝试直接解析JSON
        try:
            data = json.loads(response)
            validated = self._validate_response_data(data)
            if validated:
                return json.dumps(validated, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logger.debug(f"直接JSON解析失败: {e}")

        # 步骤4: 尝试提取JSON对象
        json_patterns = [
            r'\{[^{}]*"detailed_description"[^{}]*:[^{}]*"[^"]*"[^{}]*\}(?:[^{}]*\{[^{}]*\}[^{}]*)*',
            r'\{.*?"detailed_description".*?\}(?:.*?\{.*?\})*',
            r'\{[^}]+\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # 修复可能的JSON问题
                    fixed = self._fix_json_issues(match)
                    data = json.loads(fixed)
                    validated = self._validate_response_data(data)
                    if validated:
                        return json.dumps(validated, ensure_ascii=False)
                except:
                    continue

        # 步骤5: 尝试从片段构建JSON
        desc = self._extract_description(response)
        if desc:
            return self._build_response_from_text(desc)

        return None

    def _fix_json_issues(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""

        # 修复换行符
        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')

        # 修复未转义的引号
        # 这里需要更智能的处理，避免破坏正常的JSON

        # 修复未闭合的括号
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        # 修复尾部逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        return json_str

    def _extract_description(self, text: str) -> Optional[str]:
        """从文本中提取描述内容"""

        # 尝试提取detailed_description的值
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

        # 如果找不到，但文本包含有用信息
        if "家具" in text or "茶桌" in text or "床" in text or "材质" in text:
            # 清理文本，提取有用部分
            clean_text = re.sub(r'[{}\[\]":]', ' ', text)
            clean_text = ' '.join(clean_text.split())
            if len(clean_text) > 10:
                return clean_text[:200]

        return None

    def _validate_response_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """验证和修复响应数据"""

        if not isinstance(data, dict):
            return None

        # 提取描述
        desc = data.get("detailed_description", "").strip()

        # 如果描述为空，尝试其他字段
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

        # 如果还是没有描述，返回None
        if not desc or len(desc) < 5:
            logger.warning(f"描述内容太短或为空: '{desc}'")
            return None

        # 构建完整的响应
        result = {
            "detailed_description": desc,
            "entity_info": data.get("entity_info", {})
        }

        # 确保entity_info完整
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
            current_item_hashes = set()
            for img_key, local_path in img_path_dict.items():
                if not local_path or not isinstance(local_path, str):
                    continue

                # 检查是否已处理（全局级别）
                skip_reason = None

                # 尝试通过本地路径反查映射信息
                content_hash = None
                if image_manager:
                    # 遍历所有映射找到对应的hash
                    for orig_url, mapping in image_manager.mappings.items():
                        if mapping.local_path == local_path:
                            content_hash = mapping.content_hash
                            break

                if content_hash:
                    # 检查全局是否已处理
                    if content_hash in self.processed_content_hashes:
                        skip_reason = f"全局已处理 (hash: {content_hash[:8]})"
                    # 检查当前商品内是否已处理
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
                        # logger.info(f"图片处理结果: {result}")
                        processed_count += 1
                        logger.info(f"图片处理成功: {img_key}")

                except Exception as e:
                    logger.error(f"处理失败 {img_key}: {e}")
                    continue
            logger.info(
                f"商品处理完成: {entity_name}, 处理 {processed_count} 张，跳过 {skipped_count} 张（共 {total_images} 张）")
            return True

        except Exception as e:
            logger.error(f"多模态处理失败: {e}")
            await self._fallback_text_processing(modal_content, entity_name)
            return False

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
            result = await self.lightrag_instance.aquery(
                query,  # 直接使用用户查询
                param=QueryParam(mode=mode)
            )

            if result:
                # 确保图片URL是远程访问格式
                processed_result = post_process_response_urls(result)
                logger.info(f"查询完成，结果长度: {len(processed_result)}")
                return processed_result

        except Exception as e:
            logger.error(f"查询失败: {e}")

        return f"抱歉，暂无与「{query}」相关的产品推荐信息。请稍后再试或换个关键词。"
