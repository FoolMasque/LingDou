"""
简化的增强版启动程序 - enhanced_simple.py
确保可以正常运行的版本
"""

import asyncio
import json
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import time

# 导入基础组件
try:
    from main import SimpleImageManager, SimpleMultiModalProcessor, BusinessConfig, MockRAGInstance

    BASE_AVAILABLE = True
except ImportError:
    print("请确保 main.py 在同一目录下")
    BASE_AVAILABLE = False

# 检查RAG-Anything可用性
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc

    RAG_AVAILABLE = True
    print("RAG-Anything 可用")
except ImportError:
    RAG_AVAILABLE = False
    print("RAG-Anything 未安装，将使用模拟版本")


class SimpleRealRAG:
    """简化的真实RAG实例"""

    def __init__(self, business_id: str, api_key: str, provider: str = "zhipu"):
        self.business_id = business_id
        self.api_key = api_key
        self.provider = provider

        if not RAG_AVAILABLE:
            raise ImportError("RAG-Anything未安装")

        # 创建RAG配置
        self.rag_config = RAGAnythingConfig(
            working_dir=f"./rag_storage_{business_id}",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 初始化RAG实例
        try:
            self.rag_instance = RAGAnything(
                config=self.rag_config,
                llm_model_func=self._get_llm_func(),
                vision_model_func=self._get_vision_func(),
                embedding_func=self._get_embedding_func()
            )
            print(f"真实RAG实例创建成功: {business_id}")
        except Exception as e:
            print(f"RAG实例创建失败: {e}")
            raise

    def _get_llm_func(self):
        """获取LLM函数"""
        if self.provider == "zhipu":
            from lightrag.llm.zhipu import zhipu_complete_if_cache
            def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return zhipu_complete_if_cache(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    model="glm-4-flash",
                    api_key=self.api_key,
                    **kwargs
                )

            return llm_func
        else:
            # OpenAI兼容
            from lightrag.llm.openai import openai_complete_if_cache
            def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    model="gpt-4o-mini",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    **kwargs
                )

            return llm_func

    def _get_vision_func(self):
        """获取视觉模型函数"""
        if self.provider == "zhipu":
            from lightrag.llm.zhipu import zhipu_complete_if_cache
            def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
                if image_data:
                    return zhipu_complete_if_cache(
                        prompt=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ],
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        model="glm-4v-flash",
                        api_key=self.api_key,
                        **kwargs
                    )
                else:
                    return zhipu_complete_if_cache(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        model="glm-4v-flash",
                        api_key=self.api_key,
                        **kwargs
                    )

            return vision_func
        else:
            # OpenAI兼容的简化版本
            from lightrag.llm.openai import openai_complete_if_cache
            def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
                return openai_complete_if_cache(
                    model="gpt-4o",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    **kwargs
                )

            return vision_func

    def _get_embedding_func(self):
        """获取embedding函数"""
        if self.provider == "zhipu":
            from lightrag.llm.zhipu import zhipu_embedding
            return EmbeddingFunc(
                embedding_dim=2048,
                max_token_size=8192,
                func=lambda texts: zhipu_embedding(
                    texts,
                    model="embedding-3",
                    api_key=self.api_key
                )
            )
        else:
            from lightrag.llm.openai import openai_embed
            return EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=self.api_key
                )
            )

    async def process_multimodal_content(self, modal_content: Dict[str, Any],
                                         entity_name: str, file_path: str):
        """处理多模态内容 - 使用正确的RAG-Anything工作流"""
        print(f"使用真实RAG处理: {entity_name}")

        try:
            # RAG-Anything主要用于查询，不是用于插入数据
            # 我们需要使用LightRAG的底层功能来插入数据

            # 获取LightRAG实例（RAG-Anything的底层）
            lightrag_instance = None

            # 尝试不同方式获取LightRAG实例
            if hasattr(self.rag_instance, 'lightrag'):
                lightrag_instance = self.rag_instance.lightrag
            elif hasattr(self.rag_instance, 'rag'):
                lightrag_instance = self.rag_instance.rag
            elif hasattr(self.rag_instance, '_lightrag'):
                lightrag_instance = self.rag_instance._lightrag

            if lightrag_instance and hasattr(lightrag_instance, 'ainsert'):
                # 使用LightRAG插入数据
                text_content = self._convert_modal_to_text(modal_content, entity_name)
                await lightrag_instance.ainsert(text_content)
                print(f"通过LightRAG处理完成: {entity_name}")
            else:
                # 如果无法找到LightRAG实例，直接创建一个简单的数据存储
                # 这样至少查询时能找到数据
                if not hasattr(self, 'manual_data_store'):
                    self.manual_data_store = []

                text_content = self._convert_modal_to_text(modal_content, entity_name)
                self.manual_data_store.append({
                    'entity_name': entity_name,
                    'content': text_content,
                    'modal_content': modal_content
                })
                print(f"存储到手动数据库: {entity_name}")

        except Exception as e:
            print(f"真实RAG处理失败 {entity_name}: {e}")
            print(f"错误类型: {type(e).__name__}")

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """查询 - 使用真实RAG或手动存储"""
        print(f"真实RAG查询: {query} (模式: {mode})")

        try:
            # 首先尝试使用RAG-Anything查询
            result = await self.rag_instance.aquery(query, mode=mode)
            print(f"RAG-Anything查询成功，结果长度: {len(result)}")
            return result

        except Exception as e:
            print(f"RAG-Anything查询失败，尝试手动查询: {e}")

            # 降级到手动查询
            if hasattr(self, 'manual_data_store'):
                return self._manual_query(query)
            else:
                return f"查询失败: {str(e)}"

    def _manual_query(self, query: str) -> str:
        """手动查询存储的数据"""
        if not hasattr(self, 'manual_data_store') or not self.manual_data_store:
            return "没有找到相关数据"

        query_lower = query.lower()
        matched_items = []

        # 简单的关键词匹配
        for item in self.manual_data_store:
            content_lower = item['content'].lower()
            entity_lower = item['entity_name'].lower()

            score = 0
            for word in query_lower.split():
                if len(word) > 1:
                    if word in content_lower:
                        score += 1
                    if word in entity_lower:
                        score += 2

            if score > 0:
                matched_items.append((item, score))

        if not matched_items:
            return f"抱歉，没有找到与「{query}」相关的信息。"

        # 按分数排序并生成回答
        matched_items.sort(key=lambda x: x[1], reverse=True)

        response_parts = [f"根据您的查询「{query}」，我为您推荐以下产品：\n"]

        for i, (item, score) in enumerate(matched_items[:3], 1):
            response_parts.append(f"{i}. **{item['entity_name']}**")

            # 提取关键信息
            content_lines = item['content'].split('\n')
            for line in content_lines:
                if ':' in line and not line.startswith('商品名称'):
                    response_parts.append(f"   • {line}")

            # 添加图片信息
            modal_content = item.get('modal_content', {})
            img_path = modal_content.get('img_path', {})
            if img_path.get('cover_pic'):
                response_parts.append(f"   • 查看图片: {img_path['cover_pic']}")

            response_parts.append("")

        return "\n".join(response_parts)

    def _convert_modal_to_text(self, modal_content: Dict[str, Any], entity_name: str) -> str:
        """将多模态内容转换为文本格式"""
        text_parts = [f"商品名称: {entity_name}"]

        # 处理caption信息
        captions = modal_content.get("img_caption", [])
        if captions:
            for caption in captions:
                lines = caption.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- '):
                        # 提取结构化信息
                        field_info = line[2:]  # 移除'- '
                        if ':' in field_info:
                            text_parts.append(field_info)
                    elif line and not any(
                            line.startswith(prefix) for prefix in ['请', '以下', '重点', '1.', '2.', '3.', '4.']):
                        if len(line) > 3:  # 过滤太短的行
                            text_parts.append(line)

        # 添加图片信息
        img_path = modal_content.get("img_path", {})
        if img_path:
            if img_path.get("cover_pic"):
                text_parts.append(f"封面图片: {img_path['cover_pic']}")
            if img_path.get("detail_images") and len(img_path["detail_images"]) > 0:
                text_parts.append(f"详情图片: {len(img_path['detail_images'])}张")

        return "\n".join(text_parts)

    def _prepare_modal_content(self, modal_content: Dict[str, Any]) -> Dict[str, Any]:
        """准备多模态内容，转换URL为本地路径"""
        processed = modal_content.copy()

        img_path = processed.get("img_path", {})
        if isinstance(img_path, dict):
            # 转换HTTP URL为本地文件路径
            if img_path.get("cover_pic") and img_path["cover_pic"].startswith("http://localhost:8000"):
                img_path["cover_pic"] = img_path["cover_pic"].replace("http://localhost:8000/", "./")

            if img_path.get("detail_images"):
                converted_details = []
                for img_url in img_path["detail_images"]:
                    if img_url.startswith("http://localhost:8000"):
                        converted_details.append(img_url.replace("http://localhost:8000/", "./"))
                    else:
                        converted_details.append(img_url)
                img_path["detail_images"] = converted_details

        return processed

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """查询"""
        print(f"真实RAG查询: {query} (模式: {mode})")

        try:
            result = await self.rag_instance.aquery(query, mode=mode)
            print(f"真实RAG查询完成，结果长度: {len(result)}")
            return result
        except Exception as e:
            print(f"真实RAG查询失败: {e}")
            return f"查询出现错误: {str(e)}"

class SimplifiedEnhancedSystem:
    """简化的增强系统"""

    def __init__(self):
        self.businesses: Dict[str, BusinessConfig] = {}
        self.rag_instances: Dict[str, Any] = {}
        self.processors: Dict[str, SimpleMultiModalProcessor] = {}
        self.image_manager = SimpleImageManager()

        # 从环境变量或配置文件加载API配置
        self.api_key = self._get_api_key()
        self.provider = os.getenv("LLM_PROVIDER", "zhipu")

        print(f"API配置: 提供商={self.provider}, 密钥={'已配置' if self.api_key else '未配置'}")

    def _get_api_key(self) -> str:
        """获取API密钥"""
        # 优先从环境变量
        api_key = os.getenv("API_KEY")
        if api_key:
            return api_key

        # 从配置文件读取
        config_file = Path("config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("api", {}).get("api_key", "")
            except Exception as e:
                print(f"读取配置文件失败: {e}")

        return ""

    def register_business(self, config: BusinessConfig):
        """注册业务"""
        self.businesses[config.business_id] = config
        self.processors[config.business_id] = SimpleMultiModalProcessor(config.business_id)

        # 尝试创建真实RAG实例
        if RAG_AVAILABLE and self.api_key:
            try:
                self.rag_instances[config.business_id] = SimpleRealRAG(
                    config.business_id,
                    self.api_key,
                    self.provider
                )
                print(f"业务 {config.name} 注册成功 - 使用真实RAG")
                return
            except Exception as e:
                print(f"真实RAG创建失败，降级使用模拟RAG: {e}")

        # 降级使用模拟RAG
        self.rag_instances[config.business_id] = MockRAGInstance(config.business_id)
        print(f"业务 {config.name} 注册成功 - 使用模拟RAG")

    async def process_crawler_data(self, business_id: str, json_file: str):
        """处理爬虫数据"""
        if business_id not in self.businesses:
            raise ValueError(f"未注册的业务: {business_id}")

        # 读取数据
        async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        print(f"开始处理 {len(data)} 条数据")

        processor = self.processors[business_id]
        rag = self.rag_instances[business_id]

        for i, item in enumerate(data):
            try:
                # 下载图片
                image_urls = self._extract_image_urls(item)
                if image_urls:
                    url_mapping = await self.image_manager.download_images(image_urls, business_id)
                    self._update_item_urls(item, url_mapping)

                # 构建多模态内容
                modal_content = processor.build_modal_content(item)

                # 处理到RAG系统
                entity_name = item.get("商品名", f"Item_{i}")
                await rag.process_multimodal_content(
                    modal_content=modal_content,
                    entity_name=entity_name,
                    file_path=f"{business_id}_{i}.json"
                )

                print(f"处理完成 {i + 1}/{len(data)}: {entity_name}")

            except Exception as e:
                print(f"处理数据项失败 {i}: {e}")

        print("数据处理完成")

    def _extract_image_urls(self, item: Dict[str, Any]) -> List[str]:
        """提取图片URL"""
        urls = []
        if item.get("cover_pic"):
            urls.append(item["cover_pic"])
        if item.get("detail_images"):
            if isinstance(item["detail_images"], list):
                urls.extend(item["detail_images"])
        return urls

    def _update_item_urls(self, item: Dict[str, Any], url_mapping: Dict[str, str]):
        """更新item中的URL"""
        if item.get("cover_pic") and item["cover_pic"] in url_mapping:
            item["cover_pic"] = url_mapping[item["cover_pic"]]

        if item.get("detail_images") and isinstance(item["detail_images"], list):
            item["detail_images"] = [
                url_mapping.get(url, url) for url in item["detail_images"]
            ]

    async def query(self, business_id: str, query: str, mode: str = "hybrid") -> str:
        """查询"""
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]
        return await rag.aquery(query, mode)

    def get_status(self, business_id: str) -> Dict[str, Any]:
        """获取状态"""
        if business_id not in self.businesses:
            return {"error": "业务不存在"}

        rag_type = "真实RAG" if isinstance(self.rag_instances[business_id], SimpleRealRAG) else "模拟RAG"

        return {
            "business_id": business_id,
            "name": self.businesses[business_id].name,
            "rag_type": rag_type,
            "api_configured": bool(self.api_key),
            "status": "活跃"
        }


class FixedRealRAGInstance:
    """修复后的真实RAG实例"""

    def __init__(self, business_id: str, api_config: Dict[str, Any]):
        self.business_id = business_id
        self.api_config = api_config
        self.working_dir = f"./rag_storage_{business_id}"

        if not RAG_AVAILABLE:
            raise ImportError("RAG-Anything未安装")

        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)

        # 初始化核心LightRAG实例
        self.lightrag_instance = self._create_lightrag_instance()

        # 创建RAG配置
        self.rag_config = RAGAnythingConfig(
            working_dir=self.working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 初始化RAG-Anything实例，传入预创建的LightRAG实例
        try:
            self.rag_instance = RAGAnything(
                config=self.rag_config,
                lightrag_instance=self.lightrag_instance  # 关键：传入预初始化的实例
            )
            print(f"RAG-Anything实例创建成功: {business_id}")
        except Exception as e:
            print(f"RAG-Anything创建失败，尝试其他方式: {e}")
            # 备用方案：直接使用function方式
            self.rag_instance = RAGAnything(
                config=self.rag_config,
                llm_model_func=self._get_llm_func(),
                vision_model_func=self._get_vision_func(),
                embedding_func=self._get_embedding_func()
            )

    def _create_lightrag_instance(self) -> LightRAG:
        """创建核心LightRAG实例"""
        print(f"创建LightRAG实例: {self.working_dir}")

        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._get_llm_func(),
            embedding_func=self._get_embedding_func()
        )

    def _get_llm_func(self):
        """获取LLM函数"""
        api_key = self.api_config.get("api_key")
        provider = self.api_config.get("provider", "openai")
        model = self.api_config.get("llm_model", "gpt-4o-mini")

        if provider == "zhipu":
            from lightrag.llm.zhipu import zhipu_complete_if_cache
            def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return zhipu_complete_if_cache(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    model=model,
                    api_key=api_key,
                    **kwargs
                )

            return llm_func
        else:
            # OpenAI兼容
            from lightrag.llm.openai import openai_complete_if_cache
            def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    **kwargs
                )

            return llm_func

    def _get_vision_func(self):
        """获取视觉模型函数"""
        api_key = self.api_config.get("api_key")
        provider = self.api_config.get("provider", "openai")
        vision_model = self.api_config.get("vision_model", "gpt-4o")

        if provider == "zhipu":
            from lightrag.llm.zhipu import zhipu_complete_if_cache
            def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
                if image_data:
                    # 多模态请求
                    content = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                    return zhipu_complete_if_cache(
                        prompt=content,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        model=vision_model,
                        api_key=api_key,
                        **kwargs
                    )
                else:
                    # 纯文本请求
                    return zhipu_complete_if_cache(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        model=vision_model,
                        api_key=api_key,
                        **kwargs
                    )

            return vision_func
        else:
            # OpenAI兼容
            from lightrag.llm.openai import openai_complete_if_cache
            def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
                if image_data:
                    # 构建消息格式
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})

                    messages.extend(history_messages)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    })

                    return openai_complete_if_cache(
                        model=vision_model,
                        prompt="",  # 清空prompt，使用messages
                        system_prompt=None,  # 清空system_prompt，已在messages中
                        history_messages=[],  # 清空history_messages，已在messages中
                        messages=messages,
                        api_key=api_key,
                        **kwargs
                    )
                else:
                    return openai_complete_if_cache(
                        model=vision_model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=api_key,
                        **kwargs
                    )

            return vision_func

    def _get_embedding_func(self):
        """获取embedding函数"""
        api_key = self.api_config.get("api_key")
        provider = self.api_config.get("provider", "openai")
        embedding_model = self.api_config.get("embedding_model", "text-embedding-3-large")
        embedding_dim = self.api_config.get("embedding_dim", 3072)

        if provider == "zhipu":
            from lightrag.llm.zhipu import zhipu_embedding
            return EmbeddingFunc(
                embedding_dim=2048,  # 智谱embedding维度
                max_token_size=8192,
                func=lambda texts: zhipu_embedding(
                    texts,
                    model="embedding-3",
                    api_key=api_key
                )
            )
        else:
            from lightrag.llm.openai import openai_embed
            return EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embedding_model,
                    api_key=api_key
                )
            )

    async def process_multimodal_content(self, modal_content: Dict[str, Any],
                                         entity_name: str, file_path: str):
        """处理多模态内容 - 修复版本"""
        print(f"处理多模态内容: {entity_name}")

        try:
            # 方法1：使用RAG-Anything的标准工作流
            # 首先尝试直接使用RAG-Anything处理
            processed_modal_content = self._prepare_modal_content_for_rag(modal_content)

            # 检查是否有图片需要处理
            img_paths = processed_modal_content.get("img_path", {})
            has_images = bool(img_paths.get("cover_pic") or img_paths.get("detail_images"))

            if has_images:
                # 有图片：使用多模态处理
                await self.rag_instance.process_multimodal_content(
                    modal_content=processed_modal_content,
                    content_type="image",
                    file_path=file_path,
                    entity_name=entity_name
                )
            else:
                # 无图片：处理纯文本内容
                text_content = self._convert_modal_to_text(modal_content, entity_name)
                await self._process_text_content(text_content)

            print(f"多模态内容处理完成: {entity_name}")

        except Exception as e:
            print(f"RAG-Anything处理失败，尝试备用方案: {e}")
            # 备用方案：直接使用LightRAG处理
            await self._fallback_process(modal_content, entity_name)

    def _prepare_modal_content_for_rag(self, modal_content: Dict[str, Any]) -> Dict[str, Any]:
        """为RAG-Anything准备模态内容"""
        processed = modal_content.copy()

        # 处理图片路径
        img_path = processed.get("img_path", {})
        if isinstance(img_path, dict):
            # 转换HTTP URL为本地路径
            if img_path.get("cover_pic"):
                cover_url = img_path["cover_pic"]
                if cover_url.startswith("http://localhost:8000/images/"):
                    local_path = cover_url.replace("http://localhost:8000/", "./")
                    img_path["cover_pic"] = local_path

            if img_path.get("detail_images"):
                processed_details = []
                for img_url in img_path["detail_images"]:
                    if img_url.startswith("http://localhost:8000/images/"):
                        local_path = img_url.replace("http://localhost:8000/", "./")
                        processed_details.append(local_path)
                    else:
                        processed_details.append(img_url)
                img_path["detail_images"] = processed_details

        return processed

    async def _process_text_content(self, text_content: str):
        """处理纯文本内容"""
        if hasattr(self.lightrag_instance, 'ainsert'):
            await self.lightrag_instance.ainsert(text_content)
        elif hasattr(self.rag_instance, 'lightrag') and hasattr(self.rag_instance.lightrag, 'ainsert'):
            await self.rag_instance.lightrag.ainsert(text_content)
        else:
            print("无法找到文本插入方法，使用手动存储")
            # 手动存储到类属性中作为备用
            if not hasattr(self, 'manual_text_store'):
                self.manual_text_store = []
            self.manual_text_store.append(text_content)

    async def _fallback_process(self, modal_content: Dict[str, Any], entity_name: str):
        """备用处理方案"""
        print(f"使用备用方案处理: {entity_name}")
        text_content = self._convert_modal_to_text(modal_content, entity_name)
        await self._process_text_content(text_content)

    def _convert_modal_to_text(self, modal_content: Dict[str, Any], entity_name: str) -> str:
        """将多模态内容转换为文本"""
        text_parts = [f"商品名称: {entity_name}"]

        # 处理图片标题
        captions = modal_content.get("img_caption", [])
        if captions:
            for caption in captions:
                lines = caption.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- '):
                        # 提取结构化信息
                        field_info = line[2:]  # 移除'- '
                        if ':' in field_info:
                            text_parts.append(field_info)
                    elif line and len(line) > 3:
                        # 过滤掉提示性文字
                        skip_prefixes = ['请', '以下', '重点', '1.', '2.', '3.', '4.']
                        if not any(line.startswith(prefix) for prefix in skip_prefixes):
                            text_parts.append(line)

        # 添加图片信息
        img_path = modal_content.get("img_path", {})
        if img_path:
            if img_path.get("cover_pic"):
                text_parts.append(f"封面图片: {img_path['cover_pic']}")
            if img_path.get("detail_images") and len(img_path["detail_images"]) > 0:
                text_parts.append(f"详情图片: {len(img_path['detail_images'])}张")

        return "\n".join(text_parts)

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """查询 - 修复版本"""
        print(f"执行查询: {query} (模式: {mode})")

        try:
            # 方法1：尝试使用RAG-Anything查询
            if hasattr(self.rag_instance, 'aquery'):
                result = await self.rag_instance.aquery(query, mode=mode)
                if result and len(result.strip()) > 0:
                    print(f"RAG-Anything查询成功，结果长度: {len(result)}")
                    return result

        except Exception as e:
            print(f"RAG-Anything查询失败: {e}")

        try:
            # 方法2：尝试使用LightRAG查询
            if hasattr(self.lightrag_instance, 'aquery'):
                # 尝试不同的查询模式
                query_modes = ["hybrid", "local", "global"]
                if mode in query_modes:
                    query_modes = [mode] + [m for m in query_modes if m != mode]

                for qmode in query_modes:
                    try:
                        result = await self.lightrag_instance.aquery(
                            query,
                            param=QueryParam(mode=qmode)
                        )
                        if result and len(result.strip()) > 0:
                            print(f"LightRAG查询成功 (模式: {qmode})，结果长度: {len(result)}")
                            return result
                    except Exception as mode_error:
                        print(f"LightRAG查询模式 {qmode} 失败: {mode_error}")
                        continue

        except Exception as e:
            print(f"LightRAG查询失败: {e}")

        # 方法3：手动查询备用数据
        if hasattr(self, 'manual_text_store') and self.manual_text_store:
            return self._manual_query(query, self.manual_text_store)

        return f"抱歉，暂无与「{query}」相关的信息。请先处理一些数据。"

    def _manual_query(self, query: str, text_store: List[str]) -> str:
        """手动查询存储的文本数据"""
        query_lower = query.lower()
        matches = []

        for i, text in enumerate(text_store):
            text_lower = text.lower()
            score = 0

            # 简单的关键词匹配
            for word in query_lower.split():
                if len(word) > 1 and word in text_lower:
                    score += text_lower.count(word)

            if score > 0:
                matches.append((text, score, i))

        if not matches:
            return f"在已处理的数据中未找到与「{query}」相关的信息。"

        # 按分数排序
        matches.sort(key=lambda x: x[1], reverse=True)

        # 构建回答
        response_parts = [f"根据您的查询「{query}」，找到以下相关信息：\n"]

        for i, (text, score, idx) in enumerate(matches[:3], 1):
            lines = text.split('\n')
            if len(lines) > 0:
                title_line = lines[0]
                response_parts.append(f"{i}. {title_line}")

                # 添加相关信息行
                for line in lines[1:6]:  # 最多显示5行
                    if line.strip() and ':' in line:
                        response_parts.append(f"   • {line}")

                response_parts.append("")

        return "\n".join(response_parts)


class EnhancedCoreSystem:
    """增强版核心系统 - 使用修复后的RAG集成"""

    def __init__(self):
        self.businesses: Dict[str, Any] = {}
        self.rag_instances: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}

        # 动态导入
        try:
            from main import SimpleImageManager, SimpleMultiModalProcessor, MockRAGInstance
            self.image_manager = SimpleImageManager()
            self.MockRAGInstance = MockRAGInstance
            self.SimpleMultiModalProcessor = SimpleMultiModalProcessor
        except ImportError:
            print("警告: 无法导入main模块的组件，请检查main.py是否存在")

        # API配置
        self.api_config = self._load_api_config()
        print(f"API配置加载完成: {self.api_config}")

    def _load_api_config(self) -> Dict[str, Any]:
        """加载API配置"""
        # 优先从环境变量读取
        api_config = {
            "provider": os.getenv("LLM_PROVIDER", "zhipu"),
            "api_key": os.getenv("API_KEY", ""),
            "base_url": os.getenv("BASE_URL"),
            "llm_model": os.getenv("LLM_MODEL", "glm-4-flash"),
            "vision_model": os.getenv("VISION_MODEL", "glm-4v-flash"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "embedding-3"),
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", "2048"))
        }

        # 如果没有设置API_KEY，尝试从配置文件读取
        if not api_config["api_key"]:
            config_file = Path("config.json")
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                        api_config.update(file_config.get("api", {}))
                except Exception as e:
                    print(f"读取配置文件失败: {e}")

        return api_config

    def register_business(self, config):
        """注册业务 - 使用修复后的RAG"""
        self.businesses[config.business_id] = config

        # 尝试创建真实RAG实例
        if RAG_AVAILABLE and self.api_config.get("api_key"):
            try:
                self.rag_instances[config.business_id] = FixedRealRAGInstance(
                    config.business_id,
                    self.api_config
                )
                print(f"✅ 业务 {config.name} 注册成功 - 使用真实RAG-Anything")
            except Exception as e:
                print(f"❌ 创建真实RAG实例失败: {e}")
                # 降级到模拟版本
                if self.MockRAGInstance:
                    self.rag_instances[config.business_id] = self.MockRAGInstance(config.business_id)
                    print(f"⚠️ 业务 {config.name} 注册成功 - 降级使用模拟RAG")
                else:
                    print("❌ 无法创建任何RAG实例")
                    return
        else:
            # 使用模拟版本
            if self.MockRAGInstance:
                self.rag_instances[config.business_id] = self.MockRAGInstance(config.business_id)
                print(f"⚠️ 业务 {config.name} 注册成功 - 使用模拟RAG (API未配置)")
            else:
                print("❌ 无法创建任何RAG实例")
                return

        # 创建处理器
        if self.SimpleMultiModalProcessor:
            self.processors[config.business_id] = self.SimpleMultiModalProcessor(config.business_id)

    async def process_crawler_data(self, business_id: str, json_file: str):
        """处理爬虫数据"""
        if business_id not in self.businesses:
            raise ValueError(f"未注册的业务: {business_id}")

        # 读取JSON数据
        async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        print(f"开始处理 {len(data)} 条数据")

        # 获取处理器和RAG实例
        processor = self.processors[business_id]
        rag = self.rag_instances[business_id]

        success_count = 0
        # 处理每个数据项
        for i, item in enumerate(data):
            try:
                # 下载图片
                if hasattr(self, 'image_manager'):
                    image_urls = self._extract_image_urls(item)
                    if image_urls:
                        url_mapping = await self.image_manager.download_images(image_urls, business_id)
                        self._update_item_urls(item, url_mapping)

                # 构建多模态内容
                modal_content = processor.build_modal_content(item)

                # 处理到RAG系统
                entity_name = item.get("商品名", f"Item_{i}")
                await rag.process_multimodal_content(
                    modal_content=modal_content,
                    entity_name=entity_name,
                    file_path=f"{business_id}_{i}.json"
                )

                success_count += 1
                print(f"✅ 处理完成 {success_count}/{len(data)}: {entity_name}")

            except Exception as e:
                print(f"❌ 处理数据项失败 {i}: {e}")

        print(f"✅ 数据处理完成，成功处理 {success_count}/{len(data)} 条数据")

    def _extract_image_urls(self, item: Dict[str, Any]) -> List[str]:
        """提取图片URL"""
        urls = []
        if item.get("cover_pic"):
            urls.append(item["cover_pic"])
        if item.get("detail_images"):
            if isinstance(item["detail_images"], list):
                urls.extend(item["detail_images"])
        return urls

    def _update_item_urls(self, item: Dict[str, Any], url_mapping: Dict[str, str]):
        """更新item中的URL"""
        if item.get("cover_pic") and item["cover_pic"] in url_mapping:
            item["cover_pic"] = url_mapping[item["cover_pic"]]

        if item.get("detail_images") and isinstance(item["detail_images"], list):
            item["detail_images"] = [
                url_mapping.get(url, url) for url in item["detail_images"]
            ]

    async def query(self, business_id: str, query: str, mode: str = "hybrid") -> str:
        """查询"""
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]
        return await rag.aquery(query, mode)

    def get_business_status(self, business_id: str) -> Dict[str, Any]:
        """获取业务状态"""
        if business_id not in self.businesses:
            return {"error": "业务不存在"}

        rag_instance = self.rag_instances[business_id]
        rag_type = "FixedRealRAG" if isinstance(rag_instance, FixedRealRAGInstance) else "MockRAG"

        # 检查RAG实例的存储目录
        storage_status = "未知"
        if isinstance(rag_instance, FixedRealRAGInstance):
            if os.path.exists(rag_instance.working_dir) and os.listdir(rag_instance.working_dir):
                storage_status = "有数据"
            else:
                storage_status = "空"

        return {
            "business_id": business_id,
            "name": self.businesses[business_id].name,
            "rag_type": rag_type,
            "api_configured": bool(self.api_config.get("api_key")),
            "storage_status": storage_status,
            "status": "活跃"
        }


# 简化的API接口
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 全局系统实例
enhanced_system = SimplifiedEnhancedSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    print("=== 简化增强版RAG系统启动 ===")

    # 注册默认业务
    furniture_config = BusinessConfig(
        business_id="furniture",
        name="侘界家具",
        image_fields=["cover_pic", "detail_images"],
        text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
    )
    enhanced_system.register_business(furniture_config)

    print("=== 系统启动完成 ===")
    yield
    print("系统关闭")


app = FastAPI(
    title="简化增强版RAG系统",
    version="1.0.0",
    lifespan=lifespan
)

# 中间件
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 静态文件
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")


# API模型
class ProcessRequest(BaseModel):
    business_id: str
    json_file: str


class QueryRequest(BaseModel):
    business_id: str
    query: str
    mode: str = "hybrid"


# API路由
@app.post("/api/process_data")
async def process_data(request: ProcessRequest, background_tasks: BackgroundTasks):
    """处理数据"""
    try:
        if not Path(request.json_file).exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {request.json_file}")

        background_tasks.add_task(
            enhanced_system.process_crawler_data,
            request.business_id,
            request.json_file
        )
        return {"success": True, "message": "数据处理任务已启动"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(request: QueryRequest):
    """查询"""
    try:
        start_time = time.time()
        result = await enhanced_system.query(request.business_id, request.query, request.mode)
        processing_time = time.time() - start_time

        return {
            "success": True,
            "query": request.query,
            "result": result,
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{business_id}")
async def get_status(business_id: str):
    """获取状态"""
    return enhanced_system.get_status(business_id)


@app.get("/api/system_info")
async def system_info():
    """系统信息"""
    return {
        "rag_available": RAG_AVAILABLE,
        "api_configured": bool(enhanced_system.api_key),
        "provider": enhanced_system.provider,
        "businesses": list(enhanced_system.businesses.keys())
    }


@app.post("/dify/knowledge/query")
async def dify_query(business_id: str, query: str):
    """Dify接口"""
    try:
        result = await enhanced_system.query(business_id, query)
        return {
            "records": [{
                "content": result,
                "score": 0.9,
                "title": f"{business_id}智能推荐",
                "metadata": {"business_id": business_id}
            }]
        }
    except Exception as e:
        return {"error": str(e), "records": []}


if __name__ == "__main__":
    import uvicorn

    print("""
=== 简化增强版RAG系统 ===

配置方法:
1. 环境变量: 
   export API_KEY="your-zhipu-api-key"
   export LLM_PROVIDER="zhipu"

2. 配置文件: 创建 config.json
   {"api": {"api_key": "your-key", "provider": "zhipu"}}

API文档: http://localhost:8000/docs
系统信息: http://localhost:8000/api/system_info
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)