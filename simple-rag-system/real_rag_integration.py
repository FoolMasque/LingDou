"""
集成真实RAG-Anything引擎
替换MockRAGInstance为真正的RAG-Anything系统
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64
from main import SimpleImageManager,SimpleMultiModalProcessor
# 真实的RAG-Anything导入
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    RAG_AVAILABLE = True
except ImportError:
    print("RAG-Anything未安装，请运行: pip install rag-anything==1.2.7")
    RAG_AVAILABLE = False


class RealRAGInstance:
    """真实的RAG实例"""

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

    def _get_llm_func(self):
        """获取LLM函数"""
        api_key = self.api_config.get("api_key")
        base_url = self.api_config.get("base_url", "https://api.openai.com/v1")
        model = self.api_config.get("llm_model", "gpt-4o-mini")

        if self.api_config.get("provider") == "zhipu":
            from lightrag.llm.zhipu import zhipu_complete_if_cache
            return lambda prompt, system_prompt=None, history_messages=[], **kwargs: zhipu_complete_if_cache(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                model=model,
                api_key=api_key,
                **kwargs
            )
        else:
            # 默认使用OpenAI兼容接口
            from lightrag.llm.openai import openai_complete_if_cache
            return lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )

    def _get_vision_func(self):
        """获取视觉模型函数"""
        api_key = self.api_config.get("api_key")
        base_url = self.api_config.get("base_url", "https://api.openai.com/v1")
        vision_model = self.api_config.get("vision_model", "gpt-4o")

        if self.api_config.get("provider") == "zhipu":
            from lightrag.llm.zhipu import zhipu_complete_if_cache
            return lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: (
                zhipu_complete_if_cache(
                    prompt=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ] if image_data else prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    model=vision_model,
                    api_key=api_key,
                    **kwargs
                ) if image_data else zhipu_complete_if_cache(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    model=vision_model,
                    api_key=api_key,
                    **kwargs
                )
            )
        else:
            from lightrag.llm.openai import openai_complete_if_cache
            return lambda prompt, system_prompt=None, history_messages=[], image_data=None,
                          **kwargs: openai_complete_if_cache(
                model=vision_model,
                prompt="",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            }
                        ] if image_data else prompt
                    }
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs
            ) if image_data else openai_complete_if_cache(
                model=vision_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )

    def _get_embedding_func(self):
        """获取embedding函数"""
        api_key = self.api_config.get("api_key")
        base_url = self.api_config.get("base_url", "https://api.openai.com/v1")
        embedding_model = self.api_config.get("embedding_model", "text-embedding-3-large")
        embedding_dim = self.api_config.get("embedding_dim", 3072)

        if self.api_config.get("provider") == "zhipu":
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
                    api_key=api_key,
                    base_url=base_url
                )
            )

    async def process_multimodal_content(self, modal_content: Dict[str, Any],
                                         entity_name: str, file_path: str):
        """处理多模态内容"""
        print(f"使用RAG-Anything处理: {entity_name}")

        # 处理图片路径，转换为本地路径供RAG-Anything使用
        img_path = modal_content.get("img_path", {})
        processed_img_path = self._prepare_image_paths(img_path)

        # 更新modal_content
        processed_modal_content = {
            **modal_content,
            "img_path": processed_img_path
        }

        # 使用RAG-Anything的process_multimodal_content
        await self.rag_instance.process_multimodal_content(
            modal_content=processed_modal_content,
            content_type="image",
            file_path=file_path,
            entity_name=entity_name
        )

    def _prepare_image_paths(self, img_path: Dict[str, Any]) -> Dict[str, Any]:
        """准备图片路径，转换为本地路径"""
        processed = {}

        # 处理封面图
        if img_path.get("cover_pic"):
            cover_url = img_path["cover_pic"]
            if cover_url.startswith("http://localhost:8000/images/"):
                # 转换为本地路径
                local_path = cover_url.replace("http://localhost:8000/", "./")
                processed["cover_pic"] = local_path
            else:
                processed["cover_pic"] = cover_url

        # 处理详情图
        if img_path.get("detail_images"):
            detail_images = img_path["detail_images"]
            processed_details = []
            for img_url in detail_images:
                if img_url.startswith("http://localhost:8000/images/"):
                    local_path = img_url.replace("http://localhost:8000/", "./")
                    processed_details.append(local_path)
                else:
                    processed_details.append(img_url)
            processed["detail_images"] = processed_details

        return processed

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """使用RAG-Anything进行查询"""
        print(f"使用RAG-Anything查询: {query} (模式: {mode})")

        try:
            # 使用RAG-Anything的VLM-Enhanced Query
            result = await self.rag_instance.aquery(query, mode=mode)
            return result
        except Exception as e:
            print(f"RAG查询失败: {e}")
            return f"查询时出现错误: {str(e)}"


class EnhancedCoreSystem:
    """增强版核心系统 - 集成真实RAG"""

    def __init__(self):
        self.businesses: Dict[str, Any] = {}
        self.rag_instances: Dict[str, RealRAGInstance] = {}
        self.image_manager = SimpleImageManager()
        self.processors: Dict[str, SimpleMultiModalProcessor] = {}

        # API配置
        self.api_config = self._load_api_config()

    def _load_api_config(self) -> Dict[str, Any]:
        """加载API配置"""
        # 优先从环境变量读取
        api_config = {
            "provider": os.getenv("LLM_PROVIDER", "openai"),  # openai 或 zhipu
            "api_key": os.getenv("API_KEY", ""),
            "base_url": os.getenv("BASE_URL", "https://api.openai.com/v1"),
            "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "vision_model": os.getenv("VISION_MODEL", "gpt-4o"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", "3072"))
        }

        # 如果没有设置API_KEY，尝试从配置文件读取
        if not api_config["api_key"]:
            config_file = Path("config.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    api_config.update(file_config.get("api", {}))

        return api_config

    def register_business(self, config):
        """注册业务 - 使用真实RAG"""
        self.businesses[config.business_id] = config

        if RAG_AVAILABLE and self.api_config.get("api_key"):
            try:
                self.rag_instances[config.business_id] = RealRAGInstance(
                    config.business_id,
                    self.api_config
                )
                print(f"注册业务 {config.name} - 使用真实RAG-Anything")
            except Exception as e:
                print(f"创建真实RAG实例失败: {e}")
                # 降级到模拟版本
                from main import MockRAGInstance
                self.rag_instances[config.business_id] = MockRAGInstance(config.business_id)
                print(f"注册业务 {config.name} - 降级使用模拟RAG")
        else:
            # 使用模拟版本
            from main import MockRAGInstance
            self.rag_instances[config.business_id] = MockRAGInstance(config.business_id)
            print(f"注册业务 {config.name} - 使用模拟RAG (RAG-Anything未可用)")

        self.processors[config.business_id] = SimpleMultiModalProcessor(config.business_id)

    async def process_crawler_data(self, business_id: str, json_file: str):
        """处理爬虫数据 - 与之前相同的逻辑"""
        if business_id not in self.businesses:
            raise ValueError(f"未注册的业务: {business_id}")

        # 读取JSON数据
        async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        print(f"加载了 {len(data)} 条数据")

        # 获取处理器
        processor = self.processors[business_id]
        rag = self.rag_instances[business_id]

        # 处理每个数据项
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

        print(f"数据处理完成，共处理 {len(data)} 条数据")

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
        """查询 - 使用真实RAG"""
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]
        return await rag.aquery(query, mode)

    def get_business_status(self, business_id: str) -> Dict[str, Any]:
        """获取业务状态"""
        if business_id not in self.businesses:
            return {"error": "业务不存在"}

        rag_type = "RealRAG" if isinstance(self.rag_instances[business_id], RealRAGInstance) else "MockRAG"

        return {
            "business_id": business_id,
            "name": self.businesses[business_id].name,
            "rag_type": rag_type,
            "api_configured": bool(self.api_config.get("api_key")),
            "status": "active"
        }


# 配置文件示例
CONFIG_TEMPLATE = {
    "api": {
        "provider": "zhipu",  # 或 "openai"
        "api_key": "your-api-key-here",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "llm_model": "glm-4-flash",
        "vision_model": "glm-4v-flash",
        "embedding_model": "embedding-3",
        "embedding_dim": 2048
    }
}

# 使用说明
USAGE_INSTRUCTIONS = """
使用真实RAG-Anything的配置步骤：

1. 安装RAG-Anything：
   pip install rag-anything==1.2.7

2. 配置API密钥（三种方式任选一种）：

   方式1 - 环境变量：
   export API_KEY="your-api-key"
   export LLM_PROVIDER="zhipu"  # 或 "openai"

   方式2 - 配置文件：
   创建 config.json 文件，内容参考 CONFIG_TEMPLATE

   方式3 - 直接修改代码中的 api_config

3. 运行系统：
   python enhanced_main.py

注意：如果没有配置API密钥，系统会自动降级使用模拟RAG版本。
"""

if __name__ == "__main__":
    print(USAGE_INSTRUCTIONS)