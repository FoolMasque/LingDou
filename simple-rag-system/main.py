"""
核心模块最小可行版本
先实现基本功能，确保能跑通整个流程
"""

import asyncio
import json
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
import base64
import time


# 模拟RAG-Anything导入 - **MockRAGInstance**
# ↓↓↓↓ 替换MockRAGInstance为真正的RAG-Anything | Real RAG-Anything
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    RAG_AVAILABLE = True
except ImportError:
    print("RAG-Anything未安装，请运行: pip install rag-anything==1.2.7")
    RAG_AVAILABLE = False

@dataclass
class BusinessConfig:
    """业务配置"""
    business_id: str
    name: str
    image_fields: List[str]
    text_fields: List[str]


class SimpleImageManager:
    """简化版图片管理器"""

    def __init__(self, storage_path: str = "./images"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

    async def download_images(self, urls: List[str], business_id: str) -> Dict[str, str]:
        """下载图片并返回本地URL映射"""
        business_dir = self.storage_path / business_id
        business_dir.mkdir(exist_ok=True)

        url_mapping = {}
        print(f"开始下载 {len(urls)} 张图片...")

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for i, url in enumerate(urls, 1):
                try:
                    filename = self._generate_filename(url)
                    local_file = business_dir / filename

                    if not local_file.exists():
                        print(f"下载图片 {i}/{len(urls)}: {url}")
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                content = await resp.read()
                                async with aiofiles.open(local_file, 'wb') as f:
                                    await f.write(content)
                                print(f"下载成功: {filename}")
                            else:
                                print(f"下载失败 {url}: HTTP {resp.status}")
                    else:
                        print(f"文件已存在: {filename}")

                    # 生成可访问的URL（简化版，实际部署时需要配置nginx）
                    local_url = f"http://localhost:8000/images/{business_id}/{filename}"
                    url_mapping[url] = local_url

                except Exception as e:
                    print(f"下载图片失败 {url}: {e}")
                    url_mapping[url] = url  # 保留原URL

        print(f"图片下载完成，成功 {len(url_mapping)} 张")
        return url_mapping

    def _generate_filename(self, url: str) -> str:
        """生成文件名"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = Path(url.split('?')[0]).suffix or '.jpg'
        return f"{url_hash}{ext}"


class SimpleMultiModalProcessor:
    """简化版多模态处理器 - 集成🤷‍♀️李馨逻辑"""

    def __init__(self, business_id: str):
        self.business_id = business_id

    def generate_product_caption(self, item: Dict[str, Any]) -> str:
        """生成产品描述 - 参考🤷‍♀️李馨逻辑"""
        # 根据业务类型定制字段映射
        field_mappings = self._get_field_mappings()

        fields = []
        for display_name, field_key in field_mappings.items():
            value = item.get(field_key)
            if value:
                fields.append(f"- {display_name}: {value}")

        fields_str = "\n".join(fields) if fields else "无可用字段信息"

        # by🤷‍♀️李馨
        caption = f"""以下是该商品的已知信息（仅供缺失字段回填，禁止臆造）：
{fields_str}

请分析产品图像，重点提取：
1. 材质工艺和质感特征
2. 设计风格和美学元素  
3. 功能特点和使用场景
4. 尺寸规格和空间适配性
"""
        return caption

    def _get_field_mappings(self) -> Dict[str, str]:
        """获取字段映射"""
        mappings = {
            "furniture": {
                "风格": "风格",
                "子类": "子类",
                "商品名": "商品名",
                "规格说明": "subtitle",
                "关键词": "keyword"
            }
        }
        return mappings.get(self.business_id, {})

    def build_modal_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """构建多模态内容 - 参考实习生结构"""
        # 处理图片路径
        img_path = {
            "cover_pic": item.get("cover_pic", ""),
            "detail_images": item.get("detail_images", [])
        }

        # 生成caption
        caption = self.generate_product_caption(item)

        return {
            "img_path": img_path,
            "img_caption": [caption],
            "img_footnote": []
        }


class MockRAGInstance:
    """模拟RAG实例 - 用于测试，实际使用时替换为真实的RAG-Anything"""

    def __init__(self, business_id: str):
        self.business_id = business_id
        self.knowledge_base = []  # 简单的内存存储

    async def process_multimodal_content(self, modal_content: Dict[str, Any],
                                         entity_name: str, file_path: str):
        """模拟多模态内容处理"""
        # 简单存储到内存中
        record = {
            "entity_name": entity_name,
            "content": modal_content,
            "file_path": file_path,
            "processed_time": time.time()
        }
        self.knowledge_base.append(record)
        print(f"处理多模态内容: {entity_name}")

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """模拟查询处理 - 改进匹配算法"""
        relevant_items = []
        query_lower = query.lower()

        # 提取查询中的关键词
        query_keywords = []
        for word in query_lower.replace('，', ' ').replace('。', ' ').split():
            if len(word) > 1:  # 过滤单字符
                query_keywords.append(word)

        for record in self.knowledge_base:
            content_str = json.dumps(record["content"], ensure_ascii=False).lower()
            score = 0

            # 计算匹配分数
            for keyword in query_keywords:
                if keyword in content_str:
                    score += 1
                # 模糊匹配
                if keyword in record["entity_name"].lower():
                    score += 2  # 实体名匹配权重更高

            if score > 0:
                relevant_items.append((record, score))

        if not relevant_items:
            # 尝试更宽泛的匹配
            for record in self.knowledge_base:
                entity_name = record["entity_name"].lower()
                if any(char in entity_name for char in ['茶', '桌', '床', '凳', '椅']):
                    if any(keyword in ['茶', '桌', '床', '凳', '椅', '家具', '推荐'] for keyword in query_keywords):
                        relevant_items.append((record, 0.5))

        if not relevant_items:
            return "抱歉，没有找到相关信息。您可以尝试搜索：茶桌、长凳、床等关键词。"

        # 按分数排序
        relevant_items.sort(key=lambda x: x[1], reverse=True)

        # 生成回答
        response = f"根据您的查询「{query}」，我为您推荐以下产品：\n\n"

        for i, (item, score) in enumerate(relevant_items[:3], 1):
            entity_name = item["entity_name"]
            response += f"{i}. **{entity_name}**\n"

            # 提取详细信息
            modal_content = item["content"]
            captions = modal_content.get("img_caption", [])
            if captions:
                caption = captions[0]
                lines = caption.split('\n')
                for line in lines:
                    if line.strip().startswith('- '):
                        field_info = line.strip()[2:]  # 移除"- "
                        response += f"   • {field_info}\n"

            # 添加图片信息
            img_path = modal_content.get("img_path", {})
            if img_path.get("cover_pic"):
                response += f"   • 封面图：{img_path['cover_pic']}\n"

            response += "\n"

        return response.strip()


class SimpleCoreSystem:
    """简化版核心系统"""

    def __init__(self):
        self.businesses: Dict[str, BusinessConfig] = {}
        self.rag_instances: Dict[str, MockRAGInstance] = {}
        self.image_manager = SimpleImageManager()
        self.processors: Dict[str, SimpleMultiModalProcessor] = {}

    def register_business(self, config: BusinessConfig):
        """注册业务"""
        self.businesses[config.business_id] = config
        self.rag_instances[config.business_id] = MockRAGInstance(config.business_id)
        self.processors[config.business_id] = SimpleMultiModalProcessor(config.business_id)
        print(f"注册业务: {config.name}")

    async def process_crawler_data(self, business_id: str, json_file: str):
        """处理爬虫数据"""
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
                    # 更新item中的图片URL
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
        """查询"""
        if business_id not in self.rag_instances:
            raise ValueError(f"未注册的业务: {business_id}")

        rag = self.rag_instances[business_id]
        return await rag.aquery(query, mode)

    def get_business_status(self, business_id: str) -> Dict[str, Any]:
        """获取业务状态"""
        if business_id not in self.businesses:
            return {"error": "业务不存在"}

        rag = self.rag_instances[business_id]
        return {
            "business_id": business_id,
            "name": self.businesses[business_id].name,
            "knowledge_count": len(rag.knowledge_base),
            "status": "active"
        }


# 简单的FastAPI应用
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager


# 创建生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    furniture_config = BusinessConfig(
        business_id="furniture",
        name="家具商城",
        image_fields=["cover_pic", "detail_images"],
        text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
    )
    core_system.register_business(furniture_config)
    print("应用启动完成")

    yield  # 应用运行期间

    # 关闭时执行（可选）
    print("应用关闭")


app = FastAPI(
    title="简化版RAG系统",
    version="0.1.0",
    lifespan=lifespan
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
app.mount("/images", StaticFiles(directory="images"), name="images")

core_system = SimpleCoreSystem()


class ProcessDataRequest(BaseModel):
    business_id: str
    json_file: str


class QueryRequest(BaseModel):
    business_id: str
    query: str
    mode: str = "hybrid"


class BusinessRegistration(BaseModel):
    business_id: str
    name: str
    image_fields: List[str]
    text_fields: List[str]


@app.post("/api/register_business")
async def register_business(request: BusinessRegistration):
    """注册业务"""
    config = BusinessConfig(
        business_id=request.business_id,
        name=request.name,
        image_fields=request.image_fields,
        text_fields=request.text_fields
    )
    core_system.register_business(config)
    return {"success": True, "message": f"业务 {request.business_id} 注册成功"}


@app.post("/api/process_data")
async def process_data(request: ProcessDataRequest, background_tasks: BackgroundTasks):
    """处理数据"""
    try:
        # 使用后台任务处理，避免请求超时
        background_tasks.add_task(
            core_system.process_crawler_data,
            request.business_id,
            request.json_file
        )
        return {"success": True, "message": "数据处理任务已启动，请稍后查询"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(request: QueryRequest):
    """查询"""
    try:
        result = await core_system.query(request.business_id, request.query, request.mode)
        return {
            "success": True,
            "query": request.query,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{business_id}")
async def get_status(business_id: str):
    """获取业务状态"""
    return core_system.get_business_status(business_id)


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "businesses": list(core_system.businesses.keys())}


# Dify集成接口
@app.post("/dify/knowledge/query")
async def dify_query(business_id: str, query: str):
    """Dify外部知识库接口"""
    try:
        result = await core_system.query(business_id, query)
        return {
            "records": [{
                "content": result,
                "score": 0.9,
                "title": f"{business_id}产品信息",
                "metadata": {"business_id": business_id}
            }]
        }
    except Exception as e:
        return {"error": str(e), "records": []}



async def main():
    """使用示例"""
    system = SimpleCoreSystem()

    # 1. 注册家具业务
    furniture_config = BusinessConfig(
        business_id="furniture",
        name="侘界家具",
        image_fields=["cover_pic", "detail_images"],
        text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
    )
    system.register_business(furniture_config)

    # 2. 处理爬虫数据
    await system.process_crawler_data("furniture", "mosyy_goods.json")

    # 3. 测试查询
    result = await system.query("furniture", "茶桌")
    print("查询结果:")
    print(result)

    # 4. 查看业务状态
    status = system.get_business_status("furniture")
    print("业务状态:")
    print(status)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())

    # 或者启动API服务
    # import uvicorn
    #
    # uvicorn.run(app, host="0.0.0.0", port=8000)