# core/components.py
"""
核心组件
"""
import base64
import logging
import os
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
from PIL import Image
from config.settings import settings
from utils.logger import setup_logger
from utils.url_helper import path_manager, build_remote_url
import io

logger = setup_logger(__name__)


@dataclass
class ImageMapping:
    """图片映射信息"""
    original_url: str
    local_path: str
    remote_url: str
    content_hash: str = ""  # 新增：内容哈希用于去重


@dataclass
class BusinessConfig:
    """业务配置"""
    business_id: str
    name: str
    image_fields: List[str]
    text_fields: List[str]


class ImageOptimizer:
    """图片优化器 - 专门用于VLM处理前的图片优化"""

    def __init__(self, max_dimension: int = 512, jpeg_quality: int = 85):
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def optimize_for_vlm(self, image_path: str) -> bytes:
        """
        为VLM优化图片（读取本地文件并压缩）
        返回: 优化后的图片字节数据
        """
        try:
            with Image.open(image_path) as img:
                # 转换RGBA为RGB
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # 计算新尺寸
                width, height = img.size
                if width > self.max_dimension or height > self.max_dimension:
                    if width > height:
                        new_width = self.max_dimension
                        new_height = int(height * (self.max_dimension / width))
                    else:
                        new_height = self.max_dimension
                        new_width = int(width * (self.max_dimension / height))

                    # 使用高质量重采样
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    logger.debug(f"VLM图片优化: {width}x{height} -> {new_width}x{new_height}")

                # 保存为优化的JPEG
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=self.jpeg_quality, optimize=True)
                return output.getvalue()

        except Exception as e:
            logger.error(f"图片优化失败 {image_path}: {e}")
            # 失败时返回原图
            with open(image_path, 'rb') as f:
                return f.read()

    def get_base64_optimized(self, image_path: str) -> str:
        """获取优化后的base64编码图片"""
        optimized_bytes = self.optimize_for_vlm(image_path)
        return base64.b64encode(optimized_bytes).decode('utf-8')

class ImageManager:
    """图片管理器"""

    def __init__(self):
        self.download_dir = Path(settings.image_storage)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.mappings: Dict[str, ImageMapping] = {}
        self.content_hash_map: Dict[str, str] = {}  # 内容哈希 -> 本地路径的映射

    async def download_images(self, image_urls: List[str], business_id: str) -> Dict[str, ImageMapping]:
        """
        下载图片并返回完整映射信息
        返回: {原始URL: ImageMapping对象}
        """
        if not image_urls:
            return {}

        business_dir = self.download_dir / business_id
        business_dir.mkdir(exist_ok=True)

        # 去重URL
        unique_urls = list(set(image_urls))
        logger.info(f"URL去重后需要下载 {len(unique_urls)} 张图片（原始 {len(image_urls)} 张）。下载中......")

        tasks = []
        semaphore = asyncio.Semaphore(10)  # 控制下载并发数

        async with aiohttp.ClientSession() as session:
            for url in image_urls:
                task = self._download_single_image(session, url, business_dir, business_id, semaphore)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # 构建映射字典
        mappings = {}
        duplicate_count = 0

        for result in results:
            if isinstance(result, ImageMapping):
                mappings[result.original_url] = result
                # 注册到全局路径管理器
                path_manager.register_mapping(result.local_path, result.remote_url)

                # 记录内容哈希
                if result.content_hash:
                    if result.content_hash in self.content_hash_map:
                        duplicate_count += 1
                        logger.debug(f"发现内容重复图片: {result.original_url}")
                    else:
                        self.content_hash_map[result.content_hash] = result.local_path

        success_count = len([r for r in results if isinstance(r, ImageMapping)])
        logger.info(f"图片下载完成，成功 {success_count}/{len(image_urls)} 张，内容重复 {duplicate_count} 张")

        self.mappings.update(mappings)
        return mappings

    async def _download_single_image(self, session, url, business_dir, business_id, semaphore):
        """下载单张图片"""
        async with semaphore:
            try:
                # 生成文件名
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                file_extension = self._get_file_extension(url)
                filename = f"{url_hash}.{file_extension}"
                file_path = business_dir / filename

                # 构建远程URL
                remote_url = build_remote_url(business_id, filename)
                local_path_str = str(file_path)

                # 检查文件是否已存在
                if file_path.exists():
                    # 读取已存在的文件计算哈希
                    content = file_path.read_bytes()
                    content_hash = hashlib.md5(content).hexdigest()
                    logger.debug(f"文件已存在: {filename}")
                    return ImageMapping(url, local_path_str, remote_url, content_hash)

                # 下载图片
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.read()

                        # 计算内容哈希
                        content_hash = hashlib.md5(content).hexdigest()
                        # 检查是否已有相同内容的图片
                        if content_hash in self.content_hash_map:
                            existing_path = self.content_hash_map[content_hash]
                            logger.debug(f"内容重复，复用已有图片: {existing_path}")
                            # 创建硬链接或复制文件
                            if not file_path.exists():
                                import shutil
                                shutil.copy2(existing_path, file_path)
                            return ImageMapping(url, local_path_str, remote_url, content_hash)

                        # 保存原图
                        file_path.write_bytes(content)
                        logger.debug(f"下载成功: {filename} (大小: {len(content) / 1024:.1f}KB)")

                        return ImageMapping(url, local_path_str, remote_url, content_hash)
                    else:
                        logger.warning(f"下载失败 {url}: HTTP {response.status}")
                        return None

            except Exception as e:
                logger.error(f"下载图片失败 {url}: {e}")
                return None

    def _get_file_extension(self, url: str) -> str:
        """获取文件扩展名"""
        url_lower = url.lower()
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            if f'.{ext}' in url_lower:
                return ext
        return 'jpg'  # 默认

    def get_mapping(self, original_url: str) -> ImageMapping:
        """获取URL的映射信息"""
        return self.mappings.get(original_url)

    def get_unique_images_by_content(self, image_urls: List[str]) -> Dict[str, List[str]]:
        """
        基于内容哈希获取去重后的图片
        返回: {content_hash: [url1, url2...]} 相同内容的URL列表
        """
        content_groups = {}

        for url in image_urls:
            mapping = self.mappings.get(url)
            if mapping and mapping.content_hash:
                if mapping.content_hash not in content_groups:
                    content_groups[mapping.content_hash] = []
                content_groups[mapping.content_hash].append(url)

        return content_groups

class MultiModalProcessor:
    """多模态处理器"""

    def __init__(self, business_id: str):
        self.business_id = business_id

    def build_modal_content(self, item: Dict[str, Any],
                                  image_manager: ImageManager = None) -> Dict[str, Any]:
        """
        构建多模态内容
        处理URL映射和内容去重
        """
        img_path = {}
        processed_hashes = set()
        processed_urls = set()

        # 收集所有图片信息（使用原始URL）
        all_images = []

        # 使用原始URL字段
        cover_pic_url = item.get("cover_pic_original") or item.get("cover_pic")
        if cover_pic_url:
            all_images.append(("cover_pic", cover_pic_url))

        # 详情图片 - 同样使用原始URL
        detail_images_urls = item.get("detail_images_original") or item.get("detail_images", [])
        if detail_images_urls:
            if isinstance(detail_images_urls, list):
                for i, img_url in enumerate(detail_images_urls):
                    if img_url:
                        all_images.append((f"detail_image_{i}", img_url))
            elif isinstance(detail_images_urls, str):
                all_images.append(("detail_image_0", detail_images_urls))

        logger.debug(f"收集到 {len(all_images)} 个图片位置")

        # 基于内容和URL去重
        for img_key, original_url in all_images:
            # 先检查URL是否已处理
            if original_url in processed_urls:
                logger.debug(f"跳过重复URL: {img_key} -> {original_url[-30:]}")
                continue

            # 获取映射信息
            if image_manager:
                mapping = image_manager.get_mapping(original_url)
                if mapping:
                    # 基于内容哈希去重
                    if mapping.content_hash:
                        if mapping.content_hash in processed_hashes:
                            logger.debug(f"跳过内容重复: {img_key}, hash={mapping.content_hash[:8]}")
                            continue
                        processed_hashes.add(mapping.content_hash)

                    # 使用本地路径
                    img_path[img_key] = mapping.local_path
                    processed_urls.add(original_url)
                    logger.debug(f"添加图片: {img_key} -> {Path(mapping.local_path).name}")
                else:
                    logger.warning(f"未找到映射: {img_key} -> {original_url[-30:]}")
                    # 降级：使用item中已有的本地路径
                    if img_key == "cover_pic" and item.get("cover_pic"):
                        img_path[img_key] = item["cover_pic"]
                    elif img_key.startswith("detail_image_"):
                        idx = int(img_key.split("_")[-1])
                        detail_images = item.get("detail_images", [])
                        if isinstance(detail_images, list) and idx < len(detail_images):
                            img_path[img_key] = detail_images[idx]
            else:
                # 没有image_manager时，使用URL去重
                if original_url not in processed_urls:
                    img_path[img_key] = original_url
                    processed_urls.add(original_url)

        # 统计去重效果
        original_count = len(all_images)
        final_count = len(img_path)
        if original_count != final_count:
            logger.info(f"图片去重: {original_count} -> {final_count} 张 (去除 {original_count - final_count} 张重复)")

        # 构建最终内容
        img_caption = self._build_caption(item)

        return {
            "img_path": img_path,
            "img_caption": [img_caption] if img_caption else [],
            "img_footnote": []
        }

    def _build_caption(self, item: Dict[str, Any]) -> str:
        """构建图片说明 - 家具业务专用"""
        caption_parts = ["以下是该商品的已知信息（仅供缺失字段回填，禁止臆造）："]
        # print(item)

        # 提取关键字段
        key_fields = {
            "风格": item.get("风格", ""),
            "子类": item.get("子类", ""),
            "商品名": item.get("商品名", ""),
            "材质规格": item.get("subtitle", ""),
            "关键词": item.get("keyword", "")
        }

        for field, value in key_fields.items():
            if value:
                caption_parts.append(f"- {field}: {value}")
                # print(f"- {field}: {value}")

        # 添加分析指导
        caption_parts.extend([
            "",
            "请分析产品图像，重点提取：",
            "1. 材质工艺和质感特征",
            "2. 设计风格和美学元素",
            "3. 功能特点和使用场景",
            "4. 尺寸规格和空间适配性"
        ])

        return "\n".join(caption_parts)
