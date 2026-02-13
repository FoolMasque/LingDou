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
    # 定制化配置
    caption_template: Optional[str] = None  # 自定义图片描述模板
    caption_fields: Optional[Dict[str, str]] = None  # 字段映射：{显示名: JSON字段名}
    caption_instructions: Optional[List[str]] = None  # 图片分析指令列表
    entity_name_field: Optional[str] = None  # 实体名称字段（默认使用第一个text_field或"商品名"）
    vision_prompt_template: Optional[str] = None  # 自定义视觉分析提示词模板
    response_instruction: Optional[str] = None  # 自定义回复指导（例如：简短输出）
    field_mapping: Optional[Dict[str, str]] = None  # 通用字段映射 {标准字段: JSON字段}
    system_prompt_template: Optional[str] = None  # 自定义系统Prompt模板（覆盖默认的Role/Instructions）


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
        # ✅ 新目录结构：结构化数据图片存储在 rag_storage_{business_id}/images/
        # 但为了兼容，仍然使用settings.image_storage作为基础目录
        # 实际存储路径会在download_images中按business_id创建
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

        # ✅ 新目录结构：结构化数据图片存储在 rag_storage_{business_id}/images/
        # 为了统一管理，使用working_dir下的images目录
        # 确保路径格式正确：./rag_storage_{business_id}/images 或 rag_storage_{business_id}/images
        if settings.working_dir.startswith('./'):
            base_dir = Path(settings.working_dir.replace('./', ''))
        else:
            base_dir = Path(settings.working_dir)
        
        # 如果base_dir是rag_storage，直接拼接business_id
        if base_dir.name == 'rag_storage':
            business_dir = base_dir.parent / f"rag_storage_{business_id}" / "images"
        else:
            # 否则使用标准格式
            business_dir = Path(f"./rag_storage_{business_id}") / "images"
        business_dir.mkdir(parents=True, exist_ok=True)

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
                # 注册到全局路径管理器（支持多种路径格式）
                path_manager.register_mapping(result.local_path, result.remote_url)
                path_manager.register_mapping(result.local_path.replace('\\', '/'), result.remote_url)
                # 注册相对路径（相对于项目根目录）
                try:
                    project_root = Path(__file__).parent.parent.parent
                    rel_path = Path(result.local_path).relative_to(project_root)
                    path_manager.register_mapping(str(rel_path).replace('\\', '/'), result.remote_url)
                except ValueError:
                    pass  # 如果路径不在项目根目录下，忽略

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
                # ✅ 新路径格式：rag_storage_{business_id}/images/{filename}
                from config.settings import settings
                remote_url = f"{settings.static_base_url}/images/rag_storage_{business_id}/images/{filename}"
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

from collections import UserDict
class SafeDict(UserDict):
    def __missing__(self, key):
        return ""

class MultiModalProcessor:
    """多模态处理器"""

    def __init__(self, business_id: str, config: Optional[BusinessConfig] = None):
        self.business_id = business_id
        self.config = config

        # 如果配置中有自定义caption模板，使用自定义构建器
        if config and config.caption_template:
            self.caption_builder = self._build_custom_caption
        else:
            # 导入业务专用的caption构建器
            try:
                from config.runtime_prompt_patch import get_business_specific_caption_builder
                self.caption_builder = get_business_specific_caption_builder(business_id)
            except ImportError:
                self.caption_builder = self._build_generic_caption
                logger.warning(f"未找到业务专用caption构建器，使用通用构建器: {business_id}")




    def build_modal_content(self, item: Dict[str, Any],
                                  image_manager: ImageManager = None) -> Dict[str, Any]:
        """
        构建多模态内容 - 支持不同业务
        处理URL映射和内容去重
        """
        img_path = {}
        processed_hashes = set()
        processed_urls = set()

        # 收集所有图片信息（使用原始URL）
        all_images = self._collect_all_images(item)
        logger.debug(f"[{self.business_id}] 收集到 {len(all_images)} 个图片位置")

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
                    self._fallback_image_handling(item, img_key, img_path)
            else:
                # 没有image_manager时，使用URL去重
                if original_url not in processed_urls:
                    img_path[img_key] = original_url
                    processed_urls.add(original_url)

        # 统计去重效果
        original_count = len(all_images)
        final_count = len(img_path)
        if original_count != final_count:
            logger.info(f"[{self.business_id}] 图片去重: {original_count} -> {final_count} 张 (去除 {original_count - final_count} 张重复)")

        # 构建最终内容 - 使用业务专用的caption构建器
        img_caption = self.caption_builder(item)

        return {
            "img_path": img_path,
            "img_caption": [img_caption] if img_caption else [],
            "img_footnote": []
        }

    def _collect_all_images(self, item: Dict[str, Any]) -> List[tuple]:
        """收集所有图片URL - 根据业务类型适配不同字段"""
        all_images = []

        # 通用图片字段
        cover_pic_url = item.get("cover_pic_original") or item.get("cover_pic")
        if cover_pic_url:
            all_images.append(("cover_pic", cover_pic_url))

        # 详情图片
        detail_images_urls = item.get("detail_images_original") or item.get("detail_images", [])
        if detail_images_urls:
            if isinstance(detail_images_urls, list):
                for i, img_url in enumerate(detail_images_urls):
                    if img_url:
                        all_images.append((f"detail_image_{i}", img_url))
            elif isinstance(detail_images_urls, str):
                all_images.append(("detail_image_0", detail_images_urls))

        # 业务专用图片字段
        if self.business_id == "toilet":
            # 马桶业务特有的安装图片
            installation_pic = item.get("installation_pic_original") or item.get("installation_pic")
            if installation_pic:
                all_images.append(("installation_pic", installation_pic))

        elif self.business_id == "electronics":
            # 电器业务特有的功能展示图片
            feature_pics = item.get("feature_pics_original") or item.get("feature_pics", [])
            if feature_pics:
                if isinstance(feature_pics, list):
                    for i, img_url in enumerate(feature_pics):
                        if img_url:
                            all_images.append((f"feature_pic_{i}", img_url))
                elif isinstance(feature_pics, str):
                    all_images.append(("feature_pic_0", feature_pics))

        return all_images

    def _fallback_image_handling(self, item: Dict[str, Any], img_key: str, img_path: Dict[str, str]):
        """降级图片处理 - 使用item中已有的本地路径"""
        if img_key == "cover_pic" and item.get("cover_pic"):
            img_path[img_key] = item["cover_pic"]
        elif img_key.startswith("detail_image_"):
            idx = int(img_key.split("_")[-1])
            detail_images = item.get("detail_images", [])
            if isinstance(detail_images, list) and idx < len(detail_images):
                img_path[img_key] = detail_images[idx]
        elif img_key == "installation_pic" and item.get("installation_pic"):
            img_path[img_key] = item["installation_pic"]
        elif img_key.startswith("feature_pic_"):
            idx = int(img_key.split("_")[-1])
            feature_pics = item.get("feature_pics", [])
            if isinstance(feature_pics, list) and idx < len(feature_pics):
                img_path[img_key] = feature_pics[idx]

    def _build_custom_caption(self, item: Dict[str, Any]) -> str:
        """使用自定义模板构建图片说明（稳健替换显示名与json字段名）"""
        if not self.config or not self.config.caption_template:
            return self._build_generic_caption(item)

        try:
            # 构建字段映射：同时写入 display_name 和 json_field 两种 key
            field_values: Dict[str, Any] = {}

            if self.config.caption_fields:
                # caption_fields: { "显示名": "json_field" }
                for display_name, json_field in self.config.caption_fields.items():
                    value = item.get(json_field, "")
                    if value:
                        field_values[display_name] = value  # 用显示名替换 {产品名}
                        field_values[json_field] = value  # 用 json 字段名替换 {produce}
            else:
                # 没有显式映射时，用 text_fields 列表作为 json 字段名
                for field in (self.config.text_fields or []):
                    value = item.get(field, "")
                    if value:
                        field_values[field] = value

            # 额外把 item 中常见字段也放入，增加容错
            for k, v in item.items():
                if isinstance(k, str) and k not in field_values:
                    field_values[k] = v

            # 使用安全字典替换模板中的占位符（避免 KeyError）
            caption = self.config.caption_template
            caption = caption.format_map(SafeDict(field_values))

            # 添加分析指令（若有）
            if self.config.caption_instructions:
                caption = caption + "\n\n" + "\n".join(self.config.caption_instructions)

            return caption
        except Exception as e:
            logger.warning(f"自定义caption构建失败: {e}，使用通用构建器")
            return self._build_generic_caption(item)

    def _build_generic_caption(self, item: Dict[str, Any]) -> str:
        """通用图片说明构建 - 当没有业务专用构建器时使用"""
        caption_parts = ["以下是该商品的已知信息："]

        # 通用字段
        common_fields = {
            "商品名": item.get("商品名", ""),
            "品牌": item.get("品牌", ""),
            "型号": item.get("型号", ""),
            "规格": item.get("规格", "") or item.get("subtitle", ""),
            "功能": item.get("功能", ""),
            "关键词": item.get("keyword", "")
        }

        for field, value in common_fields.items():
            if value:
                caption_parts.append(f"- {field}: {value}")

        caption_parts.extend([
            "",
            "请分析产品图像，重点提取：",
            "1. 外观设计和材质特征",
            "2. 功能特点和技术亮点",
            "3. 使用场景和适用性",
            "4. 品质和工艺表现"
        ])

        return "\n".join(caption_parts)
