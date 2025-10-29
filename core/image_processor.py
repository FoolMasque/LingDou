# core/image_processor.py
"""图片处理模块"""

import base64
import asyncio
from typing import List, Optional
from PIL import Image
import io
import aiohttp

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageProcessor:
    """图片处理器"""

    DEFAULT_MAX_SIZE = 1024
    DEFAULT_QUALITY = 85

    async def process_user_images(
            self,
            image_urls: Optional[List[str]] = None,
            image_base64_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        处理用户上传的图片

        Args:
            image_urls: 图片URL列表
            image_base64_list: base64编码的图片列表

        Returns:
            优化后的base64图片列表
        """
        processed_images = []

        # 处理URL图片
        if image_urls:
            url_tasks = [
                self._download_and_optimize(url, idx)
                for idx, url in enumerate(image_urls)
            ]
            url_results = await asyncio.gather(*url_tasks, return_exceptions=True)

            for idx, result in enumerate(url_results):
                if isinstance(result, str):
                    processed_images.append(result)
                    logger.info(f"URL图片 {idx + 1} 处理成功")
                else:
                    logger.error(f"URL图片 {idx + 1} 处理失败: {result}")

        # 处理base64图片
        if image_base64_list:
            base64_tasks = [
                self._optimize_base64(img_b64, idx)
                for idx, img_b64 in enumerate(image_base64_list)
            ]
            base64_results = await asyncio.gather(*base64_tasks, return_exceptions=True)

            for idx, result in enumerate(base64_results):
                if isinstance(result, str):
                    processed_images.append(result)
                    logger.info(f"Base64图片 {idx + 1} 处理成功")
                else:
                    logger.error(f"Base64图片 {idx + 1} 处理失败: {result}")

        logger.info(f"图片处理完成: {len(processed_images)}张")
        return processed_images

    async def _download_and_optimize(self, url: str, index: int) -> str:
        """下载并优化图片"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}")

                    image_bytes = await response.read()
                    return await self._optimize_bytes(image_bytes)

        except Exception as e:
            logger.error(f"下载图片 {index + 1} 失败: {e}")
            raise

    async def _optimize_base64(self, base64_str: str, index: int) -> str:
        """优化base64图片"""
        try:
            # 清理base64
            cleaned = self._clean_base64(base64_str)

            # 在线程池中优化（CPU密集型）
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._optimize_base64_sync,
                cleaned
            )
        except Exception as e:
            logger.error(f"优化图片 {index + 1} 失败: {e}")
            raise

    def _clean_base64(self, base64_str: str) -> str:
        """清理base64字符串"""
        if not base64_str:
            raise ValueError("base64字符串为空")

        # 移除data URL前缀
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]

        # 移除空白字符
        base64_str = base64_str.strip().replace('\n', '').replace('\r', '')

        # 验证格式
        try:
            base64.b64decode(base64_str)
            return base64_str
        except Exception as e:
            raise ValueError(f"无效的base64格式: {e}")

    def _optimize_base64_sync(self, base64_str: str) -> str:
        """同步优化base64图片（在线程池中执行）"""
        try:
            # 解码
            image_bytes = base64.b64decode(base64_str)

            # 打开图片
            img = Image.open(io.BytesIO(image_bytes))

            # 调整尺寸
            img = self._resize_image(img)

            # 转换格式
            img = self._convert_to_rgb(img)

            # 保存为优化的JPEG
            output = io.BytesIO()
            img.save(
                output,
                format='JPEG',
                quality=self.DEFAULT_QUALITY,
                optimize=True
            )

            # 编码为base64
            return base64.b64encode(output.getvalue()).decode('utf-8')

        except Exception as e:
            logger.error(f"图片优化失败: {e}")
            return base64_str  # 返回原始值

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """调整图片尺寸"""
        if max(img.size) > self.DEFAULT_MAX_SIZE:
            ratio = self.DEFAULT_MAX_SIZE / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def _convert_to_rgb(self, img: Image.Image) -> Image.Image:
        """转换为RGB格式"""
        if img.mode == 'RGB':
            return img

        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                bg.paste(img, mask=img.split()[-1])
            else:
                bg.paste(img)
            return bg

        return img.convert('RGB')

    async def _optimize_bytes(self, image_bytes: bytes) -> str:
        """优化图片字节"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._optimize_bytes_sync,
            image_bytes
        )

    def _optimize_bytes_sync(self, image_bytes: bytes) -> str:
        """同步优化图片字节"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = self._resize_image(img)
            img = self._convert_to_rgb(img)

            output = io.BytesIO()
            img.save(output, format='JPEG', quality=self.DEFAULT_QUALITY, optimize=True)

            return base64.b64encode(output.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"优化图片失败: {e}")
            return base64.b64encode(image_bytes).decode('utf-8')