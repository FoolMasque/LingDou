# utils/url_helper.py
"""
URL处理工具
"""
import re
from typing import Dict, Tuple
from config.settings import settings
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PathManager:
    """路径管理器 - 维护本地路径和远程URL的映射"""

    def __init__(self):
        self.local_to_remote: Dict[str, str] = {}
        self.remote_to_local: Dict[str, str] = {}

    def register_mapping(self, local_path: str, remote_url: str):
        """注册路径映射"""
        # 标准化路径格式
        normalized_local = self._normalize_path(local_path)

        self.local_to_remote[local_path] = remote_url
        self.local_to_remote[normalized_local] = remote_url
        self.remote_to_local[remote_url] = local_path

        logger.debug(f"注册映射: {normalized_local} -> {remote_url}")

    def get_remote_url(self, local_path: str) -> str:
        """获取远程URL"""
        # 尝试直接匹配
        if local_path in self.local_to_remote:
            return self.local_to_remote[local_path]

        # 尝试标准化后匹配
        normalized = self._normalize_path(local_path)
        if normalized in self.local_to_remote:
            return self.local_to_remote[normalized]

        # 如果找不到映射，尝试根据路径构建URL
        return self._build_url_from_path(local_path)

    def get_local_path(self, remote_url: str) -> str:
        """获取本地路径"""
        return self.remote_to_local.get(remote_url, remote_url)

    def _normalize_path(self, path: str) -> str:
        """标准化路径格式"""
        if not path:
            return path

        # 统一使用正斜杠
        normalized = path.replace('\\\\', '/').replace('\\', '/')

        # 处理相对路径
        if normalized.startswith('../'):
            normalized = normalized[3:]
        elif normalized.startswith('./'):
            normalized = normalized[2:]

        return normalized

    def _build_url_from_path(self, local_path: str) -> str:
        """从本地路径构建远程URL"""
        try:
            # 标准化路径
            normalized = self._normalize_path(local_path)

            # 提取文件名和业务ID
            path_parts = normalized.split('/')

            filename = ""
            business_id = ""

            # 查找文件名（包含扩展名的部分）
            for part in reversed(path_parts):
                if '.' in part and len(part) > 4:
                    filename = part
                    break

            # 查找业务ID（通常在images后面或者在路径中）
            for i, part in enumerate(path_parts):
                if part == 'images' and i + 1 < len(path_parts):
                    business_id = path_parts[i + 1]
                    break
                elif part in ['furniture', 'electronics', 'household']:  # 已知业务ID
                    business_id = part
                    break

            # 如果没有找到业务ID，使用默认值
            if not business_id:
                business_id = "furniture"  # 默认业务

            if filename:
                remote_url = f"{settings.static_base_url}/images/{business_id}/{filename}"
                logger.debug(f"构建URL: {normalized} -> {remote_url}")
                return remote_url

        except Exception as e:
            logger.error(f"构建URL失败: {local_path}, 错误: {e}")

        return local_path  # 失败时返回原路径

    def debug_mappings(self):
        """调试映射信息"""
        logger.info(f"当前映射数量: {len(self.local_to_remote)}")
        for local, remote in list(self.local_to_remote.items())[:5]:  # 只显示前5个
            logger.info(f"  {local} -> {remote}")


# 全局路径管理器实例
path_manager = PathManager()


def build_remote_url(business_id: str, filename: str) -> str:
    """构建远程访问URL"""
    return f"{settings.static_base_url}/images/{business_id}/{filename}"


def post_process_response_urls(response_text: str) -> str:
    """把响应中的本地图片路径转换为远程URL（单次替换，避免二次匹配）"""
    if not response_text:
        return response_text

    logger.debug("开始处理响应中的URL...")

    # 统一分隔符
    text = response_text.replace('\\\\', '/').replace('\\', '/')

    # 合并为一个大正则，并禁止匹配到已有 URL（用 (?<!://) 防止命中 http://.../images/... 的 images/... 部分）
    pattern = re.compile(
        r'''(?<!://)(
              \.\./static/images/[^/\s)'"\\]+/[^/\s)'"\\]+\.[a-zA-Z]{3,4} |
              static/images/[^/\s)'"\\]+/[^/\s)'"\\]+\.[a-zA-Z]{3,4}      |
              \./images/[^/\s)'"\\]+/[^/\s)'"\\]+\.[a-zA-Z]{3,4}          |
              images/[^/\s)'"\\]+/[^/\s)'"\\]+\.[a-zA-Z]{3,4}             |
              /[^/\s)'"\\]*static/images/[^/\s)'"\\]+/[^/\s)'"\\]+\.[a-zA-Z]{3,4}
           )''',
        re.IGNORECASE | re.VERBOSE
    )

    def replace_local_path(m):
        local_path = m.group(1)

        # 构造候选键（避免重复加前缀）
        candidates = [local_path]
        norm = local_path.lstrip('./')  # 去掉开头的 ./（不去 ../，让它保持一个候选）
        if local_path != norm:
            candidates.append(norm)

        if not local_path.startswith('../') and not local_path.startswith('static/'):
            candidates.append(f"../{local_path}")
        if not norm.startswith('static/'):
            candidates.append(f"static/{norm}")

        # 尝试映射表
        for key in candidates:
            remote_url = path_manager.get_remote_url(key)
            if remote_url and remote_url.startswith(('http://', 'https://')):
                logger.debug(f"URL转换成功: {local_path} -> {remote_url} (key: {key})")
                return remote_url

        # 兜底构建
        remote_url = path_manager._build_url_from_path(local_path)
        logger.debug(f"URL兜底构建: {local_path} -> {remote_url}")
        return remote_url

    # 单次替换，避免后续 pass 重复处理新 URL
    new_text = pattern.sub(replace_local_path, text)

    if new_text != response_text:
        logger.debug("URL后处理完成，发现并替换了本地路径")
    else:
        logger.debug("URL后处理完成，未发现需要替换的本地路径")
    return new_text


def normalize_path(path: str) -> str:
    """标准化路径格式"""
    if not path:
        return path

    # 统一使用正斜杠
    normalized = path.replace('\\\\', '/').replace('\\', '/')

    # 移除多余的点号
    normalized = normalized.replace('../', '').replace('./', '')

    return normalized

# 调试工具
def debug_image_access():
    """调试图片访问问题"""
    print(f"当前配置:")
    print(f"  端口: {settings.port}")
    print(f"  静态URL基础: {settings.static_base_url}")
    print(f"  图片存储目录: {settings.image_storage}")

    # 检查图片文件是否存在
    image_dir = Path(settings.image_storage)
    furniture_dir = image_dir / "furniture"

    print(f"  图片目录存在: {furniture_dir.exists()}")

    if furniture_dir.exists():
        image_files = list(furniture_dir.glob("*.jpg"))[:3]  # 只显示前3个
        print(f"  示例图片文件:")
        for img_file in image_files:
            remote_url = f"{settings.static_base_url}/images/furniture/{img_file.name}"
            print(f"    {img_file.name} -> {remote_url}")

    # 检查路径管理器状态
    path_manager.debug_mappings()


if __name__ == "__main__":
    debug_image_access()
