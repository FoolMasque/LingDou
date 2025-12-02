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

        # 处理Windows绝对路径（如 D:/path/to/file）
        # 保留盘符，但统一为正斜杠格式
        if len(normalized) > 2 and normalized[1] == ':' and normalized[2] == '/':
            # Windows绝对路径，保持原样（如 D:/path/to/file）
            pass
        elif normalized.startswith('../'):
            normalized = normalized[3:]
        elif normalized.startswith('./'):
            normalized = normalized[2:]

        return normalized

    def _build_url_from_path(self, local_path: str) -> str:
        """从本地路径构建远程URL"""
        try:
            # 标准化路径
            normalized = self._normalize_path(local_path)

            # 检查是否是rag_storage下的路径
            # 格式：rag_storage_{business_id}/parsed/{doc}/images/{filename}
            # 或：rag_storage_{business_id}/images/{filename}
            # 支持绝对路径和相对路径
            if 'rag_storage_' in normalized:
                # 提取rag_storage_后面的部分作为相对路径
                # 例如：
                # - 相对路径：rag_storage_ARglasses/parsed/M400-AR智能眼镜/M400-AR智能眼镜/auto/images/xxx.jpg
                # - 绝对路径：D:/Dev/myDev/LingDou/rag_storage_ARglasses/parsed/M400-AR智能眼镜/M400-AR智能眼镜/auto/images/xxx.jpg
                # 转换为：rag_storage_ARglasses/parsed/M400-AR智能眼镜/M400-AR智能眼镜/auto/images/xxx.jpg
                # URL格式：/images/rag_storage_ARglasses/parsed/M400-AR智能眼镜/M400-AR智能眼镜/auto/images/xxx.jpg
                
                # 找到rag_storage_的位置
                idx = normalized.find('rag_storage_')
                if idx >= 0:
                    # 获取rag_storage_后面的所有内容
                    rel_path = normalized[idx:]
                    # URL编码中文字符
                    from urllib.parse import quote
                    # 对路径中的每个部分进行编码（保留斜杠）
                    path_parts = rel_path.split('/')
                    encoded_parts = [quote(part, safe='') for part in path_parts]
                    encoded_path = '/'.join(encoded_parts)
                    remote_url = f"{settings.static_base_url}/images/{encoded_path}"
                    logger.debug(f"构建URL（rag_storage）: {normalized} -> {remote_url}")
                    return remote_url

            # 旧格式兼容：static/images/business_id/filename
            path_parts = normalized.split('/')

            filename = ""
            business_id = ""

            # 查找文件名（包含扩展名的部分）
            for part in reversed(path_parts):
                if '.' in part and len(part) > 4:
                    filename = part
                    break

            # 查找业务ID（动态提取，不硬编码）
            for i, part in enumerate(path_parts):
                if part == 'images':
                    # 检查是否是static/images格式
                    if i > 0 and path_parts[i - 1] == 'static' and i + 1 < len(path_parts):
                        business_id = path_parts[i + 1]
                        break
                    # 检查是否是rag_storage格式：rag_storage_{business_id}/images/...
                    if i > 0:
                        prev_part = path_parts[i - 1]
                        if prev_part.startswith('rag_storage_'):
                            business_id = prev_part.replace('rag_storage_', '')
                            break
                elif part.startswith('rag_storage_'):
                    # 直接找到rag_storage_{business_id}
                    business_id = part.replace('rag_storage_', '')
                    break

            # 如果没有找到业务ID，尝试从路径中推断（兼容旧格式）
            if not business_id:
                # 检查是否在static/images目录下
                if 'static' in path_parts and 'images' in path_parts:
                    idx = path_parts.index('images')
                    if idx + 1 < len(path_parts):
                        business_id = path_parts[idx + 1]
                
                # 如果还是没找到，使用默认值
                if not business_id:
                    business_id = "furniture"  # 默认业务（向后兼容）

            if filename:
                remote_url = f"{settings.static_base_url}/images/{business_id}/{filename}"
                logger.debug(f"构建URL（旧格式）: {normalized} -> {remote_url}")
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

    logger.info("开始处理响应中的URL...")

    # 统一分隔符（先处理反斜杠，再处理双反斜杠）
    # 注意：保留原始文本用于后续处理，但创建标准化版本用于匹配
    original_text = response_text
    text = response_text.replace('\\\\', '/').replace('\\', '/')
    
    # ✅ 关键：先提取所有已经是URL的图片路径，避免重复处理
    # 匹配所有 http:// 或 https:// 开头的完整URL（包含图片扩展名）
    url_pattern = re.compile(
        r'https?://[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)',
        re.IGNORECASE
    )
    
    # 找到所有已经是URL的图片路径，并创建保护映射
    url_matches = {}
    for match in url_pattern.finditer(text):
        url = match.group(0)
        # 为每个URL创建一个唯一标记
        placeholder = f"__PROTECTED_URL_{len(url_matches)}__"
        url_matches[placeholder] = url
    
    # 用占位符替换所有已存在的URL
    protected_text = text
    for placeholder, url in url_matches.items():
        protected_text = protected_text.replace(url, placeholder)

    # 合并为一个大正则，匹配本地路径
    # 支持多种路径格式：
    # 1. static/images/business_id/filename
    # 2. rag_storage_{business_id}/parsed/{doc}/images/filename
    # 3. rag_storage_{business_id}/images/filename
    # 4. 绝对路径：D:/path/to/rag_storage_{business_id}/parsed/{doc}/images/filename
    # ✅ 关键：使用负向后顾断言，确保前面不是 http:// 或 https://
    # 同时检查前面不是 /images/ 后面跟着 http:// 或 https:// 的情况
    pattern = re.compile(
        r'''(?<!http://)(?<!https://)(?<!://)(
              \.\./static/images/[^\s)\]]+/[^\s)\]]+\.[a-zA-Z]{3,4} |
              static/images/[^\s)\]]+/[^\s)\]]+\.[a-zA-Z]{3,4}      |
              \./images/[^\s)\]]+/[^\s)\]]+\.[a-zA-Z]{3,4}          |
              images/[^\s)\]]+/[^\s)\]]+\.[a-zA-Z]{3,4}             |
              /[^\s)\]]*static/images/[^\s)\]]+/[^\s)\]]+\.[a-zA-Z]{3,4} |
              [A-Za-z]:[^\s)\]]*rag_storage_[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp) |
              /[^\s)\]]*rag_storage_[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp) |
              rag_storage_[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)
           )(?!://)(?!http://)(?!https://)''',
        re.IGNORECASE | re.VERBOSE
    )

    def replace_local_path(m):
        local_path = m.group(1)

        # ✅ 如果路径已经是完整URL，直接返回，避免重复转换
        if local_path.startswith(('http://', 'https://')):
            logger.debug(f"路径已经是URL，跳过转换: {local_path[:80]}...")
            return local_path

        # ✅ 额外检查：如果匹配的路径前面有占位符标记，说明它在被保护的URL中，应该跳过
        start_pos = m.start()
        if start_pos > 0:
            prefix = protected_text[max(0, start_pos - 30):start_pos]
            if '__PROTECTED_URL_' in prefix:
                logger.debug(f"路径在被保护的URL中，跳过: {local_path[:80]}...")
                return local_path

        # 标准化路径（处理绝对路径和相对路径）
        normalized_path = local_path.replace('\\', '/')

        # ✅ 统一处理：如果路径包含 rag_storage_，提取相对部分
        rel_path_from_rag_storage = None
        if 'rag_storage_' in normalized_path:
            idx = normalized_path.find('rag_storage_')
            if idx >= 0:
                rel_path_from_rag_storage = normalized_path[idx:]
                # 去掉开头的 ./ 或 ../
                if rel_path_from_rag_storage.startswith('./'):
                    rel_path_from_rag_storage = rel_path_from_rag_storage[2:]
                elif rel_path_from_rag_storage.startswith('../'):
                    rel_path_from_rag_storage = rel_path_from_rag_storage[3:]

        # 构造候选键（按优先级排序）
        candidates = []

        # 优先级1：原始路径
        candidates.append(local_path)
        candidates.append(normalized_path)

        # 优先级2：如果提取到了 rag_storage_ 相对路径，优先尝试
        if rel_path_from_rag_storage:
            candidates.append(rel_path_from_rag_storage)
            # 尝试映射或构建URL（早期返回优化）
            remote_url = path_manager.get_remote_url(rel_path_from_rag_storage)
            if remote_url and remote_url.startswith(('http://', 'https://')):
                logger.debug(f"URL转换成功（rag_storage相对路径）: {local_path} -> {remote_url}")
                return remote_url
            # 如果映射表中没有，使用 _build_url_from_path
            remote_url = path_manager._build_url_from_path(rel_path_from_rag_storage)
            if remote_url and remote_url.startswith(('http://', 'https://')):
                logger.debug(f"URL构建成功（rag_storage相对路径）: {local_path} -> {remote_url}")
                return remote_url

        # 优先级3：其他候选键
        norm = normalized_path.lstrip('./')
        if normalized_path != norm:
            candidates.append(norm)

        if not normalized_path.startswith('../') and not normalized_path.startswith('static/'):
            candidates.append(f"../{normalized_path}")
        if not norm.startswith('static/'):
            candidates.append(f"static/{norm}")

        # 尝试映射表（遍历所有候选键）
        for key in candidates:
            remote_url = path_manager.get_remote_url(key)
            if remote_url and remote_url.startswith(('http://', 'https://')):
                logger.debug(f"URL转换成功: {local_path} -> {remote_url} (key: {key})")
                return remote_url

        # 兜底构建
        remote_url = path_manager._build_url_from_path(local_path)
        # ✅ 确保返回的是完整URL，不是路径
        if remote_url and remote_url.startswith(('http://', 'https://')):
            logger.info(f"URL兜底构建: {local_path[:100]}... -> {remote_url[:100]}...")
            return remote_url
        else:
            # 如果构建失败，返回原路径（避免破坏文本）
            logger.warning(f"URL构建失败，返回原路径: {local_path[:100]}...")
            return local_path

    # 在保护后的文本上进行替换
    processed_text = pattern.sub(replace_local_path, protected_text)
    
    # 恢复被保护的URL
    for placeholder, url in url_matches.items():
        processed_text = processed_text.replace(placeholder, url)
    
    new_text = processed_text

    # if new_text != response_text:
    #     logger.info("URL后处理完成，发现并替换了本地路径")
    #     # 显示替换前后的差异（仅显示前200个字符）
    #     if len(response_text) > 200:
    #         logger.info(f"替换前（前200字符）: {response_text[:200]}...")
    #     else:
    #         logger.info(f"替换前: {response_text}")
    #     if len(new_text) > 200:
    #         logger.info(f"替换后（前200字符）: {new_text[:200]}...")
    #     else:
    #         logger.info(f"替换后: {new_text}")
    # else:
    #     logger.info("URL后处理完成，未发现需要替换的本地路径")
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
