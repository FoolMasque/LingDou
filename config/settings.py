# config/settings.py
"""
统一配置管理
"""
import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """会话管理配置"""
    enabled: bool = True
    storage_backend: Literal["file", "redis", "memory"] = "file"
    default_max_turns: int = 5
    default_max_tokens: int = 2000
    cache_size: int = 100
    cleanup_days: int = 7

    # 文件存储配置
    file_storage_dir: str = "conversations"

    # Redis 存储配置
    # 如果应用在Docker中，可以通过环境变量REDIS_URL覆盖
    redis_url: str = "redis://localhost:16380"
    redis_db: int = 0  # 使用独立的Redis容器，可以使用db 0
    redis_prefix: str = "lingdou:"
    redis_ttl: int = 2592000  # 30天
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    redis_decode_responses: bool = True
    redis_password: Optional[str] = None


@dataclass
class RerankConfig:
    """Rerank 配置"""
    enabled: bool = False
    model: str = "gte-rerank-v2"
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    api_key: str = ""
    device: str = "cpu"  # cpu 或 cuda
    top_n: int = 5
    score_threshold: float = 0.0 #只做排序不过滤
    use_fp16: bool = False
    max_length: int = 512

@dataclass
class RAGAnythingConfig:
    """RAGAnything 配置"""
    enabled: bool = True
    use_mineru: bool = True
    parse_method: str = "auto"  # auto/ocr/txt
    parser: str = "mineru"
    enable_formula: bool = True
    enable_table: bool = True
    enable_ocr: bool = True
    chunk_size: int = 3000
    chunk_overlap: int = 150
    enable_image_processing: bool = False
    smart_parse: bool = True
    text_density_threshold: int = 700  # 平均每页字符数阈值
    min_text_chars: int = 200  # 将页面视作文本页的最小字符数
    sample_page_limit: int = 6  # 采样页数量
    max_txt_pages: int = 80  # 当页数过大时仍使用mineru
    max_txt_file_mb: float = 15.0  # 文件过大时启用mineru
    image_page_ratio_threshold: float = 0.3  # 图片页占比阈值
    entity_extract_rounds: int = 1
    # 知识图谱抽取策略配置
    kg_extraction_mode: str = "adaptive"  # adaptive（自适应）/ratio（比例）/limit（限制数量）/all（全部）
    kg_extraction_ratio: float = 0.3  # 抽取比例（0.0-1.0），当mode=ratio时生效
    kg_max_chunks_per_doc: int = 0  # 每个文档最多抽取的chunk数（0=不限制），当mode=limit时生效
    kg_max_extraction_time: int = 300  # 最大抽取时间（秒），当mode=adaptive时，超过此时间停止抽取
    kg_chunk_selection_strategy: str = "first"  # 选择策略：first（前N个）/random（随机N个）/important（重要chunk，基于关键词）
    kg_important_keywords: Optional[List[str]] = None  # 重要chunk的关键词列表，当strategy=important时使用
    # 多模态内容权重配置（用于关系抽取和实体重要性评估）
    multimodal_weights: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "image": 1.5,
        "table": 1.2,
        "formula": 1.0,
        "text": 1.0
    })
    
    # 其他配置
    backend: Optional[str] = None
    lang: str = "ch"  # ch/en
    chinese_font_path: Optional[str] = None  # 如果为None，将自动检测Windows系统字体

@dataclass
class SystemConfig:
    """
    系统配置
    """
    # API配置
    provider: str = "" #"openai"
    api_key: str = "" #""
    base_url: str = "" #""

    # 模型配置
    llm_model: str = "" #"gpt-4o-mini"
    vision_model: str = ""# "gpt-4o"
    embedding_model: str = ""# "text-embedding-3-large"
    embedding_dim: int = 1024 #3072
    max_token_size: int = 2048 # 8192

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8008
    static_base_url: str = "http://47.100.14.93:8008"

    # 存储配置
    working_dir: str = "./rag_storage"
    image_storage: str = "./static/images" # 废弃

    log_dir: Path = Path("./logs")

    # 业务配置
    use_chinese_prompts: bool = True
    default_language: str = "zh-CN"
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    max_dimension: int = 512 # 图片压缩最长边像素
    jpeg_quality: int = 85 # 图片质量百分比
    debug : bool=False

    # 会话管理配置
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

    # Rerank 配置
    rerank: RerankConfig = field(default_factory=RerankConfig)
    
    # RAG-Anything 配置
    rag_anything: RAGAnythingConfig = field(default_factory=RAGAnythingConfig)

    # 模型提供商配置映射
    provider_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-4o-mini",
            "vision_model": "gpt-4o",
            "embedding_model": "text-embedding-3-large",
            "embedding_dim": 3072,
            "max_token_size": 8192
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "llm_model": "glm-4",
            "vision_model": "glm-4.5v",
            "embedding_model": "embedding-2",
            "embedding_dim": 1024,
            "max_token_size": 2048
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "llm_model": "deepseek-chat",
            "vision_model": "deepseek-vl",
            "embedding_model": "deepseek-embedding",
            "embedding_dim": 1536,
            "max_token_size": 8192
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_model": "qwen3.7-plus",
            "vision_model": "qwen3.7-plus",
            "embedding_model": "text-embedding-v4",
            "embedding_dim": 1024,
            "max_token_size": 8192
        }

    })

    def __post_init__(self):
        """初始化后处理 - 动态设置static_base_url"""
        if not self.static_base_url:
            # 自动构建static_base_url
            self.static_base_url = f"http://localhost:{self.port}"

    def get_storage_config(self) -> Dict[str, Any]:
        """获取当前存储后端的配置"""
        if self.conversation.storage_backend == "redis":
            return {
                "redis_url": self.conversation.redis_url,
                "db": self.conversation.redis_db,
                "prefix": self.conversation.redis_prefix,
                "ttl": self.conversation.redis_ttl,
                "max_connections": self.conversation.redis_max_connections,
                "socket_timeout": self.conversation.redis_socket_timeout,
                "decode_responses": self.conversation.redis_decode_responses,
                "password": self.conversation.redis_password
            }
        elif self.conversation.storage_backend == "file":
            return {
                "storage_dir": self.conversation.file_storage_dir
            }
        else:  # memory
            return {}

    @classmethod
    def load_config(cls) -> 'SystemConfig':
        """
        加载配置：环境变量 > config.json > 默认值
        """
        config = cls()

        # 1. 从config.json读取
        # 使用 __file__ 获取绝对路径，确保无论从哪里运行都能找到 config.json
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        config_file = project_root / "config.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # 更新基础配置
                    for key, value in file_config.items():
                        if key == 'conversation':
                            # 处理会话配置
                            config.conversation = cls._load_conversation_config(value)
                        elif key == 'rerank':
                            # 处理 rerank 配置
                            config.rerank = cls._load_rerank_config(value)
                        elif key == 'provider_configs':
                            # 更新提供商配置
                            config.provider_configs.update(value)
                        elif key == 'rag_anything':
                            config.rag_anything = cls._load_rag_anything_config(value)
                        elif hasattr(config, key):
                            # 直接设置基础配置字段
                            setattr(config, key, value)
                        else:
                            # 调试：记录未识别的配置项
                            logger.info(f"警告: 未识别的配置项 {key} (跳过)")
                    logger.info(f"读取配置文件完毕")
            except Exception as e:
                logger.error(f"读取配置文件失败: {e}")
        # 2. 环境变量覆盖
        config._load_from_env()

        # 3. 根据provider应用默认配置
        if config.provider and config.provider in config.provider_configs:
            provider_config = config.provider_configs[config.provider]
            for key, value in provider_config.items():
                current_value = getattr(config, key, None)
                default_value = getattr(cls(), key, None)
                # 如果当前值为空或等于默认值，则应用provider配置
                if not current_value or current_value == default_value:
                    setattr(config, key, value)
                else:
                    logger.info(f"保留已有配置: {key} = {current_value} (provider默认值: {value})")
        elif config.provider:
            logger.warning(f"警告: provider '{config.provider}' 不在 provider_configs 中")

        return config

    @staticmethod
    def _load_conversation_config(conv_dict: Dict[str, Any]) -> ConversationConfig:
        """从字典加载会话配置"""
        conv_config = ConversationConfig()

        # 直接映射的字段
        simple_fields = [
            'enabled', 'storage_backend', 'default_max_turns',
            'default_max_tokens', 'cache_size', 'cleanup_days'
        ]
        for field_name in simple_fields:
            if field_name in conv_dict:
                setattr(conv_config, field_name, conv_dict[field_name])

        # 处理嵌套的 storage_config
        if 'storage_config' in conv_dict:
            storage_configs = conv_dict['storage_config']

            # Redis 配置
            if 'redis' in storage_configs:
                redis_cfg = storage_configs['redis']
                conv_config.redis_url = redis_cfg.get('url', conv_config.redis_url)
                conv_config.redis_db = redis_cfg.get('db', conv_config.redis_db)
                conv_config.redis_prefix = redis_cfg.get('prefix', conv_config.redis_prefix)
                conv_config.redis_ttl = redis_cfg.get('ttl', conv_config.redis_ttl)
                conv_config.redis_max_connections = redis_cfg.get('max_connections', conv_config.redis_max_connections)
                conv_config.redis_socket_timeout = redis_cfg.get('socket_timeout', conv_config.redis_socket_timeout)
                conv_config.redis_decode_responses = redis_cfg.get('decode_responses',
                                                                   conv_config.redis_decode_responses)
                conv_config.redis_password = redis_cfg.get('password')

            # 文件配置
            if 'file' in storage_configs:
                file_cfg = storage_configs['file']
                conv_config.file_storage_dir = file_cfg.get('storage_dir', conv_config.file_storage_dir)

        return conv_config

    @staticmethod
    def _load_rerank_config(rerank_dict: Dict[str, Any]) -> RerankConfig:
        """从字典加载 rerank 配置"""
        rerank_config = RerankConfig()
        for key, value in rerank_dict.items():
            if hasattr(rerank_config, key):
                setattr(rerank_config, key, value)
        return rerank_config

    @staticmethod
    def _load_rag_anything_config(config_dict: Dict[str, Any]) -> RAGAnythingConfig:
        rag_config = RAGAnythingConfig()

        for key, value in config_dict.items():
            if hasattr(rag_config, key):
                setattr(rag_config, key, value)
            else:
                logger.warning(f"未知的 rag_anything 配置项: {key}")

        return rag_config

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 基础配置的环境变量映射
        env_mappings = {
            "LLM_PROVIDER": "provider",
            "API_KEY": "api_key",
            "BASE_URL": "base_url",
            "LLM_MODEL": "llm_model",
            "VISION_MODEL": "vision_model",
            "EMBEDDING_MODEL": "embedding_model",
            "HOST": "host",
            "PORT": "port",
            "STATIC_BASE_URL": "static_base_url"
        }

        for env_key, attr_name in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                if attr_name in ["port", "embedding_dim"]:
                    env_value = int(env_value)
                elif attr_name in ["use_chinese_prompts"]:
                    env_value = env_value.lower() == "true"
                setattr(self, attr_name, env_value)

        # 会话配置的环境变量
        conv_env_mappings = {
            "CONVERSATION_STORAGE": "storage_backend",
            "CONVERSATION_DIR": "file_storage_dir",
            "REDIS_URL": "redis_url",
            "REDIS_DB": "redis_db",
            "REDIS_PREFIX": "redis_prefix",
            "REDIS_PASSWORD": "redis_password",
            "HISTORY_TURNS": "default_max_turns"
        }

        for env_key, attr_name in conv_env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                if attr_name in ["redis_db", "default_max_turns"]:
                    env_value = int(env_value)
                setattr(self.conversation, attr_name, env_value)

        # Rerank 配置的环境变量
        rerank_env_mappings = {
            "RERANK_ENABLED": "enabled",
            "RERANK_MODEL": "model",
            "RERANK_DEVICE": "device",
            "RERANK_TOP_K": "top_n"
        }

        for env_key, attr_name in rerank_env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                if attr_name == "enabled":
                    env_value = env_value.lower() == "true"
                elif attr_name == "top_n":
                    env_value = int(env_value)
                setattr(self.rerank, attr_name, env_value)

    @staticmethod
    def _detect_chinese_font() -> Optional[str]:
        """
        自动检测系统中文字体路径（支持Windows和Linux）
        
        RAG-Anything使用WENQUANYI_FONT_PATH环境变量，但我们也支持其他中文字体
        优先查找WenQuanYi字体，如果没有则查找其他中文字体
        """
        import platform
        import glob
        
        system = platform.system()
        logger.info(f"检测到{system}系统")
        
        try:
            # Windows系统
            if system == "Windows":
                windir = os.environ.get("WINDIR", "C:\\Windows")
                fonts_dir = os.path.join(windir, "Fonts")
                
                # 优先使用的字体列表（按优先级排序）
                preferred_fonts = [
                    "msyh.ttc",      # 微软雅黑（最常用）
                    "simsun.ttc",    # 宋体
                    "simhei.ttf",    # 黑体
                    "simkai.ttf",    # 楷体
                ]
                
                # 查找字体文件
                for font_name in preferred_fonts:
                    font_path = os.path.join(fonts_dir, font_name)
                    if os.path.exists(font_path):
                        logger.info(f"检测到Windows中文字体: {font_path}")
                        return font_path
                
                # 如果找不到优先字体，尝试查找任何中文字体
                for ext in ["*.ttc", "*.ttf"]:
                    fonts = glob.glob(os.path.join(fonts_dir, ext))
                    for font_path in fonts:
                        font_name = os.path.basename(font_path).lower()
                        if any(keyword in font_name for keyword in ["sim", "msyh", "song", "hei", "kai"]):
                            logger.info(f"检测到Windows中文字体: {font_path}")
                            return font_path
            
            # Linux系统
            elif system == "Linux":
                # Linux常见字体目录
                font_dirs = [
                    "/usr/share/fonts",           # 系统字体目录
                    "/usr/share/fonts/truetype",  # TrueType字体
                    "/usr/share/fonts/opentype",  # OpenType字体
                    "/usr/local/share/fonts",     # 本地字体目录
                    os.path.expanduser("~/.fonts"),  # 用户字体目录
                    os.path.expanduser("~/.local/share/fonts"),  # 用户本地字体
                ]
                
                # 优先查找WenQuanYi字体（RAG-Anything默认使用）
                wenquanyi_paths = [
                    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                ]
                
                for font_path in wenquanyi_paths:
                    if os.path.exists(font_path):
                        logger.info(f"检测到Linux WenQuanYi字体: {font_path}")
                        return font_path
                
                # 查找其他中文字体
                chinese_font_keywords = ["wqy", "noto-cjk", "source-han", "droid", "arphic", "ukai", "uming"]
                
                for font_dir in font_dirs:
                    if not os.path.exists(font_dir):
                        continue
                    
                    # 递归查找字体文件
                    for root, dirs, files in os.walk(font_dir):
                        for file in files:
                            if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                                file_lower = file.lower()
                                if any(keyword in file_lower for keyword in chinese_font_keywords):
                                    font_path = os.path.join(root, file)
                                    logger.info(f"检测到Linux中文字体: {font_path}")
                                    return font_path
                
                # 如果都找不到，尝试使用fc-list查找（需要fontconfig）
                try:
                    import subprocess
                    result = subprocess.run(
                        ["fc-list", ":lang=zh", "file"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout:
                        # 取第一行字体路径
                        first_line = result.stdout.strip().split('\n')[0]
                        if first_line and os.path.exists(first_line):
                            logger.info(f"通过fc-list检测到中文字体: {first_line}")
                            return first_line
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    pass
            
        except Exception as e:
            logger.warning(f"检测中文字体失败: {e}")
        
        return None

# 全局配置实例
settings = SystemConfig.load_config()

# print("\n=== 配置加载结果 ===")
# print(f"provider: {settings.provider}")
# print(f"base_url: {settings.base_url}")
# print(f"api_key: {settings.api_key[:10] + '...' if settings.api_key else 'None'}")
# print(f"llm_model: {settings.llm_model}")
# print(f"vision_model: {settings.vision_model}")
# print(f"embedding_model: {settings.embedding_model}")
# print(f"static_base_url: {settings.static_base_url}")
# print("==================\n")