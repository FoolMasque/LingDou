# config/settings.py
"""
统一配置管理
"""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Literal


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
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
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
    score_threshold: float = 0.5
    use_fp16: bool = False
    max_length: int = 512

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
    image_storage: str = "./static/images"

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
            "llm_model": "qwen2.5-14b-instruct",
            "vision_model": "qwen2.5-vl-32b-instruct",
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
                "url": self.conversation.redis_url,
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
        config_file = Path("./config.json")
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
                        # if hasattr(config, key) and key != 'provider_configs':
                        #     setattr(config, key, value)
                        elif key == 'provider_configs':
                            # 更新提供商配置
                            config.provider_configs.update(value)
                        elif hasattr(config, key):
                            setattr(config, key, value)
                    # 更新提供商配置
                    # if 'provider_configs' in file_config:
                    #     config.provider_configs.update(file_config['provider_configs'])
                    print(f"读取配置文件完毕")
            except Exception as e:
                print(f"读取配置文件失败: {e}")
        # 2. 环境变量覆盖
        config._load_from_env()

        # 3. 根据provider应用默认配置
        if config.provider in config.provider_configs:
            provider_config = config.provider_configs[config.provider]
            for key, value in provider_config.items():
                if not getattr(config, key, None) or getattr(config, key) == getattr(cls(), key):
                    setattr(config, key, value)

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

# 全局配置实例
settings = SystemConfig.load_config()