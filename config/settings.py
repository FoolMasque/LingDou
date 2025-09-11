# config/settings.py
"""
统一配置管理
"""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class SystemConfig:
    """
    系统配置
    """
    # API配置
    provider: str = "zhipu" #"openai"
    api_key: str = "e06326b43c6e469e863ccbbb60f1ee6a.iBlzS0xRSXwoPnPF" #""
    base_url: str = "https://open.bigmodel.cn/api/paas/v4" #""

    # 模型配置
    llm_model: str = "glm-4" #"gpt-4o-mini"
    vision_model: str = "glm-4v"# "gpt-4o"
    embedding_model: str = "embedding-2"# "text-embedding-3-large"
    embedding_dim: int = 1024 #3072
    max_token_size: int = 2048 # 8192

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8008
    static_base_url: str = ""

    # 存储配置
    working_dir: str = "../rag_storage"
    image_storage: str = "../static/images"

    log_dir: Path = Path("../logs")

    # 业务配置
    use_chinese_prompts: bool = True
    default_language: str = "zh-CN"
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    max_dimension: int = 512 # 图片压缩最长边像素
    jpeg_quality: int = 85 # 图片质量百分比

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

    @classmethod
    def load_config(cls) -> 'SystemConfig':
        """
        加载配置：环境变量 > config.json > 默认值
        """
        config = cls()

        # 1. 从config.json读取
        config_file = Path("../config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # 更新基础配置
                    for key, value in file_config.items():
                        if hasattr(config, key) and key != 'provider_configs':
                            setattr(config, key, value)
                    # 更新提供商配置
                    if 'provider_configs' in file_config:
                        config.provider_configs.update(file_config['provider_configs'])
                        print(f"读取配置文件完毕")
            except Exception as e:
                print(f"读取配置文件失败: {e}")

        # 2. 环境变量覆盖
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
                setattr(config, attr_name, env_value)

        # 3. 根据provider应用默认配置
        if config.provider in config.provider_configs:
            provider_config = config.provider_configs[config.provider]
            for key, value in provider_config.items():
                if not getattr(config, key, None) or getattr(config, key) == getattr(cls(), key):
                    setattr(config, key, value)

        return config

# 全局配置实例
settings = SystemConfig.load_config()