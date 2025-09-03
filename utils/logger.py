# utils/logger.py
"""
日志配置
"""
import logging
import sys
from pathlib import Path
from config.settings import settings


def setup_logger(name: str = "rag_system", level: str = "INFO") -> logging.Logger:
    """设置日志系统"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    # log_dir = Path("./logs")
    settings.log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(settings.log_dir / "rag_system.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
