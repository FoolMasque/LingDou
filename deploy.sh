#!/bin/bash

# 获取脚本所在的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "脚本目录: $SCRIPT_DIR"

# 切换到脚本所在目录
cd "$SCRIPT_DIR"
echo "当前工作目录: $(pwd)"

echo "=== 生产环境RAG系统部署 ==="


# 检查config.json
if [ ! -f "config.json" ]; then
    echo "config.json文件不存在，请先创建配置文件"
    exit 1
fi

echo "启动RAG服务..."
echo "工作目录设置为: $SCRIPT_DIR"

# 确保在正确目录下启动Python应用
cd "$SCRIPT_DIR"
python -m api.server

echo "部署完成！"
echo "访问 http://localhost:8008/docs 查看API文档"