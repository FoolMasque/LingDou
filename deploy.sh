#!/bin/bash
# shellcheck disable=SC2164
cd "$(dirname "$0")"
echo "=== 生产环境RAG系统部署 ==="

# 1. 创建目录结构
#mkdir -p config core api utils static/images logs
#
## 2. 安装依赖
#pip install -r requirements.txt

# 3. 设置环境变量（根据实际情况修改）
export API_KEY="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"
export LLM_PROVIDER="openai"  # 或 "zhipu", "deepseek"
export STATIC_BASE_URL="http://localhost:8008"
export HOST="0.0.0.0"
export PORT="8008"

# 4. 启动服务
echo "启动RAG服务..."
python -m api.server

echo "部署完成！"
echo "访问 http://localhost:8008/docs 查看API文档"