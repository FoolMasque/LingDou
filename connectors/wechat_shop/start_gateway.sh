#!/bin/bash

# 切换到脚本所在目录 (connectors/wechat_shop)
cd "$(dirname "$0")"

LOG_DIR="logs"
# 如果目录不存在则创建
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "WeChat Shop Intent Gateway - Guardian Process (Linux)"
echo "========================================================"
echo "提示: 建议使用 tmux 或 screen 运行此脚本，或者在末尾加 & 后台运行:"
echo "nohup ./start_gateway.sh > /dev/null 2>&1 &"
echo "========================================================"
echo ""

while true; do
    # 生成带时间戳的日志文件名
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/gateway_$TIMESTAMP.log"

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 启动智能网关..."
    echo "[+] 日志将持续写入: logs/ 目录下"

    conda activate ling-dou2

    # 使用 -u 参数强制不使用缓冲，实时写入日志
    python3 -u gateway.py >> "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    echo ""
    echo "[!] 网关进程已异常中断！(退出码: $EXIT_CODE)"
    echo "[!] 距离下次自动重启还有 5 秒钟... (按 Ctrl+C 彻底终止守护进程)"
    sleep 5
    echo ""
done
