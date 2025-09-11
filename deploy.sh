#!/bin/bash

# RAG系统简化启动脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 项目配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENTRY="api/server.py"
PID_FILE="rag_system.pid"
LOG_FILE="deploy.log"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建必要目录
create_directories() {
    local dirs=("logs" "rag_storage" "static/images")
    for dir in "${dirs[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${dir}"
    done
}

# 检查端口占用
check_port() {
    local port=${1:-8008}
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        log_warn "端口 $port 被进程 $pid 占用，正在终止..."
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

# 启动服务
start_service() {
    log_info "启动 RAG 系统..."
    
    cd "${PROJECT_ROOT}"
    create_directories
    check_port
    
    # 检查配置文件
    if [[ -f "config.json" ]]; then
        log_info "✅ 找到配置文件: config.json"
    else
        log_warn "⚠️  未找到配置文件: config.json"
    fi
    
    # 启动服务
    nohup python3 -u "${PYTHON_ENTRY}" > "${LOG_FILE}" 2>&1 < /dev/null &
    local pid=$!
    disown
    echo $pid > "${PID_FILE}"
    
    sleep 3
    
    if kill -0 $pid 2>/dev/null; then
        log_info "✅ 服务启动成功 (PID: $pid)"
        log_info "������ 访问地址: http://localhost:8008/docs"
        log_info "������ 健康检查: http://localhost:8008/health"
        log_info "������ 查看日志: ./deploy.sh logs"
    else
        log_error "❌ 服务启动失败，请查看日志: ${LOG_FILE}"
        rm -f "${PID_FILE}"
        exit 1
    fi
}

# 停止服务
stop_service() {
    log_info "停止 RAG 系统..."
    
    if [[ -f "${PID_FILE}" ]]; then
        local pid=$(cat "${PID_FILE}")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            sleep 2
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid
            fi
            log_info "✅ 服务已停止"
        else
            log_warn "进程不存在，清理PID文件"
        fi
        rm -f "${PID_FILE}"
    else
        log_warn "服务未运行"
    fi
}

# 重启服务
restart_service() {
    stop_service
    sleep 1
    start_service
}

# 查看状态
status_service() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid=$(cat "${PID_FILE}")
        if kill -0 $pid 2>/dev/null; then
            log_info "✅ 服务运行中 (PID: $pid)"
            log_info "������ 访问地址: http://localhost:8008/docs"
        else
            log_warn "进程已停止，清理PID文件"
            rm -f "${PID_FILE}"
        fi
    else
        log_info "❌ 服务未运行"
    fi
}

# 查看日志
show_logs() {
    if [[ -f "${LOG_FILE}" ]]; then
        local lines=${1:-50}
        if [[ "$lines" == "live" ]]; then
            log_info "实时日志 (Ctrl+C 退出):"
            echo "----------------------------------------"
            tail -f "${LOG_FILE}"
        else
            log_info "最近 $lines 行日志:"
            echo "----------------------------------------"
            tail -n $lines "${LOG_FILE}"
        fi
    else
        log_warn "日志文件不存在"
    fi
}

# 主函数
main() {
    cd "${PROJECT_ROOT}"
    
    case ${1:-start} in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            status_service
            ;;
        logs)
            show_logs "$2"
            ;;
        *)
            echo "用法: ./deploy.sh [command]"
            echo "  start        启动服务 (默认)"
            echo "  stop         停止服务"
            echo "  restart      重启服务"
            echo "  status       查看状态"
            echo "  logs [n]     查看最近n行日志 (默认50行)"
            echo "  logs live    查看实时日志"
            exit 1
            ;;
    esac
}

main "$@"