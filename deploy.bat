@echo off
chcp 65001 >nul
echo === Windows版本RAG系统部署 ===

REM 获取脚本所在目录的绝对路径
set SCRIPT_DIR=%~dp0
echo 脚本目录: %SCRIPT_DIR%

REM 切换到脚本所在目录
cd /d "%SCRIPT_DIR%"
echo 当前工作目录: %cd%

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python未安装，请先安装Python
    pause
    exit /b 1
)

echo Python版本:
python --version

REM 检查config.json
if not exist "config.json" (
    echo config.json文件不存在，请先创建配置文件
    pause
    exit /b 1
)

REM 设置环境变量
set PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%
set APP_ROOT=%SCRIPT_DIR%

echo 启动前目录状态:
dir /b

echo.
echo 启动RAG服务...
echo 工作目录: %cd%
python -m api.server

echo.
echo 启动后目录状态:
dir /b

echo.
echo 部署完成！
echo 访问 http://localhost:8008/docs 查看API文档
pause