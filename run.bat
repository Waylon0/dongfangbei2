@echo off
chcp 65001 >nul
echo ============================================
echo   断层多边形自动追踪系统 v2.0
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

:: 检查依赖
echo [1/3] 检查依赖库...
pip show PyQt5 >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖库...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

:: 运行主程序
echo [2/3] 启动应用程序...
python app.py

echo.
echo [3/3] 程序已退出。
pause
