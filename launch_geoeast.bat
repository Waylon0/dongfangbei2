@echo off
REM GeoEast 启动脚本 — 设置 DLL 路径后启动 geomanager.exe

set NGP=D:\GeoEastRC\GeoEast-RC-V2.2
set GEOEAST=D:\GeoEastRC\iEcoV2.1
set PG_HOME=%NGP%

REM 将 GeoEast 所需 DLL 目录加入 PATH
set PATH=%NGP%\bin;%NGP%\libso\ndp;%NGP%\libso\ndp\plugins;%NGP%\libso\ndp\plugins\drivers;%NGP%\libso\common;%NGP%\libso\iecopy;%GEOEAST%\bin;%GEOEAST%\bin\common;%GEOEAST%\support\pg\bin;%GEOEAST%\support\pg\lib;%PATH%

REM 用 geomanager.exe 启动（guide.pdf 里的正确方式）
start "" "%GEOEAST%\bin\common\geomanager.exe"
