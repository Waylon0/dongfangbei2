@echo off
chcp 65001 >nul
set "NGP=D:\GeoEastRC\GeoEast-RC-V2.2"
set "GEOEAST=D:\GeoEastRC\iEcoV2.1"
set "PG_HOME=%NGP%"
set "PATH=%NGP%\bin;%NGP%\libso\ndp;%NGP%\libso\ndp\plugins;%NGP%\libso\ndp\plugins\drivers;%NGP%\libso\common;%NGP%\libso\iecopy;%GEOEAST%\bin;%GEOEAST%\bin\common;%GEOEAST%\support\pg\bin;%GEOEAST%\support\pg\lib;%PATH%"
set "LIC_CERT_PATH=%GEOEAST%\resource\standardinfo\license"
set "PYTHON=D:\GeoEastRC\support\miniconda3\envs\nv\python.exe"

echo GeoEast Fault Polygon Pipeline
echo ================================
echo Python: %PYTHON%
echo.

"%PYTHON%" setup_geoeast_project.py

echo.
echo Done. Open GeoEast - File - Open Project - DFB_S7_WZY
pause
