@echo off
chcp 65001 >nul
set "NGP=D:\GeoEastRC\GeoEast-RC-V2.2"
set "GEOEAST=D:\GeoEastRC\iEcoV2.1"
set "PG_HOME=D:\GeoEastRC\GeoEast-RC-V2.2"
set "PATH=%NGP%\bin;%NGP%\libso\ndp;%NGP%\libso\ndp\plugins;%NGP%\libso\ndp\plugins\drivers;%NGP%\libso\common;%NGP%\libso\iecopy;%GEOEAST%\bin;%GEOEAST%\bin\common;%GEOEAST%\support\pg\bin;%GEOEAST%\support\pg\lib;%PATH%"
set "PYTHON=%GEOEAST%\..\support\miniconda3\envs\nv\python.exe"

echo ========================================
echo  GeoEast Integration Demo
echo  Python: %PYTHON%
echo ========================================
echo.

"%PYTHON%" run_geoeast.py --project DFB_S7 --file references\test.dat --output fault_polygons --visualize

echo.
echo ========================================
echo  Done. Check fault_polygons_overlay.png
echo ========================================
pause
