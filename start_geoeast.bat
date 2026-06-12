@echo off
chcp 65001 >nul
REM GeoEast Launcher

set "NGP=D:\GeoEastRC\GeoEast-RC-V2.2"
set "GEOEAST=D:\GeoEastRC\iEcoV2.1"
set "PG_HOME=D:\GeoEastRC\GeoEast-RC-V2.2"

set "PATH=%NGP%\bin;%NGP%\libso\ndp;%NGP%\libso\ndp\plugins;%NGP%\libso\ndp\plugins\drivers;%NGP%\libso\common;%NGP%\libso\iecopy;%GEOEAST%\bin;%GEOEAST%\bin\common;%GEOEAST%\support\pg\bin;%GEOEAST%\support\pg\lib;%PATH%"

echo ========================================
echo  GeoEast Environment
echo ========================================
echo NGP      = %NGP%
echo GEOEAST  = %GEOEAST%
echo PG_HOME  = %PG_HOME%
echo.

if not exist "%NGP%\bin\geoeast.exe" (
    echo [ERROR] geoeast.exe not found
    pause
    exit /b 1
)
echo [OK] geoeast.exe found

if not exist "%NGP%\hostname.pg" (
    echo [WARN] hostname.pg not found
) else (
    echo [OK] hostname.pg found
)

echo.
echo Starting geoeast.exe ...

cd /d "%NGP%\bin"
start "" geoeast.exe

echo [OK] Done
pause
