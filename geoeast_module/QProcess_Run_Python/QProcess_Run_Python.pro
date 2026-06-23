QT += core gui widgets

TEMPLATE = app
TARGET = FaultPolygonTracker
CONFIG += c++17

# GeoEast 框架依赖（编译时需配置 NGP 环境）
NGP_DIR = $$(NGP)
INCLUDEPATH += $$NGP_DIR/include
INCLUDEPATH += $$NGP_DIR/include/iecopy
INCLUDEPATH += $$NGP_DIR/include/ndp

LIBS += -L$$NGP_DIR/libso/ndp
LIBS += -L$$NGP_DIR/libso/iecopy
LIBS += -L$$NGP_DIR/libso/common

SOURCES += \
    Application.cpp \
    DataManager.cpp \
    LoginDialog.cpp \
    MainWindow.cpp \
    main.cpp

HEADERS += \
    Application.h \
    DataManager.h \
    LoginDialog.h \
    MainWindow.h

FORMS += \
    LoginDialog.ui \
    MainWindow.ui

# 输出到 GeoEast 模块目录
DESTDIR = $$PWD/../bin
