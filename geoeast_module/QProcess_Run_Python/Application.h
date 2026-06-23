#pragma once

#include <QApplication>

#ifdef USE_FW_APP_EX
#include "fw_applicationex.h"
#endif

class DataBaseInfo
{
public:
    QString m_dbName;
    QString m_userName;
    QString m_passWord;
};

class DataTreeInfo
{
public:
    QString m_projectName;
    QString m_surveyName;
    QString m_lineName;
    QString m_dataName;
    QString m_datasetName;
};

#ifdef USE_FW_APP_EX
class Application : public FW_ApplicationEx
#else
class Application : public QApplication
#endif
{
    Q_OBJECT

public:
    Application(int &argc, char **argv);
    bool initializeDataManager();
    void showMainWindow();
};
