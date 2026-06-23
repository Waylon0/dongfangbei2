#include "Application.h"
#include "DataManager.h"
#include "LoginDialog.h"
#include "MainWindow.h"

Application::Application(int &argc, char **argv)
    :
#ifdef USE_FW_APP_EX
      FW_ApplicationEx(argc, argv)
#else
      QApplication(argc, argv)
#endif
{
    // 连接 GeoEast 框架通讯
#ifdef USE_FW_APP_EX
    setOrganizationDomain("东方杯V2");
#endif
}

bool Application::initializeDataManager()
{
    // 弹出登录对话框
    LoginDialog dlg;
    if (dlg.exec() != QDialog::Accepted)
        return false;

    DataBaseInfo dbInfo;
    dbInfo.m_dbName   = dlg.getDbName();
    dbInfo.m_userName = dlg.getUserName();
    dbInfo.m_passWord = dlg.getPassword();

    DataTreeInfo dtInfo;
    dtInfo.m_projectName = dlg.getProjectName();
    dtInfo.m_surveyName  = dlg.getSurveyName();

    return DataManager::getInstance().connectDataBase(dbInfo, dtInfo);
}

void Application::showMainWindow()
{
    MainWindow *w = new MainWindow;
    w->show();
}
