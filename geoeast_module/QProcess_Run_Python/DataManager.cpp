#include "DataManager.h"

DataManager::DataManager() = default;

DataManager::~DataManager() = default;

DataManager& DataManager::getInstance()
{
    static DataManager instance;
    return instance;
}

bool DataManager::connectDataBase(const DataBaseInfo &dbInfo,
                                   const DataTreeInfo &dtInfo)
{
    // 连接 NDP 数据库
    m_boSystemRoot = std::make_shared<ngp::bo::BOSystemRoot>();
    if (!m_boSystemRoot->open(dbInfo.m_dbName.toStdString(),
                               dbInfo.m_userName.toStdString(),
                               dbInfo.m_passWord.toStdString()))
        return false;

    // 打开项目
    m_boProject = std::make_shared<ngp::bo::BOProject>(
        m_boSystemRoot->getProject(dtInfo.m_projectName.toStdString()));

    return true;
}
