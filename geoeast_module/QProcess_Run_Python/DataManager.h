#pragma once

#include <memory>
#include "bosystemroot.h"
#include "boproject.h"
#include "bosurvey.h"
#include "Application.h"

class DataManager
{
    Q_DECLARE_TR_FUNCTIONS(DataManager)

public:
    ~DataManager();
    DataManager(const DataManager &) = delete;
    DataManager& operator=(const DataManager &) = delete;

    static DataManager& getInstance();
    bool connectDataBase(const DataBaseInfo &dbInfo, const DataTreeInfo &dtInfo);

private:
    DataManager();

    std::shared_ptr<ngp::bo::BOSystemRoot> m_boSystemRoot;
    std::shared_ptr<ngp::bo::BOProject> m_boProject;
    std::shared_ptr<ngp::bo::BOSurvey> m_boSurvey;
};
