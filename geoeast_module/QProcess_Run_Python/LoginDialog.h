#pragma once

#include <QDialog>

namespace Ui {
class LoginDialog;
}

class LoginDialog : public QDialog
{
    Q_OBJECT

public:
    explicit LoginDialog(QWidget *parent = nullptr);
    ~LoginDialog();

    QString getDbName() const;
    QString getUserName() const;
    QString getPassword() const;
    QString getProjectName() const;
    QString getSurveyName() const;

private:
    Ui::LoginDialog *ui;
};
