#include "LoginDialog.h"
#include "ui_LoginDialog.h"

LoginDialog::LoginDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::LoginDialog)
{
    ui->setupUi(this);
    // 默认值
    ui->lineEditDbName->setText("ndp");
    ui->lineEditUserName->setText("admin1");
    ui->lineEditPassword->setText("admin1");
    ui->lineEditProject->setText("DFB_S7_WZY");
    ui->lineEditSurvey->setText("survey1");
}

LoginDialog::~LoginDialog()
{
    delete ui;
}

QString LoginDialog::getDbName() const    { return ui->lineEditDbName->text(); }
QString LoginDialog::getUserName() const  { return ui->lineEditUserName->text(); }
QString LoginDialog::getPassword() const  { return ui->lineEditPassword->text(); }
QString LoginDialog::getProjectName() const { return ui->lineEditProject->text(); }
QString LoginDialog::getSurveyName() const  { return ui->lineEditSurvey->text(); }
