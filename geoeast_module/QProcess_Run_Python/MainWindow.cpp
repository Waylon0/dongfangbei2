#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <iostream>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>

using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_process(new QProcess(this))
{
    ui->setupUi(this);

    connect(m_process, &QProcess::started, this, &MainWindow::slot_started);
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::slot_finished);
    connect(m_process, &QProcess::errorOccurred, this, &MainWindow::slot_errorOccurred);
    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &MainWindow::slot_readyReadStandardOutput);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &MainWindow::slot_readyReadStandardError);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButtonRun_clicked()
{
    // [1] 指定 Python 算法工作目录
    QString workingDirPath = qApp->applicationDirPath() + "/pyalgo_wzy";
    m_process->setWorkingDirectory(workingDirPath);

    // [2] 指定 Python 解释器路径
    QString pythonExePath = qgetenv("NGP") +
        QString("/support/miniconda3/envs/nv/python");

    // [3] 组装调用参数 — 从 UI 控件读取
    QString projectName = ui->lineEditProject->text();
    QString surveyName  = ui->lineEditSurvey->text();
    QString gridName    = ui->lineEditGrid->text();
    QString outputName  = ui->lineEditOutput->text();
    QString thresholdMode = ui->comboBoxThresholdMode->currentText();
    double otsuScale     = ui->doubleSpinBoxOtsuScale->value();
    double gaussianSigma = ui->doubleSpinBoxGaussianSigma->value();
    int closingRadius    = ui->spinBoxClosingRadius->value();
    int openingRadius    = ui->spinBoxOpeningRadius->value();
    double minPolygonArea = ui->doubleSpinBoxMinPolygonArea->value();
    double dpEpsilon      = ui->doubleSpinBoxDpEpsilon->value();
    QString polygonMode   = ui->comboBoxPolygonMode->currentText();

    QStringList arguments;
    arguments << "-c";
    arguments << QString(
        "import entry;"
        "entry.run('%1','%2','%3','%4','%5',%6,%7,%8,%9,%10,%11,'%12')")
        .arg(projectName).arg(surveyName).arg(gridName).arg(outputName)
        .arg(thresholdMode).arg(otsuScale).arg(gaussianSigma)
        .arg(closingRadius).arg(openingRadius).arg(minPolygonArea)
        .arg(dpEpsilon).arg(polygonMode);

    cout << "[" << __FILE__ << ":" << __LINE__ << "] "
         << pythonExePath.toStdString() << " " << arguments.at(1).toStdString()
         << endl;

    m_process->start(pythonExePath, arguments);
}

void MainWindow::slot_started()
{
    qDebug() << "[MainWindow] Python 进程已启动";
    ui->textEditLog->append("[系统] 算法进程已启动...");
}

void MainWindow::slot_finished(int exitCode, QProcess::ExitStatus exitStatus)
{
    qDebug() << "[MainWindow] 进程结束: exitCode=" << exitCode;
    if (exitCode != 0) {
        QByteArray qba = m_process->readAllStandardError();
        QString str = QString::fromUtf8(qba).trimmed();
        ui->textEditLog->append("[错误] " + str);
    } else {
        ui->textEditLog->append("[系统] 算法执行完成");
    }
}

void MainWindow::slot_errorOccurred(QProcess::ProcessError error)
{
    qDebug() << "[MainWindow] 进程错误: " << error;
    ui->textEditLog->append("[错误] 进程异常: " + QString::number(error));
}

void MainWindow::slot_readyReadStandardOutput()
{
    m_process->setCurrentReadChannel(QProcess::StandardOutput);
    QByteArray qba = m_process->readLine();
    QString str = QString::fromUtf8(qba).trimmed();

    // 解析 JSON 结果
    if (str.contains("RESULT_JSON:")) {
        QString jsonStr = str.mid(str.indexOf("{"));
        QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
        if (!doc.isNull()) {
            QJsonObject obj = doc.object();
            ui->lineEditFaultCount->setText(
                QString::number(obj["count"].toInt()));
            ui->lineEditElapsed->setText(
                QString::number(obj["elapsed"].toDouble(), 'f', 2) + "s");
            ui->lineEditTotalArea->setText(
                QString::number(obj["total_area"].toDouble(), 'f', 1));
        }
    }

    ui->textEditLog->append(str);
}

void MainWindow::slot_readyReadStandardError()
{
    m_process->setCurrentReadChannel(QProcess::StandardError);
    QByteArray qba = m_process->readLine();
    QString str = QString::fromUtf8(qba).trimmed();
    ui->textEditLog->append("[stderr] " + str);
}

void MainWindow::on_pushButtonStop_clicked()
{
    m_process->write("exit\n");
    m_process->waitForBytesWritten();
    ui->textEditLog->append("[系统] 已发送停止信号");
}
