#pragma once

#include <QMainWindow>
#include <QProcess>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private Q_SLOTS:
    void on_pushButtonRun_clicked();
    void on_pushButtonStop_clicked();

    // QProcess 监控槽
    void slot_started();
    void slot_finished(int exitCode, QProcess::ExitStatus exitStatus);
    void slot_errorOccurred(QProcess::ProcessError error);
    void slot_readyReadStandardOutput();
    void slot_readyReadStandardError();

private:
    Ui::MainWindow *ui;
    QProcess *m_process;
};
