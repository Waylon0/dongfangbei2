#include "Application.h"

int main(int argc, char *argv[])
{
    Application app(argc, argv);

    if (!app.initializeDataManager())
        return -1;

    app.showMainWindow();
    return app.exec();
}
