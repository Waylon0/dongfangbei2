"""PyBO connectivity verification script.

Verification items:
    1. PyBO import
    2. Connect to PostgreSQL database (ndp)
    3. List projects, surveys, map grids, fault polygons

Notes:
    - PyDataProviderConfig.Load() corrupts NDP internal state, causing
      PyBOSystemRoot.instance() to crash. Call instance() before Load().
    - Only ndp (PostgreSQL) is usable; ndpsqlite has DLL compatibility issues.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pybo_importer import pybo  # _setup() runs automatically on import


def step1_import():
    print('=' * 60, flush=True)
    print('Step 1: PyBO Import Verification', flush=True)
    print('=' * 60, flush=True)
    print(f'NGP     = {os.environ.get("NGP", "(not set)")}', flush=True)
    print(f'GEOEAST = {os.environ.get("GEOEAST", "(not set)")}', flush=True)
    print(f'NDPDIR  = {os.environ.get("NDPDIR", "(not set)")}', flush=True)
    print(f'PyBO module: {pybo.__file__}', flush=True)
    attrs = ['PyBOSystemRoot', 'PyBOMapGrid', 'PyBOFaultPolygon',
             'PyBOProject', 'PyBOSurvey']
    for a in attrs:
        print(f'  {a}: {"OK" if hasattr(pybo, a) else "MISSING"}', flush=True)
    print(flush=True)


def step2_connect(db_name='ndp'):
    print('=' * 60, flush=True)
    print(f'Step 2: Connect to "{db_name}"', flush=True)
    print('=' * 60, flush=True)

    root = pybo.PyBOSystemRoot.instance(db_name)
    print(f'  SystemRoot: {root.getName()}', flush=True)

    projects = root.listProject()
    print(f'  Projects ({len(projects)}):', flush=True)
    for p in projects:
        print(f'    - {p.getName()}', flush=True)
    print(flush=True)
    return root, projects


def step3_explore(root, project_name):
    print('=' * 60, flush=True)
    print(f'Step 3: Explore "{project_name}"', flush=True)
    print('=' * 60, flush=True)

    project = root.getProject(project_name)
    if project is None:
        print(f'  FAILED: cannot open project "{project_name}"', flush=True)
        return

    surveys = project.listSurvey()
    print(f'  Surveys ({len(surveys)}):', flush=True)
    for s in surveys:
        print(f'    - {s.getName()}', flush=True)

    try:
        grids = project.listMapGrid()
        print(f'  MapGrids ({len(grids)}):', flush=True)
        for g in grids:
            print(f'    - {g.getName()}', flush=True)
    except Exception as e:
        print(f'  MapGrids: error ({e})', flush=True)

    try:
        faults = project.listFaultPolygon()
        print(f'  FaultPolygons ({len(faults)}):', flush=True)
        for f in faults:
            print(f'    - {f.getName()}', flush=True)
    except Exception as e:
        print(f'  FaultPolygons: error ({e})', flush=True)

    for s in surveys[:3]:
        try:
            survey = project.getSurvey(s.getName())
            binset = survey.getDefaultBinsetInfo()
            il = f'{binset.getMinInlineNo()}-{binset.getMaxInlineNo()}'
            xl = f'{binset.getMinXlineNo()}-{binset.getMaxXlineNo()}'
            print(f'  Survey [{s.getName()}]: Inline[{il}] Xline[{xl}]',
                  flush=True)
        except Exception as e:
            print(f'  Survey [{s.getName()}]: error ({e})', flush=True)

    print(flush=True)
    return project


def step_info():
    print('=' * 60, flush=True)
    print('Info: Database Configuration', flush=True)
    print('=' * 60, flush=True)
    try:
        dp = pybo.PyDataProviderConfig()
        dbs = pybo.PyDbsConf()
        if dbs.Load() and dp.Load():
            print(f'  Available databases: {dp.listofDpnames()}', flush=True)
            print(f'  Default database:    {dp.getDefaultDataprovider()}',
                  flush=True)
        else:
            print('  (config load failed)', flush=True)
    except Exception as e:
        print(f'  (config load error: {e})', flush=True)
    print(flush=True)


def main():
    print('PyBO Connectivity Verification', flush=True)
    print('=' * 60, flush=True)
    print(flush=True)

    step1_import()

    db_name = os.environ.get('PYBO_DB', 'ndp')
    root, projects = step2_connect(db_name)

    if projects:
        step3_explore(root, projects[0].getName())

    step_info()

    print('=' * 60, flush=True)
    print('Verification complete', flush=True)


if __name__ == '__main__':
    main()
