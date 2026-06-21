import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TEST_RUNS = 5
FINAL_RUNS = 30

TOTAL_PHASES = {
    'test': {'runs': TEST_RUNS, 'phase_dir': 'test_runs'},
    'final': {'runs': FINAL_RUNS, 'phase_dir': 'final_runs'},
}

DATASET_LIST = ['pima', 'creditcard', 'tcga']
METHOD_LIST = ['dynesty', 'ultranest']

METHOD_CONFIG = {
    'dynesty': {
        'pima':       {'nlive': 150, 'sample': 'rwalk', 'walks': 30, 'dlogz': 0.5, 'maxcall': 1000000},
        'creditcard': {'nlive': 400, 'sample': 'rwalk', 'walks': 40, 'dlogz': 0.5, 'maxcall': 3000000},
        'tcga':       {'nlive': 250, 'sample': 'rslice',               'dlogz': 0.5, 'maxcall': 8000000},
    },
    'ultranest': {
        'pima': {'min_live': 400, 'min_ess': 400},
        'creditcard': {'min_live': 600, 'min_ess': 400},
        'tcga': {'min_live': 800, 'min_ess': 400},
    },
}
