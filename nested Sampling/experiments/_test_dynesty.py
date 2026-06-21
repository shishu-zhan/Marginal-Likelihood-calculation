import sys, os, warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = r'D:\homeworkcode\统计计算\Marginal-Likelihood-calculation'
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.data_loader import load_dataset, get_param_names
from src.dynesty_runner import run_dynesty
from src.io_utils import save_run, save_summary

X, y, info = load_dataset('pima')
ndim = info['ndim']

print(f'Testing dynesty on pima (d={ndim})...')
metrics = run_dynesty(X, y, ndim, nlive=300, sample='rwalk', walks=25, dlogz=0.5, seed=42)
print(f'Results: logZ={metrics["logz"]:.4f}, logZerr={metrics["logzerr"]:.4f}, H={metrics["H"]:.4f}, runtime={metrics["runtime"]:.1f}s, ncall={metrics["ncall"]}')

save_run('dynesty', 'pima', 0, metrics, phase='test_runs')
save_summary('dynesty', 'pima', [metrics], phase='test_runs')
print('Dynesty test passed!')
