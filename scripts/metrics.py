import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.metrics import MetricsCalculator
from utils.utils import save_csv, print_metrics

data_path = "/disk/cdy/Generation-Framework-for-Medical-Image/logs/cyclegan/IXI/test_CycleGAN-2025-06-06/09-28-04"
metrics_calculator = MetricsCalculator(data_path, max_dataset_size=100000, direction="AtoB")
metrics = metrics_calculator.calculate_all_metrics(verbose=True)
save_csv(metrics, data_path)
print_metrics(metrics)