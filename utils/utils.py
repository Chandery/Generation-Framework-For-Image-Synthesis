import os
import csv

def save_csv(metrics, path):
    with open(os.path.join(path, "metrics.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["-"*100])
        writer.writerow(["Metrics"])
        writer.writerow(["-"*100])
        for key, value in metrics.items():
            try:
                writer.writerow([f"{key}: {value[0]:.4f} ± {value[1]:.4f}"])
            except:
                writer.writerow([f"{key}: {value:.4f}"])
        writer.writerow(["-"*100])

def print_metrics(metrics):
    print("-"*100)
    print("Metrics")
    print("-"*100)
    for key, value in metrics.items():
        try:
            print(f"{key}: {value[0]:.4f} ± {value[1]:.4f}")
        except:
            print(f"{key}: {value:.4f}")
    print("-"*100)