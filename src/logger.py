import csv
import os
from datetime import datetime
from typing import Dict, Any


class CSVLogger:
    def __init__(self, log_dir="logs", filename=None, fieldnames=None):
        os.makedirs(log_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"train_log_{timestamp}.csv"

        self.log_path = os.path.join(log_dir, filename)

        # Explicit schema (must be fixed per run)
        self.fieldnames = list(fieldnames) if fieldnames is not None else [
            "epoch", "phase", "loss", "recall@20", "ndcg@20", "accuracy"
        ]

        self._init_file()

    def _init_file(self):
        """
        Initialize CSV file with header if it does not exist.
        """
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, metrics: Dict[str, Any]):
        """
        Write a row to the CSV log file.
        - Missing fields are filled with None
        - Extra fields are ignored (safe for schema evolution)
        """
        # Normalize row to fixed schema
        row = {key: metrics.get(key, None) for key in self.fieldnames}

        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
                extrasaction="ignore"  # ignore unexpected keys safely
            )
            writer.writerow(row)
