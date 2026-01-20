"""
Notebook Execution Pipeline

Runs all project notebooks in the correct order with structured logging.

Use cases:
- Reproducibility
- Model retraining pipelines
- CI/CD automation
"""

import subprocess
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import List


class NotebookRunner:
    """
    Executes Jupyter notebooks sequentially using nbconvert.
    """

    def __init__(self, notebooks: List[str], log_dir: str = "logs") -> None:
        self.notebooks = notebooks
        self.log_dir = Path(log_dir)
        self.logger = self._setup_logger()

    # Logger Setup
    def _setup_logger(self) -> logging.Logger:
        self.log_dir.mkdir(exist_ok=True)
        log_file = self.log_dir / "notebooks_runner.log"

        logger = logging.getLogger("notebook_runner")
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers in interactive runs
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=5
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def run(self) -> None:
        """
        Run all notebooks in sequence.
        """
        self.logger.info("Starting notebook execution pipeline")

        for nb in self.notebooks:
            self._validate_notebook(nb)
            self._run_single_notebook(nb)

        self.logger.info("All notebooks executed successfully!")
        
    def _validate_notebook(self, nb_path: str) -> None:
        if not Path(nb_path).exists():
            self.logger.error(f"Notebook not found: {nb_path}")
            sys.exit(1)

    def _run_single_notebook(self, nb_path: str) -> None:
        self.logger.info(f"Running notebook: {nb_path}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                nb_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.logger.error(f"Notebook failed: {nb_path}")
            self.logger.error(result.stderr)
            sys.exit(1)

        self.logger.info(f"Completed notebook: {nb_path}")

def main() -> None:
    notebooks = [
        "notebooks/data_cleaning_feature_engineering.ipynb",
        "notebooks/flight_delay_eda.ipynb",
        "notebooks/flight_delay_ml_modeling.ipynb",
    ]

    runner = NotebookRunner(notebooks)
    runner.run()


if __name__ == "__main__":
    main()