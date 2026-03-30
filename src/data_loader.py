import logging
from pathlib import Path
import requests
import zipfile
import io
import pandas as pd

logging.basicConfig(level=logging.INFO)

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_URL = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip"
INFERENCE_URL = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_inference_dataset.zip"


def download_and_extract_csv(url: str, output_name: str):
    output_path = RAW_DATA_DIR / f"{output_name}.csv"

    logging.info(f"Downloading dataset from {url}")
    response = requests.get(url)
    response.raise_for_status()

    logging.info("Extracting ZIP file...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for file in z.namelist():
            if file.endswith(".csv"):
                logging.info(f"Found CSV inside zip: {file}")
                with z.open(file) as f:
                    df = pd.read_csv(f)
                    df.to_csv(output_path, index=False)
                    logging.info(f"Saved CSV to {output_path}")
                    return

    raise ValueError("ZIP archive does not contain a CSV file.")


if __name__ == "__main__":
    logging.info("Starting data loading process...")

    download_and_extract_csv(TRAIN_URL, "train")
    download_and_extract_csv(INFERENCE_URL, "inference")

    logging.info("All datasets downloaded and prepared.")
