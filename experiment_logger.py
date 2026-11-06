
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "experiment_data"

RESPONSES_FILE = DATA_DIR / "responses.json"
ERRORS_FILE = DATA_DIR / "errors.json"


def append_json_record(file_path, record):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                records = json.load(file)

            if not isinstance(records, list):
                records = []

        except json.JSONDecodeError:
            records = []

    else:
        records = []

    records.append(record)

    temp_path = file_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )

    temp_path.replace(file_path)


def save_response(record):
    append_json_record(RESPONSES_FILE, record)


def save_error(record):
    append_json_record(ERRORS_FILE, record)
