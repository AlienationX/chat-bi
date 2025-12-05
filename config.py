from pathlib import Path

BASE_PATH = Path(__file__).parent
TEMP_PATH = f"{BASE_PATH}/temp"


class Config:
    PYTHON_PATH = f"{BASE_PATH}/.venv/bin/python"
    PYTHON_CODE_PATH = f"{BASE_PATH}/temp/scripts"
