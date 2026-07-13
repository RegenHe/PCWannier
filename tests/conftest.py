from pathlib import Path

def pytest_configure(config) -> None:
    (Path(config.rootpath) / "tmp").mkdir(exist_ok=True)
