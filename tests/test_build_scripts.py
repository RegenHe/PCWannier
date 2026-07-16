from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _run_build_script(script_name: str, *arguments: str):
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(REPO_ROOT / "scripts" / script_name),
            *arguments,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("script_name", ["build_nuitka.ps1", "build_pyinstaller.ps1"])
def test_build_script_dry_run_creates_nothing_and_accepts_safe_clean_target(script_name):
    target = REPO_ROOT / "dist" / f"pytest-{Path(script_name).stem}"
    assert not target.exists()

    completed = _run_build_script(
        script_name,
        "-OutDir",
        str(target),
        "-Clean",
        "-DryRun",
    )

    assert completed.returncode == 0, completed.stderr
    assert not target.exists()
    assert str(target) in completed.stdout


@pytest.mark.parametrize("script_name", ["build_nuitka.ps1", "build_pyinstaller.ps1"])
@pytest.mark.parametrize("unsafe_name", ["dist", "build"])
def test_build_script_refuses_to_clean_repository_build_roots(script_name, unsafe_name):
    completed = _run_build_script(
        script_name,
        "-OutDir",
        str(REPO_ROOT / unsafe_name),
        "-Clean",
        "-DryRun",
    )

    assert completed.returncode != 0
    assert "Refusing to clean" in completed.stderr


@pytest.mark.parametrize("script_name", ["build_nuitka.ps1", "build_pyinstaller.ps1"])
def test_build_script_refuses_external_clean_and_dry_run_does_not_create_external_output(
    script_name, tmp_path
):
    external = tmp_path / "external-output"
    rejected = _run_build_script(
        script_name,
        "-OutDir",
        str(external),
        "-Clean",
        "-DryRun",
    )
    assert rejected.returncode != 0
    assert "Refusing to clean" in rejected.stderr
    assert not external.exists()

    dry_run = _run_build_script(script_name, "-OutDir", str(external), "-DryRun")
    assert dry_run.returncode == 0, dry_run.stderr
    assert not external.exists()
