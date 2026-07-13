param(
    [string]$Python = "python",
    [string]$OutDir = "dist\pyinstaller",
    [string]$Name = "pcwannier",
    [switch]$OneFile,
    [switch]$WithNumba,
    [switch]$Clean,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Clean -and (Test-Path $OutDir)) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$entry = Join-Path $repoRoot "scripts\pyinstaller_entry.py"
$workDir = Join-Path $repoRoot "build\pyinstaller"

$pyinstallerOptions = @(
    "-m", "PyInstaller",
    "--clean",
    "--noconfirm",
    "--name", $Name,
    "--console",
    "--distpath", $OutDir,
    "--workpath", $workDir,
    "--specpath", $workDir,
    "--collect-submodules", "scipy",
    "--collect-data", "matplotlib",
    "--collect-data", "pcwannier.symmetries"
)

if ($OneFile) {
    $pyinstallerOptions = $pyinstallerOptions + @("--onefile")
}

if ($WithNumba) {
    $pyinstallerOptions = $pyinstallerOptions + @(
        "--collect-submodules", "numba",
        "--collect-submodules", "llvmlite"
    )
}

$argsList = $pyinstallerOptions + @($entry)

Write-Host "Repository: $repoRoot"
Write-Host "Python: $Python"
Write-Host "Output: $OutDir"
Write-Host "Command: $Python $($argsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run only; PyInstaller was not executed."
    exit 0
}

$timer = [Diagnostics.Stopwatch]::StartNew()
& $Python @argsList
$timer.Stop()

$exe = Get-ChildItem -Path $OutDir -Recurse -Filter "$Name.exe" | Select-Object -First 1

Write-Host ""
if ($exe) {
    Write-Host "Build complete: $($exe.FullName)"
} else {
    Write-Host "Build complete. Check output directory: $OutDir"
}
Write-Host ("Build time: {0:hh\:mm\:ss\.fff}" -f $timer.Elapsed)
