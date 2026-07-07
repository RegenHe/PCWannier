param(
    [string]$Python = "python",
    [string]$OutDir = "dist\nuitka",
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

$entry = Join-Path $repoRoot "scripts\nuitka_entry.py"

$nuitkaOptions = @(
    "-m", "nuitka",
    "--standalone",
    "--assume-yes-for-downloads",
    "--enable-plugin=matplotlib",
    "--include-package=pcwannier",
    "--output-dir=$OutDir",
    "--output-filename=$Name.exe"
)

if ($OneFile) {
    $nuitkaOptions = @("-m", "nuitka", "--onefile") + $nuitkaOptions[2..($nuitkaOptions.Count - 1)]
}

if ($WithNumba) {
    $nuitkaOptions = $nuitkaOptions + @("--include-package=numba", "--include-package=llvmlite")
}

$argsList = $nuitkaOptions + @($entry)

Write-Host "Repository: $repoRoot"
Write-Host "Python: $Python"
Write-Host "Output: $OutDir"
Write-Host "Command: $Python $($argsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run only; Nuitka was not executed."
    exit 0
}

& $Python @argsList

$exe = Get-ChildItem -Path $OutDir -Recurse -Filter "$Name.exe" | Select-Object -First 1

Write-Host ""
if ($exe) {
    Write-Host "Build complete: $($exe.FullName)"
} else {
    Write-Host "Build complete. Check output directory: $OutDir"
}

